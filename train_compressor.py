"""
Symbolic Compression Training Loop

This script implements a multi-stage GRPO training pipeline for symbolic compression and decompression
using the verifiers framework. Each training stage uses specific prompt templates with variable tags.

=== TRAINING STAGES & PROMPTS ===

Stage 1: Identity (α=0.005, β=3.0) - Focus on fidelity
├── prompts/identity_compression.txt     → {content}
├── prompts/identity_decompression.txt   → {compressed}
└── prompts/evaluation.txt               → {original}, {decompressed}

Stage 2: Structured (α=0.02, β=2.0) - Learn abbreviations/symbols  
├── prompts/structured_compression.txt   → {content}
├── prompts/structured_decompression.txt → {compressed}
└── prompts/evaluation.txt               → {original}, {decompressed}

Stage 3: Freeform (α=0.05, β=1.5) - Develop symbolic patterns
├── prompts/freeform_compression.txt     → {content}
├── prompts/freeform_decompression.txt   → {compressed}
└── prompts/evaluation.txt               → {original}, {decompressed}

Stage 4: Cognition (α=0.1, β=1.0) - Compression-first reasoning
├── prompts/cognition_compression.txt    → {content}
├── prompts/cognition_decompression.txt  → {compressed}
└── prompts/evaluation.txt               → {original}, {decompressed}

=== PROMPT TAG SOURCES ===

{content}       ← Dataset samples (from 'text' field)
                  • agentlans/wikipedia-paragraphs (stages 1-3)
                  • willcb/gsm8k-python-test (stage 4, mapped to 'text' field)

{compressed}    ← Output from compression rollout
                  • Extracted from <compress>...</compress> tags in completion
                  • Used as input for decompression rollout

{original}      ← Original dataset sample content
                  • Same as {content}, preserved for evaluation
                  
{decompressed}  ← Output from decompression rollout  
                  • Extracted from <decompress>...</decompress> tags in completion
                  • Used for fidelity evaluation against {original}

=== EXECUTION FLOW ===

For each dataset sample:
1. Compression Rollout:   {content} → compression_prompt → <compress>result</compress>
2. Decompression Rollout: {compressed} → decompression_prompt → <decompress>result</decompress>  
3. Evaluation:            {original} + {decompressed} → evaluation_prompt → fidelity_score
4. Reward Calculation:    reward = base_score - α×tokens - β×(1-fidelity)
5. Both rollouts receive the same reward for gradient updates

=== PROMPT FORMATS ===

All prompts support two formats:
• Legacy: Simple string templates with {tag} replacement
• Multi-turn: Conversation format using <|im_start|>role...content<|im_end|> structure

The system auto-detects format based on "# Multi-turn conversation format" header.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datasets import Dataset, load_dataset
import torch
import re
import os

# Rich imports for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
from rich.rule import Rule
from rich import box

import verifiers as vf
from verifiers.envs import Environment
from verifiers.rubrics import Rubric
from prompts import PromptLibrary

# Create rich console
console = Console()

# Set up Rich logging to capture all logs
logging.basicConfig(
    level=logging.ERROR,  # Only show errors from libraries
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
)

# Suppress specific noisy loggers
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class CompressorEnv(Environment):
    """
    Compression environment that generates compression/decompression pairs.
    
    For each prompt, generates:
    1. Compression rollout: original_content → compressed_form
    2. Decompression rollout: compressed_form → decompressed_content
    
    Both rollouts receive the same reward based on compression quality + fidelity.
    This works within standard GRPO framework without trainer modifications.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        evaluator_model: str = "Qwen/Qwen2.5-7B-Instruct",
        alpha: float = 0.01,
        beta: float = 1.0,
        base_score: float = 10.0,
        stage: str = "identity",
        max_concurrent: int = 64,
        **kwargs
    ):
        # Initialize the frozen evaluator client
        from openai import OpenAI
        self.evaluator_client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="none"
        )
        self.evaluator_model = evaluator_model

        # Compression parameters
        self.alpha = alpha
        self.beta = beta
        self.base_score = base_score
        self.stage = stage

        # Initialize prompt library
        self.prompt_lib = PromptLibrary()

        # Load stage-specific prompts from files
        self.compression_prompts = {
            "identity": self.prompt_lib.load_prompt("identity_compression.txt"),
            "structured": self.prompt_lib.load_prompt("structured_compression.txt"),
            "freeform": self.prompt_lib.load_prompt("freeform_compression.txt"),
            "cognition": self.prompt_lib.load_prompt("cognition_compression.txt")
        }

        self.decompression_prompts = {
            "identity": self.prompt_lib.load_prompt("identity_decompression.txt"),
            "structured": self.prompt_lib.load_prompt("structured_decompression.txt"),
            "freeform": self.prompt_lib.load_prompt("freeform_decompression.txt"),
            "cognition": self.prompt_lib.load_prompt("cognition_decompression.txt")
        }

        # Load evaluation prompt
        self.evaluation_prompt = self.prompt_lib.load_prompt("evaluation.txt")

        # Custom rubric for compression rewards
        rubric = Rubric(funcs=[self._compression_reward_func], weights=[1.0])

        # Initialize parent Environment
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=self._get_system_prompt(),
            rubric=rubric,
            max_concurrent=max_concurrent,
            message_type='chat',
            **kwargs
        )

    def _get_system_prompt(self) -> str:
        return f"You are training to learn symbolic compression and decompression. Current stage: {self.stage}"

    def _extract_compressed(self, completion: str) -> str:
        """Extract content from <compress> tags"""
        match = re.search(r'<compress>\s*(.*?)\s*(?:</compress>|$)', completion, re.DOTALL)
        return match.group(1).strip() if match else completion.strip()

    def _extract_decompressed(self, completion: str) -> str:
        """Extract content from <decompress> tags"""
        match = re.search(r'<decompress>\s*(.*?)\s*(?:</decompress>|$)', completion, re.DOTALL)
        return match.group(1).strip() if match else completion.strip()

    def _extract_fidelity_score(self, evaluation: str) -> float:
        """Extract fidelity score from evaluation"""
        match = re.search(r'Fidelity Score:\s*([0-9]*\.?[0-9]+)', evaluation)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.5  # default fallback score

    def _parse_structured_evaluation(self, evaluation: str) -> Dict[str, Any]:
        """Parse JSON evaluation format from model output"""
        result = {
            'deviations': [],
            'inaccuracies': [],
            'missing_statements': [],
            'acceptable_extensions': [],
            'total_issues': 0,
            'severity': 'MINOR',
            'quality': 'GOOD',
            'raw_evaluation': evaluation
        }
        
        try:
            # Extract JSON from anywhere in the response
            # Look for ```json blocks first
            json_block_match = re.search(r'```json\s*(.*?)\s*```', evaluation, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1).strip()
            else:
                # Look for standalone JSON object
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', evaluation, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.warning("No JSON found in evaluation")
                    return result
            
            # Parse the JSON
            import json
            eval_data = json.loads(json_str)
            
            # Update result with parsed data
            result.update({
                'deviations': eval_data.get('deviations', []),
                'inaccuracies': eval_data.get('inaccuracies', []), 
                'missing_statements': eval_data.get('missing_statements', []),
                'acceptable_extensions': eval_data.get('acceptable_extensions', []),
                'total_issues': eval_data.get('total_issues', 0),
                'severity': eval_data.get('severity', 'MINOR'),
                'quality': eval_data.get('quality', 'GOOD'),
                'raw_evaluation': evaluation
            })
            
            # Auto-calculate total_issues if not provided or seems wrong
            calculated_issues = len(result['deviations']) + len(result['inaccuracies']) + len(result['missing_statements'])
            if result['total_issues'] == 0 and calculated_issues > 0:
                result['total_issues'] = calculated_issues
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON evaluation: {e}")
        except Exception as e:
            logger.warning(f"Error parsing structured evaluation: {e}")
            
        return result

    def _calculate_fidelity_from_structured_eval(self, eval_result: Dict[str, Any]) -> float:
        """Calculate fidelity score from structured evaluation"""
        total_issues = eval_result['total_issues']
        severity = eval_result['severity']
        quality = eval_result['quality']
        
        # Start with perfect score
        fidelity = 1.0
        
        # Penalize based on number of issues
        if total_issues > 0:
            # Base penalty per issue
            base_penalty = 0.1
            
            # Adjust penalty based on severity
            severity_multiplier = {
                'MINOR': 0.5,
                'MODERATE': 1.0, 
                'MAJOR': 2.0
            }.get(severity, 1.0)
            
            # Calculate penalty
            penalty = min(total_issues * base_penalty * severity_multiplier, 0.9)  # Cap at 0.9 to keep minimum 0.1
            fidelity -= penalty
        
        # Additional adjustment based on overall quality assessment
        quality_adjustments = {
            'EXCELLENT': 0.0,   # No adjustment
            'GOOD': -0.05,      # Small penalty
            'FAIR': -0.15,      # Moderate penalty  
            'POOR': -0.3        # Large penalty
        }
        fidelity += quality_adjustments.get(quality, 0.0)
        
        # Ensure fidelity stays within bounds
        return max(0.0, min(1.0, fidelity))

    def _count_tokens(self, text: str) -> int:
        """Rough token count approximation"""
        return len(text) // 4 + 1

    def _evaluate_compression(self, original: str, compressed: str, decompressed: str) -> Tuple[float, str]:
        """Use frozen evaluator to assess compression quality with structured feedback"""
        formatted_prompt = self.prompt_lib.format_prompt(
            self.evaluation_prompt,
            original=original,
            decompressed=decompressed
        )
        
        if isinstance(formatted_prompt, list):
            # Multi-turn conversation - ensure proper message format
            evaluation_messages = [{"role": msg["role"], "content": str(msg["content"])} for msg in formatted_prompt]
        else:
            # Single-turn format
            evaluation_messages = [{"role": "user", "content": str(formatted_prompt)}]

        try:
            response = self.evaluator_client.chat.completions.create(
                model=self.evaluator_model.split('/')[-1],
                messages=evaluation_messages,
                max_tokens=1500,  # Increased for detailed structured output
                temperature=0.1
            )
            evaluation = response.choices[0].message.content or ""
            
            # Parse structured evaluation
            eval_result = self._parse_structured_evaluation(evaluation)
            
            # Calculate fidelity from structured assessment
            fidelity_score = self._calculate_fidelity_from_structured_eval(eval_result)
            
            # Store structured evaluation in the raw evaluation string for logging
            return fidelity_score, evaluation
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.5, "Evaluation failed"

    def _calculate_reward(self, compressed: str, fidelity_score: float) -> float:
        """Calculate reward based on compression and fidelity"""
        token_count = self._count_tokens(compressed)
        compression_penalty = self.alpha * token_count
        fidelity_penalty = self.beta * (1.0 - fidelity_score)

        reward = self.base_score - compression_penalty - fidelity_penalty

        # Rich debug logging
        if torch.rand(1).item() < 0.05:  # 5% chance for detailed logging
            debug_text = Text()
            debug_text.append("🔍 Reward Analysis\n", style="bold yellow")
            debug_text.append(f"Stage: {self.stage} | ", style="dim")
            debug_text.append(f"Tokens: {token_count} | ", style="cyan")
            debug_text.append(f"Fidelity: {fidelity_score:.3f} | ", style="green" if fidelity_score > 0.7 else "yellow")
            debug_text.append(f"Final: {reward:.3f}", style="bold white")
            console.print(debug_text)

        return max(0.0, reward)

    def log_compression_sample(self, original: str, compressed: str, decompressed: str, fidelity_score: float, reward: float, evaluation: str = "") -> None:
        """Log a sample compression with detailed evaluation feedback"""
        # Truncate for display
        orig_display = (original[:80] + "...") if len(original) > 80 else original
        comp_display = (compressed[:80] + "...") if len(compressed) > 80 else compressed
        decomp_display = (decompressed[:80] + "...") if len(decompressed) > 80 else decompressed

        # Color coding based on fidelity score
        fidelity_color = "green" if fidelity_score > 0.8 else "yellow" if fidelity_score > 0.5 else "red"
        reward_color = "green" if reward > 5.0 else "yellow" if reward > 2.0 else "red"
        
        # Create the sample panel
        sample_text = Text()
        sample_text.append("📝 Original: ", style="bold blue")
        sample_text.append(f"{orig_display}\n", style="dim")
        sample_text.append("🗜️  Compressed: ", style="bold cyan")
        sample_text.append(f"{comp_display}\n", style="cyan")
        sample_text.append("📤 Decompressed: ", style="bold magenta")
        sample_text.append(f"{decomp_display}\n", style="magenta")
        
        # Metrics
        compression_ratio = len(original) / max(1, len(compressed))
        sample_text.append(f"📊 Ratio: ", style="bold")
        sample_text.append(f"{compression_ratio:.1f}x  ", style="bright_white")
        sample_text.append(f"🎯 Fidelity: ", style="bold")
        sample_text.append(f"{fidelity_score:.3f}  ", style=fidelity_color)
        sample_text.append(f"🏆 Reward: ", style="bold")
        sample_text.append(f"{reward:.2f}", style=reward_color)
        
        # Evaluation details if available
        if evaluation:
            eval_result = self._parse_structured_evaluation(evaluation)
            if eval_result['total_issues'] > 0:
                sample_text.append(f"\n⚠️  Issues: {eval_result['total_issues']} ({eval_result['severity']})", style="yellow")
            else:
                sample_text.append(f"\n✅ Perfect preservation", style="green")

        console.print(Panel(
            sample_text,
            title=f"[bold white]🧠 {self.stage.upper()} Sample[/]",
            border_style="blue",
            box=box.ROUNDED
        ))

    def _rollout_compression(self, client, model: str, prompt, answer: str, sampling_args: Dict[str, Any] = {}, **kwargs) -> Tuple[Union[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Context A: Compression rollout (original content → compressed form).
        
        Args:
            prompt: Original content to compress
            answer: Expected answer (unused in compression)
            
        Returns:
            Tuple of (completion, state) with compressed content
        """
        # Extract content from prompt
        if isinstance(prompt, list):
            # Chat format - extract the user message content
            user_messages = [msg for msg in prompt if msg['role'] == 'user']
            content = user_messages[-1]['content'] if user_messages else str(prompt)
        else:
            content = prompt

        try:
            prompt_template = self.compression_prompts[self.stage]
            formatted_prompt = self.prompt_lib.format_prompt(prompt_template, content=content)
            
            if isinstance(formatted_prompt, list):
                # Multi-turn conversation
                compression_messages = formatted_prompt
            else:
                # Single-turn format
                compression_messages = [{"role": "user", "content": formatted_prompt}]
            
            completion = self.get_model_response(
                prompt=compression_messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type='chat'
            )
            
            compressed = self._extract_compressed(completion)
            
            return [{"role": "assistant", "content": completion}], {
                "mode": "compression",
                "compressed": compressed,
                "original_content": content
            }
            
        except Exception as e:
            logger.error(f"Compression rollout failed: {e}")
            error_completion = [{"role": "assistant", "content": f"[ERROR] Compression failed"}]
            error_state = {"mode": "compression", "error": str(e)}
            return error_completion, error_state

    def _rollout_decompression(self, client, model: str, compressed_content: str, answer: str, sampling_args: Dict[str, Any] = {}, **kwargs) -> Tuple[Union[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Context B: Decompression rollout (compressed form → decompressed content).
        
        Args:
            compressed_content: Compressed form to decompress  
            answer: Expected answer (unused in decompression)
            
        Returns:
            Tuple of (completion, state) with decompressed content
        """
        try:
            # Extract compressed content (in case it's still wrapped in completion)
            compressed = self._extract_compressed(compressed_content)
            
            prompt_template = self.decompression_prompts[self.stage]
            formatted_prompt = self.prompt_lib.format_prompt(prompt_template, compressed=compressed)
            
            if isinstance(formatted_prompt, list):
                # Multi-turn conversation
                decompression_messages = formatted_prompt
            else:
                # Single-turn format
                decompression_messages = [{"role": "user", "content": formatted_prompt}]
            
            completion = self.get_model_response(
                prompt=decompression_messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type='chat'
            )
            
            decompressed = self._extract_decompressed(completion)
            
            return [{"role": "assistant", "content": completion}], {
                "mode": "decompression",
                "compressed": compressed,
                "decompressed": decompressed
            }
            
        except Exception as e:
            logger.error(f"Decompression rollout failed: {e}")
            error_completion = [{"role": "assistant", "content": f"[ERROR] Decompression failed"}]
            error_state = {"mode": "decompression", "error": str(e)}
            return error_completion, error_state

    def rollout(self, client, model: str, prompt, answer: str, mode: str = "compression", sampling_args: Dict[str, Any] = {}, **kwargs) -> Tuple[Union[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Generic rollout method - not implemented for CompressorEnv.
        
        Use _rollout_compression() and _rollout_decompression() directly instead.
        This method exists only for interface compatibility with base Environment class.
        """
        raise NotImplementedError(
            "Generic rollout not supported. Use _rollout_compression() or _rollout_decompression() instead."
        )

    def run_rollouts(self, prompts, answers, client, model, sampling_args, max_concurrent=32, **kwargs):
        """
        Override to generate compression/decompression pairs.
        
        For N prompts, generates 2N rollouts:
        - N compression rollouts: original → compressed  
        - N decompression rollouts: compressed → decompressed
        
        Returns 2N rollouts that will be processed by rubric for rewards.
        """
        all_rollouts = []
        compression_pairs = []  # Store pairs for reward calculation
        
        # Rich progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"🔄 Generating {self.stage} compression pairs", total=len(prompts))
            
            for i, (prompt, answer) in enumerate(zip(prompts, answers)):
                try:
                    # Rollout 1: Compression (original → compressed)
                    comp_completion, comp_state = self._rollout_compression(
                        client=client,
                        model=model, 
                        prompt=prompt,
                        answer=answer,
                        sampling_args=sampling_args,
                        **kwargs
                    )
                    
                    # Rollout 2: Decompression (compressed → decompressed)
                    compressed_content = comp_state["compressed"]
                    
                    decomp_completion, decomp_state = self._rollout_decompression(
                        client=client,
                        model=model,
                        compressed_content=compressed_content,  # Clear parameter name
                        answer=answer,
                        sampling_args=sampling_args,
                        **kwargs
                    )
                    
                    # Calculate the actual reward for this pair
                    original = comp_state.get("original_content", "")
                    compressed = comp_state.get("compressed", "")
                    decompressed = decomp_state.get("decompressed", "")
                    
                    # Context C: Evaluation (frozen model)
                    fidelity_score, evaluation = self._evaluate_compression(original, compressed, decompressed)
                    reward = self._calculate_reward(compressed, fidelity_score)
                    
                    # Store reward in both states
                    comp_state["reward"] = reward
                    comp_state["fidelity_score"] = fidelity_score
                    decomp_state["reward"] = reward
                    decomp_state["fidelity_score"] = fidelity_score
                    decomp_state["original_content"] = original  # Add for completeness
                    
                    all_rollouts.append((comp_completion, comp_state))
                    all_rollouts.append((decomp_completion, decomp_state))
                    
                    # Log sample occasionally
                    if torch.rand(1).item() < 0.1:  # 10% chance to log
                        self.log_compression_sample(original, compressed, decompressed, fidelity_score, reward, evaluation)
                    
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    console.print(f"[red]❌ Failed to generate rollout pair {i}: {e}[/]")
                    # Add error rollouts
                    error_completion = [{"role": "assistant", "content": "[ERROR] Generation failed"}]
                    error_state = {"error": str(e), "reward": 0.0}
                    all_rollouts.extend([(error_completion, error_state), (error_completion, error_state)])
                    progress.update(task, advance=1)
        
        # Final summary
        pairs_generated = len(all_rollouts) // 2
        console.print(f"[green]✅ Generated {len(all_rollouts)} rollouts ({pairs_generated} pairs)[/]")
        return all_rollouts

    def _compression_reward_func(self, prompt, completion, answer, state, **kwargs) -> float:
        """
        Custom reward function that returns the pre-calculated reward from the state.
        
        The actual reward calculation happens in run_rollouts() where we have access
        to both compression and decompression parts of each pair.
        """
        if hasattr(state, 'get'):
            return state.get('reward', 0.0)
        return 0.0

    def format_dataset(self,
                       dataset: Dataset,
                       system_prompt: str | None = None,
                       few_shot: List[Dict[str, Any]] | None = None,
                       question_key: str = "text",  # Use "text" instead of "question"
                       answer_key: str = "answer") -> Dataset:
        """
        Override parent format_dataset to handle datasets with 'text' column.
        For compression training, we don't need real answers, so we add dummy ones.
        """
        # Add dummy answer column if it doesn't exist
        if "answer" not in dataset.column_names:
            dataset = dataset.map(lambda x: {**x, "answer": ""})
        
        # Extract format_prompt as a standalone function to avoid capturing self
        def format_prompt_fn(prompt: str) -> List[Dict[str, Any]]:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            if few_shot:
                messages.extend(few_shot)
            messages.append({'role': 'user', 'content': prompt})
            return messages

        return dataset.map(lambda x: {
            "prompt": format_prompt_fn(x[question_key]),
            "answer": x[answer_key]
        }, num_proc=self.max_concurrent)

def get_stage_config(stage: str) -> Dict[str, Any]:
    """Get stage-specific training configuration"""
    configs = {
        "identity": {
            "alpha": 0.005,  # Very light compression penalty - focus on fidelity
            "beta": 3.0,     # Strong fidelity requirement
            "num_iterations": 2,
            "max_steps": 150,
            "description": "Identity Pretraining: Learn basic compress → decompress → match"
        },
        "structured": {
            "alpha": 0.02,   # Moderate compression penalty
            "beta": 2.0,     # Balanced fidelity requirement
            "num_iterations": 3,
            "max_steps": 200,
            "description": "Structured Compression: Learn abbreviations and symbols"
        },
        "freeform": {
            "alpha": 0.05,   # Higher compression penalty - encourage creativity
            "beta": 1.5,     # Slightly relaxed fidelity
            "num_iterations": 4,
            "max_steps": 300,
            "description": "Freeform Compression: Develop own symbolic patterns"
        },
        "cognition": {
            "alpha": 0.1,    # Strong compression incentive
            "beta": 1.0,     # Balanced fidelity
            "num_iterations": 5,
            "max_steps": 400,
            "description": "Compression-First Cognition: Reason in compressed space"
        }
    }
    return configs[stage]

def train_stage(name: str, env: CompressorEnv, model_path: str, base_model_name: str) -> str:
    """Train a single stage and return the path to the trained model"""
    stage_config = get_stage_config(name)

    # Step 1: Model Loading
    console.print(f"[cyan]📦 Loading model: {model_path}[/]")
    
    model, tokenizer = vf.get_model_and_tokenizer(model_path)
    console.print(f"[green]✓[/] Model loaded")
    console.print()

    # Step 2: Training Configuration
    console.print(f"[cyan]⚙️ Configuring training...[/]")
    run_name = f"compressor-{name}-{base_model_name.split('/')[-1].lower()}"
    training_args = vf.grpo_defaults(run_name=run_name)
    training_args.num_iterations = stage_config["num_iterations"]
    training_args.per_device_train_batch_size = 2
    training_args.num_generations = 8
    training_args.gradient_accumulation_steps = 2
    training_args.max_prompt_length = 1024
    training_args.max_completion_length = 2048
    training_args.max_steps = stage_config["max_steps"]
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 50
    training_args.logging_steps = 10
    training_args.save_strategy = "steps"
    training_args.save_steps = 100
    training_args.output_dir = f"outputs/compressor-{name}"

    # Display configuration table
    train_table = Table(show_header=False, box=box.SIMPLE)
    train_table.add_column("Parameter", style="cyan", width=25)
    train_table.add_column("Value", style="white")
    
    train_table.add_row("Alpha (compression)", f"{stage_config['alpha']}")
    train_table.add_row("Beta (fidelity)", f"{stage_config['beta']}")
    train_table.add_row("Max steps", f"{stage_config['max_steps']}")
    train_table.add_row("Training samples", f"{len(env.dataset) if env.dataset else 0}")
    train_table.add_row("Batch size", f"{training_args.per_device_train_batch_size} prompts → {training_args.per_device_train_batch_size * 2} rollouts")
    
    console.print(train_table)
    console.print()

    # Step 3: Trainer Setup
    console.print(f"[cyan]🏋️ Creating trainer...[/]")
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=training_args,
    )
    console.print(f"[green]✓[/] Trainer ready")
    console.print()

    # Step 4: Training Execution
    console.print(f"[bold yellow]🚀 Training {name} stage...[/]")
    trainer.train()
    console.print(f"[green]✓[/] Training completed")
    console.print()

    # Step 5: Model Saving
    output_path = f"outputs/compressor-{name}"
    console.print(f"[cyan]💾 Saving model to {output_path}...[/]")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    console.print(f"[green]✓[/] Model saved")
    console.print()

    return output_path

def main():
    """Main multi-stage training function"""
    # Define the training stages in order
    stages = ["identity", "structured", "freeform", "cognition"]

    # Starting model
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    current_model_path = base_model_name

    # Clear screen for full-screen experience
    console.clear()
    
    # Simple header with full-width separator
    console.print(Rule("[bold cyan]🧠 Symbolic Compressor Training", style="cyan"))
    console.print(f"[dim]Base model: {base_model_name}[/]")
    console.print(f"[dim]Stages: {' → '.join(stages)}[/]")
    console.print(Rule(style="dim"))
    console.print()
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)

    # Train each stage progressively
    for i, stage in enumerate(stages):
        stage_config = get_stage_config(stage)
        
        # Full-width stage separator for scrolling effect
        console.print(Rule(f"[bold yellow]Stage {i+1}/{len(stages)}: {stage.upper()}", style="yellow"))
        console.print(f"[dim]{stage_config['description']}[/]")
        console.print(f"[dim]α={stage_config['alpha']} β={stage_config['beta']}[/]")
        console.print(Rule(style="yellow"))
        console.print()

        try:
            # Dataset loading
            console.print(f"[cyan]📊 Loading dataset...[/]")
            
            if stage == "cognition":
                try:
                    dataset = load_dataset('willcb/gsm8k-python-test', split='train')
                    dataset = dataset.map(lambda x: {'text': x['question']})
                    console.print("[green]✓[/] GSM8K dataset loaded")
                except:
                    dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train')
                    console.print("[yellow]⚠[/] Using Wikipedia fallback")
            else:
                dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train')
                console.print("[green]✓[/] Wikipedia dataset loaded")

            # Dataset splitting
            if hasattr(dataset, 'select') and hasattr(dataset, '__len__'):
                total_size = min(1500, len(dataset))
                eval_dataset = dataset.select(range(50))
                train_dataset = dataset.select(range(50, total_size))
                console.print(f"[green]✓[/] Split: {len(train_dataset)} train, {len(eval_dataset)} eval")
            else:
                items = []
                for i, item in enumerate(dataset):
                    if i >= 1500:
                        break
                    items.append(item)
                
                from datasets import Dataset
                eval_dataset = Dataset.from_list(items[:50])
                train_dataset = Dataset.from_list(items[50:])
                console.print(f"[green]✓[/] Processed: {len(train_dataset)} train, {len(eval_dataset)} eval")

            console.print()

            # Environment setup
            console.print(f"[cyan]🏗️ Setting up environment...[/]")
            compressor_env = CompressorEnv(
                dataset=train_dataset,
                eval_dataset=eval_dataset,
                stage=stage,
                alpha=stage_config["alpha"],
                beta=stage_config["beta"],
                max_concurrent=8,
            )
            console.print("[green]✓[/] Environment ready")
            console.print()

            # Training
            console.print(f"[cyan]🚀 Training...[/]")
            current_model_path = train_stage(stage, compressor_env, current_model_path, base_model_name)

            # Success with separator
            console.print(Rule("[green]✅ Stage Complete", style="green"))
            console.print(f"[bold green]✅ Stage {stage} complete[/]")
            console.print(f"[dim]📁 Saved to: {current_model_path}[/]")
            console.print(Rule(style="green"))
            console.print()

        except Exception as e:
            console.print(Rule("[red]❌ Stage Failed", style="red"))
            console.print(f"[bold red]❌ Stage {stage} failed: {e}[/]")
            console.print("[red]🛑 Stopping training[/]")
            console.print(Rule(style="red"))
            raise

    # Final completion with full-screen effect
    console.print(Rule("[bold green]🏆 TRAINING COMPLETED", style="green"))
    console.print(f"[bold green]🏆 Training completed![/]")
    console.print(f"[dim]Final model: {current_model_path}[/]")
    console.print(f"[dim]Usage: model, tokenizer = vf.get_model_and_tokenizer('{current_model_path}')[/]")
    console.print(Rule(style="green"))
    console.print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
