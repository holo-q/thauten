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
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import verifiers as vf
from datasets import Dataset, load_dataset
from datasets.arrow_dataset import Dataset as ArrowDataset
from datasets.iterable_dataset import IterableDataset
from openai import OpenAI
from rich import box
# Rich imports for beautiful output
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
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
        max_concurrent: int = 64,
        dry_run: bool = False,
        **kwargs
    ):
        self.dry_run = dry_run

        # In a dry run, we don't initialize the evaluator client to avoid network calls.
        if not self.dry_run:
            from openai import OpenAI
            self.evaluator_client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="none"
            )
            self.evaluator_model = evaluator_model
        else:
            self.evaluator_client = None
            self.evaluator_model = None

        # Compression parameters
        self.alpha = alpha
        self.beta = beta
        self.base_score = base_score

        # Initialize prompt library
        self.prompt_lib = PromptLibrary()

        # Load prompts for the single compression stage
        self.compression_prompt = self.prompt_lib.load_prompt("compression.txt")
        self.decompression_prompt = self.prompt_lib.load_prompt("decompression.txt")
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

    def get_model_response(self, prompt, client, model, sampling_args, message_type):
        """
        Override the base method to intercept calls during a dry run.
        """
        if self.dry_run:
            # In a dry run, we simulate the model's response.
            # The actual context logging is handled in `log_compression_sample`.
            # We need to provide dummy responses that can be parsed by the extraction logic.
            prompt_str = " ".join([msg.get('content', '') for msg in prompt])
            if "compress" in prompt_str.lower() and "decompress" not in prompt_str.lower():
                 return "<compress>dry-run compressed content</compress>"
            elif "decompress" in prompt_str.lower():
                 return "<decompress>dry-run decompressed content</decompress>"
            else: # Evaluation
                 return '```json\n{"total_issues": 0, "severity": "MINOR", "quality": "EXCELLENT"}\n```\nFidelity Score: 1.0'

        # During a real run, this calls the underlying `verifiers` library function
        return super().get_model_response(prompt, client, model, sampling_args, message_type)

    def _get_system_prompt(self) -> str:
        # System prompt is now defined in the prompt files, so this is just a fallback.
        return "You are in a symbolic compression training environment."

    def _extract_fence(self, completion: Union[str, List[Dict[str, Any]]], wrapper_tag: Optional[str]) -> str:
        """Extract content from a dynamic wrapper tag, e.g., <compress>...</compress>"""
        if not wrapper_tag:
            return completion.strip() if isinstance(completion, str) else completion[-1]["content"].strip()

        # Handle both string and conversation format
        content = completion
        if isinstance(completion, list):
            # Get the last assistant message
            for msg in reversed(completion):
                if msg["role"] == "assistant":
                    content = msg["content"]
                    break
            if not isinstance(content, str):
                return ""

        # Regex to find <tag>content</tag> or <tag>content
        tag = wrapper_tag.lower()
        # Find all matches and take the last one

        text_content = ""
        if isinstance(content, str):
            text_content = content
        elif isinstance(content, list):
            for msg in reversed(content):
                if msg["role"] == "assistant":
                    text_content = msg["content"]
                    break

        if not text_content:
            return ""

        matches = list(re.finditer(fr'<{tag}>\s*(.*?)\s*(?:</{tag}>|$)', text_content, re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()
        return text_content.strip()

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
            'deviations':            [],
            'inaccuracies':          [],
            'missing_statements':    [],
            'acceptable_extensions': [],
            'total_issues':          0,
            'severity':              'MINOR',
            'quality':               'GOOD',
            'raw_evaluation':        evaluation
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
                'deviations':            eval_data.get('deviations', []),
                'inaccuracies':          eval_data.get('inaccuracies', []),
                'missing_statements':    eval_data.get('missing_statements', []),
                'acceptable_extensions': eval_data.get('acceptable_extensions', []),
                'total_issues':          eval_data.get('total_issues', 0),
                'severity':              eval_data.get('severity', 'MINOR'),
                'quality':               eval_data.get('quality', 'GOOD'),
                'raw_evaluation':        evaluation
            })

            # Auto-calculate total_issues if not provided or seems wrong
            calculated_issues = len(result['deviations']) + len(result['inaccuracies']) + len(result['missing_statements'])
            if result['total_issues'] == 0 and calculated_issues > 0:
                result['total_issues'] = calculated_issues

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse structured evaluation: {e}")

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
                'MINOR':    0.5,
                'MODERATE': 1.0,
                'MAJOR':    2.0
            }.get(severity, 1.0)

            # Calculate penalty
            penalty = min(total_issues * base_penalty * severity_multiplier, 0.9)  # Cap at 0.9 to keep minimum 0.1
            fidelity -= penalty

        # Additional adjustment based on overall quality assessment
        quality_adjustments = {
            'EXCELLENT': 0.0,  # No adjustment
            'GOOD':      -0.05,  # Small penalty
            'FAIR':      -0.15,  # Moderate penalty
            'POOR':      -0.3  # Large penalty
        }
        fidelity += quality_adjustments.get(quality, 0.0)

        # Ensure fidelity stays within bounds
        return max(0.0, min(1.0, fidelity))

    def _count_tokens(self, text: str) -> int:
        """Counts tokens by splitting on whitespace."""
        return len(text.split())

    def _format_conversation_text(self, context: List[Dict[str, str]]) -> Text:
        """Helper to format a conversation for rich display."""
        text = Text()
        for msg in context:
            role = msg.get("role", "none")
            content = msg.get("content", "")
            color = "cyan"
            if role == "system": color = "yellow"
            elif role == "user": color = "green"
            elif role == "assistant": color = "magenta"
            text.append(f"<|{role.upper()}|>\n", style=f"bold {color}")
            text.append(content + "\n\n")
        return text

    def _evaluate_compression(self, original: str, compressed: str, decompressed: str) -> Tuple[float, str]:
        """
        Evaluate compression fidelity by comparing original and decompressed content using a rollout.
        """
        from prompts import rollout_prompt

        if self.dry_run:
            return 1.0, "Dry run evaluation"

        # Define the generation function for the evaluator
        def get_evaluation(messages):
            return self.get_model_response(
                prompt=messages,
                client=self.evaluator_client,
                model=self.evaluator_model,
                sampling_args={'temperature': 0.0},
                message_type='chat'
            )

        # Perform the evaluation rollout
        if isinstance(self.evaluation_prompt, str):
             # This should not happen with the new prompt system, but as a fallback:
            evaluation_messages = [{"role": "user", "content": self.evaluation_prompt.format(original=original, decompressed=decompressed)}]
            evaluation_messages.append({"role": "assistant", "content": get_evaluation(evaluation_messages)})
        else:
            evaluation_messages = rollout_prompt(
                self.evaluation_prompt,
                get_evaluation,
                data={"original": original, "decompressed": decompressed}
            )

        # The evaluation result is in the content of the last message
        evaluation_text = ""
        if evaluation_messages and isinstance(evaluation_messages, list):
            # The full content is in the last message of the list
            evaluation_text = evaluation_messages[-1].get('content', '')

        # Extract fidelity score from the final generated text
        structured_eval = self._parse_structured_evaluation(evaluation_text)
        fidelity_score = self._calculate_fidelity_from_structured_eval(structured_eval)

        return fidelity_score, evaluation_text

    def _calculate_reward(self, compressed: str, fidelity_score: float) -> float:
        """Calculate reward based on compression and fidelity"""
        token_count = self._count_tokens(compressed)
        penalty = self.alpha * token_count + self.beta * (1 - fidelity_score)
        reward = self.base_score - penalty

        # Rich debug logging
        if torch.rand(1).item() < 0.05:  # 5% chance for detailed logging
            debug_text = Text()
            debug_text.append("🔍 Reward Analysis\n", style="bold yellow")
            debug_text.append(f"Tokens: {token_count} | ", style="cyan")
            debug_text.append(f"Fidelity: {fidelity_score:.3f} | ", style="green" if fidelity_score > 0.7 else "yellow")
            debug_text.append(f"Final: {reward:.2f}", style="bold white")
            console.print(debug_text)

        return max(0.0, reward)

    def log_compression_sample(self, original: str, compressed: str, decompressed: str, fidelity_score: float, reward: float, evaluation: str = "", cmp_ctx: Optional[List[Dict[str, Any]]] = None, dcmp_ctx: Optional[List[Dict[str, Any]]] = None) -> None:
        if self.dry_run:
            renderables = []
            if cmp_ctx:
                renderables.append(self._format_conversation_text(cmp_ctx))
            if dcmp_ctx:
                if cmp_ctx:
                    renderables.append(Rule(style="yellow"))
                renderables.append(self._format_conversation_text(dcmp_ctx))

            console.print(Panel(
                Group(*renderables),
                title="[bold yellow]Dry Run: Full Conversation Flow[/]",
                border_style="yellow",
                box=box.ROUNDED,
                title_align="left"
            ))
            return

        # Create a table for beautiful logging
        table = Table(box=box.MINIMAL, show_header=False, expand=True)
        table.add_column(style="bold magenta")
        table.add_column(style="white")

        table.add_row("Original:", original[:200] + "..." if len(original) > 200 else original)
        table.add_row("Compressed:", compressed)
        table.add_row("Decompressed:", decompressed[:200] + "..." if len(decompressed) > 200 else decompressed)
        table.add_row("Fidelity:", f"{fidelity_score:.2f}")
        table.add_row("Reward:", f"{reward:.2f}")
        # table.add_row("Evaluation:", evaluation)

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_row(Panel(
            table,
            title=f"[bold green]Compression Sample[/]",
            border_style="green",
            title_align="left"
        ))
        console.print(grid)

    def rollout(self, client: OpenAI, model: str, prompt: str | List[Dict[str, Any]], answer: str, sampling_args=None, **kwargs: Any) -> Tuple[Union[str, List[Dict[str, Any]]], Dict[str, Any]]:
        if sampling_args is None:
            sampling_args = {}
        raise NotImplementedError("This environment uses a custom generate loop, not the standard rollout method.")

    def generate(self,
                 inputs: Dataset,
                 client,
                 model: str,
                 sampling_args=None,
                 **kwargs: Any) -> Dict[str, List[Any]]:
        """
        Overrides the base generate method to perform a full compression/decompression cycle.
        """
        if sampling_args is None:
            sampling_args = {}
        from prompts import rollout_prompt, format_prompt

        results = {
            "original":     [],
            "compressed":   [],
            "decompressed": [],
            "evaluation":   [],
            "fidelity":     [],
            "reward":       [],
            "prompt":       [],
            "completion":   [],
            "answer":       [],
            "state":        [],
        }

        # The 'generate' function for the rollout
        def get_generation(messages):
            return self.get_model_response(
                prompt=messages,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type='chat'
            )

        for sample in inputs:
            original_content = sample['text'] # type: ignore

            if isinstance(self.compression_prompt, str):
                raise TypeError("Compression prompt must be a PromptTemplate for rollouts.")
            cmp_ctx = rollout_prompt(self.compression_prompt, get_generation, data={"input": original_content})
            cmp_tag = self._extract_fence(cmp_ctx, "compress")

            if isinstance(self.decompression_prompt, str):
                raise TypeError("Decompression prompt must be a PromptTemplate for rollouts.")
            dcmp_ctx = rollout_prompt(self.decompression_prompt, get_generation, data={"input": cmp_tag})
            dcmp_tag = self._extract_fence(dcmp_ctx, "decompress")

            fidelity_score, evaluation_text = self._evaluate_compression(
                original=original_content,
                compressed=cmp_tag,
                decompressed=dcmp_tag
            )

            reward = self._calculate_reward(cmp_tag, fidelity_score)

            self.log_compression_sample(
                original=original_content,
                compressed=cmp_tag,
                decompressed=dcmp_tag,
                fidelity_score=fidelity_score,
                reward=reward,
                evaluation=evaluation_text,
                cmp_ctx=cmp_ctx,
                dcmp_ctx=dcmp_ctx
            )

            # Store results
            results["original"].append(original_content)
            results["compressed"].append(cmp_tag)
            results["decompressed"].append(dcmp_tag)
            results["evaluation"].append(evaluation_text)
            results["fidelity"].append(fidelity_score)
            results["reward"].append(reward)

            initial_prompt_template = self.prompt_lib.load_prompt("compression.txt")
            initial_prompt_messages = format_prompt(initial_prompt_template, content=original_content)
            if not isinstance(initial_prompt_messages, list):
                initial_prompt_messages = [{"role": "user", "content": initial_prompt_messages}]

            results["prompt"].append(initial_prompt_messages)

            completion_conversation = []
            if isinstance(cmp_ctx, list):
                completion_conversation.extend(cmp_ctx)

            if isinstance(dcmp_ctx, list):
                if len(dcmp_ctx) > 1:
                    completion_conversation.extend(dcmp_ctx[1:])
                elif len(dcmp_ctx) == 1:
                     completion_conversation.extend(dcmp_ctx)

            results["completion"].append(completion_conversation)
            results["answer"].append(original_content)
            results["state"].append({"reward": reward})

        return results

    def _compression_reward_func(self, prompt, completion, answer, state, **kwargs) -> float:
        """
        This is a placeholder, as the reward is calculated in the main rollout loop.
        The rubric requires at least one function.
        """
        return state.get("reward", 0.0)

    def format_dataset(self,
                       dataset: Dataset,
                       system_prompt: str | None = None,
                       few_shot: List[Dict[str, Any]] | None = None,
                       question_key: str = "text",
                       answer_key: str = "answer") -> Dataset:
        """
        Override parent format_dataset to prepare data for rollouts.

        For compression, the 'prompt' is the content we want to compress.
        The `format_prompt` in the rollout methods will handle the full template.
        """
        # Use the data_key determined from the prompt template
        # if self.data_key != question_key:
        #     question_key = self.data_key
        #
        # # Add dummy answer column if it doesn't exist
        # if "answer" not in dataset.column_names:
        #     dataset = dataset.map(lambda x: {**x, "answer": ""})
        #
        # # The 'prompt' for the trainer is just the raw content.
        # # The full prompt messages are constructed inside the rollout methods.
        # # The question_key should match what format_dataset receives.
        # final_question_key = self.data_key if self.data_key in dataset.column_names else question_key
        #
        # return dataset.map(lambda x: {
        #     "prompt": x[final_question_key],
        #     "answer": x[answer_key]
        # }, num_proc=self.max_concurrent)
        return dataset

def execute_dry_run():
    """
    Handles the dry run logic entirely, without instantiating the trainer.
    """
    console.print(Rule("[bold yellow]DRY RUN MODE[/]"))

    # --- 1. Load Dataset ---
    try:
        # Load as iterable to handle large datasets efficiently
        dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train', streaming=True)
        if not isinstance(dataset, IterableDataset):
            console.print(f"[red]❌ Expected an IterableDataset for streaming, but got {type(dataset)}.[/red]")
            return
    except Exception as e:
        console.print(f"[red]❌ Failed to load dataset for dry run: {e}[/red]")
        return

    # --- 2. Prepare a few samples ---
    sample_size = 10
    try:
        # Take a few samples from the streaming dataset
        samples = list(dataset.take(sample_size))
        if len(samples) < sample_size:
            console.print(f"[yellow]Warning: Could only fetch {len(samples)} samples for the dry run.[/yellow]")
        if not samples:
            console.print("[red]Not enough data in the training set to perform a dry run.[/red]")
            return
        # Convert to a standard ArrowDataset, which supports .select() and len()
        selected_samples = ArrowDataset.from_list(samples)

    except Exception as e:
        console.print(f"[red]❌ Failed to prepare samples for dry run: {e}[/red]")
        return

    # --- 3. Create a Dry-Run-Specific Environment ---
    # This env will override get_model_response to print contexts
    # and will not initialize the evaluator client.
    dry_run_env = CompressorEnv(
        dataset=selected_samples,
        eval_dataset=None,
        alpha=0.05,
        beta=1.5,
        question_key="text",
        dry_run=True,
    )

    # --- 4. Manually trigger rollouts ---
    # The client and model are mocked as they won't be used by the overridden method.
    console.print(f"Generating {len(selected_samples)} sample contexts without running any models...")
    dry_run_env.generate(
        inputs=selected_samples, # Use the smaller, converted dataset
        client=None,
        model="dry-run-model",
        sampling_args={}
    )
    console.print(Rule("[yellow]DRY RUN COMPLETE[/]"))

def train_compressor(model_path: str, base_model_name: str):
    """Train the compressor model in a single stage."""

    # --- Training Configuration ---
    alpha = 0.05
    beta = 1.5
    max_steps = 300
    num_iterations = 4

    # Step 1: Model Loading
    console.print(f"[cyan]📦 Loading model: {model_path}[/]")

    model, tokenizer = vf.get_model_and_tokenizer(model_path)
    console.print(f"[green]✓[/] Model loaded")
    console.print()

    # Step 2: Training Configuration
    console.print(f"[cyan]⚙️ Configuring training...[/]")
    run_name = f"compressor-{base_model_name.split('/')[-1].lower()}"
    training_args = vf.grpo_defaults(run_name=run_name)
    training_args.num_iterations = num_iterations
    training_args.per_device_train_batch_size = 2
    training_args.num_generations = 8
    training_args.gradient_accumulation_steps = 2
    training_args.max_prompt_length = 1024
    training_args.max_completion_length = 2048
    training_args.max_steps = max_steps
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 50
    training_args.logging_steps = 10
    training_args.save_strategy = "steps"
    training_args.save_steps = 100
    training_args.output_dir = "outputs/compressor"

    # --- Dataset Loading ---
    console.print(f"[cyan]📊 Loading dataset...[/]")
    try:
        dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train')
        console.print("[green]✓[/] Wikipedia dataset loaded")
    except Exception as e:
        console.print(f"[red]❌ Failed to load dataset: {e}[/red]")
        raise

    # --- Prompt-driven data key ---
    # temp_prompt = PromptLibrary().load_prompt("compression.txt")
    # data_key = 'text' # Default
    # if isinstance(temp_prompt, PromptTemplate):
    #     prompt_data_var = temp_prompt.get_data_variable()
    #     if prompt_data_var:
    #         data_key = prompt_data_var.lower()

    # --- Dataset Splitting ---
    if isinstance(dataset, ArrowDataset):
        total_size = len(dataset)
        train_size = min(1450, total_size - 50) if total_size > 50 else 0

        eval_dataset = dataset.select(range(min(50, total_size)))
        if train_size > 0:
            train_dataset = dataset.select(range(50, 50 + train_size))
        else:
            train_dataset = dataset.select(range(0))  # Empty dataset
        console.print(f"[green]✓[/] Split: {len(train_dataset)} train, {len(eval_dataset)} eval")

    elif isinstance(dataset, IterableDataset):
        console.print("[yellow]⚠ Dataset is iterable, taking first 1500 items.[/]")
        items = list(dataset.take(1500))

        eval_dataset = ArrowDataset.from_list(items[:50])
        train_dataset = ArrowDataset.from_list(items[50:])
        console.print(f"[green]✓[/] Processed: {len(train_dataset)} train, {len(eval_dataset)} eval")

    else:
        console.print(f"[red]Unsupported dataset type: {type(dataset)}. Stopping.[/]")
        raise TypeError(f"Cannot process dataset of type {type(dataset)}")

    console.print()

    # Display configuration table
    train_table = Table(show_header=False, box=box.SIMPLE)
    train_table.add_column("Parameter", style="cyan", width=25)
    train_table.add_column("Value", style="white")

    train_table.add_row("Alpha (compression)", f"{alpha}")
    train_table.add_row("Beta (fidelity)", f"{beta}")
    train_table.add_row("Max steps", f"{max_steps}")
    train_table.add_row("Training samples", f"{len(train_dataset)}")
    train_table.add_row("Batch size", f"{training_args.per_device_train_batch_size} prompts → {training_args.per_device_train_batch_size * 2} rollouts")

    console.print(train_table)
    console.print()

    # --- Environment Setup ---
    console.print(f"[cyan]🏗️ Setting up environment...[/]")
    compressor_env = CompressorEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        alpha=alpha,
        beta=beta,
        max_concurrent=8,
        dry_run=False
    )
    console.print("[green]✓[/] Environment ready")

    # Step 3: Trainer Setup
    console.print(f"[cyan]🏋️ Creating trainer...[/]")
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=compressor_env,
        args=training_args,
    )
    console.print(f"[green]✓[/] Trainer ready")
    console.print()

    # Step 4: Training Execution
    console.print(f"[bold yellow]🚀 Training compressor...[/]")
    trainer.train()
    console.print(f"[green]✓[/] Training completed")
    console.print()

    # Step 5: Model Saving
    output_path = "outputs/compressor"
    console.print(f"[cyan]💾 Saving model to {output_path}...[/]")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    console.print(f"[green]✓[/] Model saved")
    console.print()

    return output_path

def main():
    """Main training function"""
    import argparse
    parser = argparse.ArgumentParser(description="Symbolic Compressor Training")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry run mode to see generated contexts without training.")
    args = parser.parse_args()

    # Clear screen for full-screen experience
    console.clear()
    console.print(Rule("[bold cyan]🧠 Symbolic Compressor Training", style="cyan"))

    # --- Model Configuration ---
    base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    if args.dry_run:
        # Determine data_key for the dry run env
        execute_dry_run()
        return

    # --- Normal Training Flow ---
    # Starting model
    current_model_path = base_model_name
    console.print(f"[dim]Base model: {base_model_name}[/]")
    console.print(Rule(style="dim"))
    console.print()

    os.makedirs("outputs", exist_ok=True)

    try:
        final_model_path = train_compressor(current_model_path, base_model_name)

        # Final completion with full-screen effect
        console.print(Rule("[bold green]🏆 TRAINING COMPLETED", style="green"))
        console.print(f"[bold green]🏆 Training finished successfully![/]")
        console.print(f"[dim]Final model saved to: {final_model_path}[/]")
        console.print(f"[dim]Usage: model, tokenizer = vf.get_model_and_tokenizer('{final_model_path}')[/]")
        console.print(Rule(style="green"))

    except Exception as e:
        console.print(Rule("[red]❌ Training Failed", style="red"))
        console.print(f"[bold red]❌ An error occurred during training: {e}[/]")
        console.print(Rule(style="red"))
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
