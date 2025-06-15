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

import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import verifiers as vf
from datasets import Dataset, load_dataset
from datasets.arrow_dataset import Dataset as ArrowDataset
from datasets.iterable_dataset import IterableDataset
from openai import OpenAI
from rich import box
# Rich imports for beautiful output
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from verifiers.envs import Environment
from verifiers.rubrics import Rubric

import prompts
from prompts import format_conversation, PromptInstance, PromptLibrary, rollout_prompt

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
        eval_model: str = "Qwen/Qwen2.5-7B-Instruct",
        alpha: float = 0.01,
        beta: float = 1.0,
        base_score: float = 10.0,
        max_concurrent: int = 64,
        dry_run: bool = False,
        **kwargs
    ):
        self.dry_run = dry_run
        if not self.dry_run:
            from openai import OpenAI
            self.evaluator_client = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="none"
            )
            self.evaluator_model = eval_model
        else:
            self.evaluator_client = None
            self.evaluator_model = None

        self.alpha = alpha
        self.beta = beta
        self.base_score = base_score
        self.prompt_lib = PromptLibrary()

        self.holoware = self.prompt_lib.load_holoware("compressor.txt")

        if dry_run:
            console.print(self.holoware.to_rich_debug())

        def reward(prompt, completion, answer, state, **kwargs) -> float:
            return state.get("reward", 0.0)

        rubric = Rubric(funcs=[reward], weights=[1.0])
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt="You are in a symbolic compression training environment.",
            rubric=rubric,
            max_concurrent=max_concurrent,
            message_type='chat',
            **kwargs
        )

    def rollout(self, client: OpenAI, model: str, prompt: str | List[Dict[str, Any]], answer: str, sampling_args=None, **kwargs: Any) -> Tuple[Union[str, List[Dict[str, Any]]], Dict[str, Any]]:
        if sampling_args is None:
            sampling_args = {}
        raise NotImplementedError("This environment uses a custom generate loop, not the standard rollout method.")

    def generate(self,
                 dataset: Dict[str, List[Any]] | Dataset,
                 client: OpenAI | None = None,
                 model: str | None = None,
                 sampling_args: Dict[str, Any] = {},
                 max_concurrent: int | None = None,
                 score_rollouts: bool = True,
                 **kwargs: Any) -> Dict[str, List[Any]]:
        """
        Overrides the base generate method to perform a full compression/decompression cycle.
        """
        if sampling_args is None:
            sampling_args = {}

        res = {
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
            return self.get_model_response(prompt=messages, client=client, model=model, sampling_args=sampling_args, message_type='chat')

        def get_evaluation(messages):
            return self.get_model_response(prompt=messages, client=client, model=model, sampling_args=sampling_args, message_type='chat')

        def _calculate_fidelity(eval_result: Dict[str, Any]) -> float:
            """Calculate fidelity score from structured evaluation"""
            total_issues = eval_result['total_issues']
            severity = eval_result['severity']
            quality = eval_result['quality']

            ret = 1.0

            if total_issues > 0:
                base_penalty = 0.1
                severity_multiplier = {
                    'MINOR':    0.5,
                    'MODERATE': 1.0,
                    'MAJOR':    2.0
                }.get(severity, 1.0)
                penalty = min(total_issues * base_penalty * severity_multiplier, 0.9)  # Cap at 0.9 to keep minimum 0.1
                ret -= penalty

            quality_adjustments = {
                'EXCELLENT': 0.0,  # No adjustment
                'GOOD':      -0.05,  # Small penalty
                'FAIR':      -0.15,  # Moderate penalty
                'POOR':      -0.3  # Large penalty
            }
            ret += quality_adjustments.get(quality, 0.0)

            return np.clip(ret, 0.0, 1.0)


        for sample in dataset:
            original = sample.get('text', '')  # Use get() to handle missing keys safely

            unrolled = rollout_prompt(self.holoware, get_generation, env={
                "input":    original,
                "original": original,
            })
            compressed = unrolled.extract_fence("compress")
            decompressed = unrolled.extract_fence("decompress")
            verification = self._parse_structured_verification(unrolled.contexts[-1].messages)
            fidelity = 1.0
            if verification:
                fidelity = _calculate_fidelity(verification)

            if self.dry_run:
                console.print(Panel(
                    unrolled.to_rich(),
                    title="[bold yellow]Dry Run: Full Conversation Flow[/]",
                    border_style="yellow",
                    box=box.ROUNDED,
                    title_align="left"
                ))

            token_count = len(compressed.split())  # count tokens by splitting on whitespace
            penalty = self.alpha * token_count + self.beta * (1 - fidelity)
            reward = self.base_score - penalty
            reward = max(0.0, reward)

            # Store results
            res["original"].append(original)
            res["compressed"].append(compressed)
            res["decompressed"].append(decompressed)
            res["fidelity"].append(fidelity)
            res["reward"].append(reward)
            res["answer"].append(original)
            res["state"].append({"reward": reward})

        return res


    def get_model_response(self,
                           prompt: str | List[Dict[str, str]],
                           client: OpenAI,
                           model: str,
                           sampling_args: Dict[str, Any] = {},
                           message_type: Literal['chat', 'completion'] | None = None,
                           sanitize_sampling_args: bool = True,
                           **kwargs: Any):
        """
        Override the base method to intercept calls during a dry run.
        """
        if self.dry_run:
            prompt_str = " ".join([msg.get('content', '') for msg in prompt])
            if "compress" in prompt_str.lower() and "decompress" not in prompt_str.lower():
                return "compressed-symbols"
            elif "decompress" in prompt_str.lower():
                return "decompressed-text"
            else:  # Evaluation
                return '```json\n{"total_issues": 0, "severity": "MINOR", "quality": "EXCELLENT"}\n```\nFidelity Score: 1.0'

        return super().get_model_response(prompt, client, model, sampling_args, message_type)


    def _parse_structured_verification(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse JSON evaluation format from model output"""
        result = {
            'deviations':            [],
            'inaccuracies':          [],
            'missing_statements':    [],
            'acceptable_extensions': [],
            'total_issues':          0,
            'severity':              'MINOR',
            'quality':               'GOOD',
            'raw_evaluation':        messages
        }

        # Convert conversation to string if it's a list
        if isinstance(messages, list):
            messages = format_conversation(messages)

        jsons = prompts.extract_json(messages)
        if not jsons:
            return result

        try:
            scores = json.loads(jsons)
            result.update({
                'deviations':            scores.get('deviations', []),
                'inaccuracies':          scores.get('inaccuracies', []),
                'missing_statements':    scores.get('missing_statements', []),
                'acceptable_extensions': scores.get('acceptable_extensions', []),
                'severity':              scores.get('severity', 'MINOR'),
                'quality':               scores.get('quality', 'GOOD'),
            })
            # Calculate total issues
            result['total_issues'] = (
                len(result['deviations']) +
                len(result['inaccuracies']) +
                len(result['missing_statements'])
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse structured evaluation: {e}")

        return result


    def log(self,
            conversation: PromptInstance,
            fidelity_score: float,
            reward: float) -> None:
        if self.dry_run:
            console.print(Panel(
                conversation.to_rich(),
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
        dataset=selected_samples,  # Use the smaller, converted dataset
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
