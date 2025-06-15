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
from typing import List, Literal

import numpy as np
import verifiers as vf
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as ArrowDataset
from datasets.iterable_dataset import IterableDataset
from pydantic import Field
from rich import box
# Rich imports for beautiful output
from rich.rule import Rule
from rich.table import Table

from holoware_env import HolowareEnv, logger, VerificationModel
from log import console

# Create rich console

# Set up Rich logging to capture all logs

# Suppress specific noisy loggers
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)

class FidelityEvaluation(VerificationModel):
    """
    A structured representation of a fidelity evaluation, comparing an original
    text with its decompressed counterpart.
    """
    deviations: List[str] = Field(
        default=[],
        description="List of factual changes or alterations from the original."
    )
    inaccuracies: List[str] = Field(
        default=[],
        description="List of incorrect information introduced in the decompressed text."
    )
    missing_statements: List[str] = Field(
        default=[],
        description="List of important information from the original that was lost."
    )
    acceptable_extensions: List[str] = Field(
        default=[],
        description="List of valid elaborations or extensions that don't contradict the original."
    )
    severity: Literal["MINOR", "MODERATE", "MAJOR"] = Field(
        description="The overall severity of the issues found."
    )
    quality: Literal["EXCELLENT", "GOOD", "FAIR", "POOR"] = Field(
        description="The overall quality of the decompression."
    )

    @property
    def total_issues(self) -> int:
        """Calculates the total number of issues found."""
        return len(self.deviations) + len(self.inaccuracies) + len(self.missing_statements)

    def get_verification_score(self) -> float:
        """
        Calculates a numerical fidelity score based on the evaluation fields.
        The score ranges from 0.0 to 1.0.
        """
        fid = 1.0

        if self.total_issues > 0:
            base_penalty = 0.1
            severity_multiplier = {
                'MINOR':    0.5,
                'MODERATE': 1.0,
                'MAJOR':    2.0
            }.get(self.severity, 1.0)
            penalty = min(self.total_issues * base_penalty * severity_multiplier, 0.9)
            fid -= penalty

        quality_adjustments = {
            'EXCELLENT': 0.0,
            'GOOD':      -0.05,
            'FAIR':      -0.15,
            'POOR':      -0.3
        }
        fid += quality_adjustments.get(self.quality, 0.0)

        return np.clip(fid, 0.0, 1.0)

def execute_dry_run():
    """
    Handles the dry run logic entirely, without instantiating the trainer.
    """
    console.print(Rule("[bold yellow]DRY RUN MODE[/]"))

    # --- 1. Load Dataset ---
    try:
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
        samples = list(dataset.take(sample_size))
        if len(samples) < sample_size:
            console.print(f"[yellow]Warning: Could only fetch {len(samples)} samples for the dry run.[/yellow]")
        if not samples:
            console.print("[red]Not enough data in the training set to perform a dry run.[/red]")
            return
        selected_samples = ArrowDataset.from_list(samples)

    except Exception as e:
        console.print(f"[red]❌ Failed to prepare samples for dry run: {e}[/red]")
        return

    # --- 3. Create a Dry-Run-Specific Environment ---
    dry_run_env = HolowareEnv(
        path="compressor.hol",
        dataset=selected_samples,
        score_class=FidelityEvaluation,
        eval_dataset=None,
        alpha=0.05,
        beta=1.5,
        dry_run=True,
    )

    # --- 4. Manually trigger rollouts ---
    console.print(f"Generating {len(selected_samples)} sample contexts without running any models...")
    dry_run_env.generate(
        dataset=selected_samples,
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
    train_args = vf.grpo_defaults(run_name=run_name)
    train_args.num_iterations = num_iterations
    train_args.output_dir = "outputs/compressor"

    # --- Dataset Loading ---
    console.print(f"[cyan]📊 Loading dataset...[/]")
    try:
        dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train')
        console.print("[green]✓[/] Wikipedia dataset loaded")
    except Exception as e:
        console.print(f"[red]❌ Failed to load dataset: {e}[/red]")
        raise

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
    train_table.add_row("Batch size", f"{train_args.per_device_train_batch_size} prompts → {train_args.per_device_train_batch_size * 2} rollouts")

    console.print(train_table)
    console.print()

    # --- Environment Setup ---
    console.print(f"[cyan]🏗️ Setting up environment...[/]")
    compressor_env = HolowareEnv(
        path="compressor.hol",
        dataset=train_dataset,
        score_class=FidelityEvaluation,
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
        args=train_args,
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
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"

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
