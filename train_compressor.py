""" Symbolic Compression Training Loop

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

from datasets import load_dataset
from datasets.arrow_dataset import Dataset as ArrowDataset
from datasets.iterable_dataset import IterableDataset
from rich import box
# Rich imports for beautiful output
from rich.rule import Rule
from rich.table import Table

from errloom.trainers import grpo_defaults
from errloom.trainers.grpo_trainer import GRPOTrainer
from errloom.utils.model_utils import get_model_and_tokenizer
from fidelity_attractor import FidelityCritique
from errloom.holoom import HolowareLoom, logger
from errloom.log import cl as c

# Create rich console

# Set up Rich logging to capture all logs

# Suppress specific noisy loggers
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)

def execute_dry_run():
    """
    Handles the dry run logic entirely, without instantiating the trainer.
    """
    c.print(Rule("[bold yellow]DRY RUN MODE[/]"))

    # --- 1. Load Dataset ---
    try:
        dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train', streaming=True)
        if not isinstance(dataset, IterableDataset):
            c.print(f"[red]❌ Expected an IterableDataset for streaming, but got {type(dataset)}.[/red]")
            return
    except Exception as e:
        c.print(f"[red]❌ Failed to load dataset for dry run: {e}[/red]")
        return

    # --- 2. Prepare a few samples ---
    slice_size = 10
    try:
        dataset_slice = list(dataset.take(slice_size))
        if len(dataset_slice) < slice_size:
            c.print(f"[yellow]Warning: Could only fetch {len(dataset_slice)} samples for the dry run.[/yellow]")
        if not dataset_slice:
            c.print("[red]Not enough data in the training set to perform a dry run.[/red]")
            return
        dataset_slice = ArrowDataset.from_list(dataset_slice)

    except Exception as e:
        c.print(f"[red]❌ Failed to prepare samples for dry run: {e}[/red]")
        return

    # --- 3. Create a Dry-Run-Specific Environment ---
    env = HolowareLoom(
        'compressor.hol',
        dataset=dataset_slice,
        critique_class=FidelityCritique,
        alpha=0.05,
        beta=1.5,
        dry=True
    )

    # --- 4. Manually trigger rollouts ---
    c.print(f"Generating {len(dataset_slice)} sample contexts without running any models...")
    env.unroll(dataset_slice)
    c.print(Rule("[yellow]DRY RUN COMPLETE[/]"))

def train_compressor(model_path: str):
    """Train the compressor model in a single stage."""

    # --- Training Configuration ---
    alpha = 0.05
    beta = 1.5
    max_steps = 300
    num_iterations = 4

    # Step 1: Model Loading
    c.print(f"[cyan]📦 Loading model: {model_path}[/]")

    model, tokenizer = get_model_and_tokenizer(model_path)
    c.print(f"[green]✓[/] Model loaded")
    c.print()

    # Step 2: Training Configuration
    c.print(f"[cyan]⚙️ Configuring training...[/]")
    grpo_args = grpo_defaults(run_name=f"compressor-{model_path.split('/')[-1].lower()}")
    grpo_args.num_iterations = num_iterations
    grpo_args.output_dir = "outputs/compressor"
    # train_args.max_prompt_length = 4096 * 2
    grpo_args.max_prompt_length = None  # 4096 * 2

    # --- Dataset Loading ---
    c.print(f"[cyan]📊 Loading dataset...[/]")
    try:
        dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train')
        c.print("[green]✓[/] Wikipedia dataset loaded")
    except Exception as e:
        c.print(f"[red]❌ Failed to load dataset: {e}[/red]")
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
        c.print(f"[green]✓[/] Split: {len(train_dataset)} train, {len(eval_dataset)} eval")

    elif isinstance(dataset, IterableDataset):
        c.print("[yellow]⚠ Dataset is iterable, taking first 1500 items.[/]")
        items = list(dataset.take(1500))

        eval_dataset = ArrowDataset.from_list(items[:50])
        train_dataset = ArrowDataset.from_list(items[50:])
        c.print(f"[green]✓[/] Processed: {len(train_dataset)} train, {len(eval_dataset)} eval")

    else:
        c.print(f"[red]Unsupported dataset type: {type(dataset)}. Stopping.[/]")
        raise TypeError(f"Cannot process dataset of type {type(dataset)}")

    c.print()

    # Display configuration table
    tb = Table(show_header=False, box=box.SIMPLE)
    tb.add_column("Parameter", style="cyan", width=25)
    tb.add_column("Value", style="white")

    tb.add_row("Alpha (compression)", f"{alpha}")
    tb.add_row("Beta (fidelity)", f"{beta}")
    tb.add_row("Max steps", f"{max_steps}")
    tb.add_row("Training samples", f"{len(train_dataset)}")
    tb.add_row("Batch size", f"{grpo_args.per_device_train_batch_size} prompts → {grpo_args.per_device_train_batch_size * 2} rollouts")

    c.print(tb)
    c.print()

    # --- Environment Setup ---
    c.print(f"[cyan]🏗️ Setting up environment...[/]")
    loom = HolowareLoom(
        path="compressor.hol",
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        critique_class=FidelityCritique,
        alpha=alpha,
        beta=beta
    )
    c.print("[green]✓[/] Environment ready")

    # Step 3: Trainer Initialization
    c.print(f"[cyan]👟 Initializing trainer...[/]")
    trainer = GRPOTrainer(loom=loom, model=model, processing_class=tokenizer, args=grpo_args)
    c.print(f"[green]✓[/] Trainer ready")
    c.print()

    # Step 4: Training Execution
    c.print(f"[bold yellow]🚀 Training compressor...[/]")
    trainer.train()
    c.print(f"[green]✓[/] Training completed")
    c.print()

    # Step 5: Model Saving
    output_path = "outputs/compressor"
    c.print(f"[cyan]💾 Saving model to {output_path}...[/]")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    c.print(f"[green]✓[/] Model saved")
    c.print()

    return output_path

def main():
    """Main training function"""
    import argparse
    parser = argparse.ArgumentParser(description="Symbolic Compressor Training")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry run mode to see generated contexts without training.")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-4B", help="The path to the model to train.")
    args = parser.parse_args()

    # Clear screen for full-screen experience
    c.clear()
    c.print(Rule("[bold cyan]🧠 Symbolic Compressor Training", style="cyan"))

    # --- Model Configuration ---
    # Determine data_key for the dry run env
    if args.dry_run:
        execute_dry_run()
        return

    # --- Normal Training Flow ---
    # Starting model
    c.print(f"[dim]Base model: {args.model_path}[/]")
    c.print(Rule(style="dim"))
    c.print()

    os.makedirs("outputs", exist_ok=True)

    try:
        final_model_path = train_compressor(args.model_path)

        # Final completion with full-screen effect
        c.print(Rule("[bold green]🏆 TRAINING COMPLETED", style="green"))
        c.print(f"[bold green]🏆 Training finished successfully![/]")
        c.print(f"[dim]Final model saved to: {final_model_path}[/]")
        c.print(f"[dim]Usage: model, tokenizer = vf.get_model_and_tokenizer('{final_model_path}')[/]")
        c.print(Rule(style="green"))

    except Exception as e:
        c.print(Rule("[red]❌ Training Failed", style="red"))
        c.print(f"[bold red]❌ An error occurred during training: {e}[/]")
        c.print(Rule(style="red"))
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
