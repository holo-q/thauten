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

from errloom import CommModel, discovery
from errloom.holoom import HolowareLoom
from thauten.fidelity_attractor import FidelityCritique
from errloom.utils.model_utils import get_model_and_tokenizer

# Suppress specific noisy loggers
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)

logger = logging.getLogger("main")
discovery.crawl_package("thauten", [CommModel])

def log(*s):
    logger.info(s)

def logc():
    # c.clear()
    pass

def logl(*s):
    logger.info(s)
    logger.info("")

def execute_dry_run(n: int):
    """
    Handles the dry run logic entirely, without instantiating the trainer.
    """
    log(Rule("[bold yellow]DRY RUN MODE[/]"))

    # --- 1. Load Dataset ---
    try:
        dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train', streaming=True)
        if not isinstance(dataset, IterableDataset):
            log(f"[red]❌ Expected an IterableDataset for streaming, but got {type(dataset)}.[/red]")
            return
    except Exception as e:
        log(f"[red]❌ Failed to load dataset for dry run: {e}[/red]")
        return

    # --- 2. Prepare a few samples ---
    try:
        data = list(dataset.take(n))
        if len(data) < n:
            log(f"[yellow]Warning: Could only fetch {len(data)} samples for the dry run.[/yellow]")
        if not data:
            log("[red]Not enough data in the training set to perform a dry run.[/red]")
            return
        data = ArrowDataset.from_list(data)

    except Exception as e:
        log(f"[red]❌ Failed to prepare samples for dry run: {e}[/red]")
        return

    # --- 3. Create a Dry-Run-Specific Environment ---
    env = HolowareLoom(
        'compressor.hol',
        dataset=data,
        critique_class=FidelityCritique,
        alpha=0.05,
        beta=1.5,
        dry=True
    )

    # --- 4. Manually trigger rollouts ---
    log(f"Generating {len(data)} sample contexts without running any models...")
    env.unroll(data)
    log(Rule("[yellow]DRY RUN COMPLETE[/]"))

def train_compressor(model_path: str):
    """Train the compressor model in a single stage."""

    # --- Training Configuration ---
    alpha = 0.05
    beta = 1.5
    max_steps = 300
    num_iterations = 4

    log(f"[cyan]📦 Loading model: {model_path}[/]")
    model, tokenizer = get_model_and_tokenizer(model_path)

    log(f"[cyan]⚙️ Configuring training...[/]")
    from errloom.trainers import grpo_defaults
    grpo_args = grpo_defaults(run_name=f"compressor-{model_path.split('/')[-1].lower()}")
    grpo_args.num_iterations = num_iterations
    grpo_args.output_dir = "outputs/compressor"
    grpo_args.max_prompt_length = None  # 4096 * 2

    log(f"[cyan]📊 Loading dataset...[/]")
    try:
        dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train')
        log("[green]✓[/]")
    except Exception as e:
        log(f"[red]❌ Failed to load dataset: {e}[/red]")
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
        log(f"[green]✓[/] Split: {len(train_dataset)} train, {len(eval_dataset)} eval")

    elif isinstance(dataset, IterableDataset):
        log("[yellow]⚠ Dataset is iterable, taking first 1500 items.[/]")
        items = list(dataset.take(1500))

        eval_dataset = ArrowDataset.from_list(items[:50])
        train_dataset = ArrowDataset.from_list(items[50:])
        log(f"[green]✓[/] Processed: {len(train_dataset)} train, {len(eval_dataset)} eval")

    else:
        log(f"[red]Unsupported dataset type: {type(dataset)}. Stopping.[/]")
        raise TypeError(f"Cannot process dataset of type {type(dataset)}")

    log()

    # Display configuration table
    tb = Table(show_header=False, box=box.SIMPLE)
    tb.add_column("Parameter", style="cyan", width=25)
    tb.add_column("Value", style="white")

    tb.add_row("Alpha (compression)", f"{alpha}")
    tb.add_row("Beta (fidelity)", f"{beta}")
    tb.add_row("Max steps", f"{max_steps}")
    tb.add_row("Training samples", f"{len(train_dataset)}")
    tb.add_row("Batch size", f"{grpo_args.per_device_train_batch_size} prompts → {grpo_args.per_device_train_batch_size * 2} rollouts")

    logl(tb)

    # --- Environment Setup ---
    log(f"[cyan]🏗️ Setting up environment...[/]")
    loom = HolowareLoom(
        path="hol/compressor.hol",
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        critique_class=FidelityCritique,
        alpha=alpha,
        beta=beta
    )
    logl("[green]✓[/] Environment ready")

    # Step 3: Trainer Initialization
    log(f"[cyan]👟 Initializing trainer...[/]")
    from errloom.trainers.grpo_trainer import GRPOTrainer
    trainer = GRPOTrainer(loom=loom, model=model, processing_class=tokenizer, args=grpo_args)
    logl(f"[green]✓[/] Trainer ready")

    # Step 4: Training Execution
    log(f"[bold yellow]🚀 Training compressor...[/]")
    trainer.train()
    logl(f"[green]✓[/] Training completed")

    # Step 5: Model Saving
    output_path = "outputs/compressor"
    log(f"[cyan]💾 Saving model to {output_path}...[/]")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logl(f"[green]✓[/] Model saved")

    return output_path

def main():
    """Main training function"""
    import argparse
    parser = argparse.ArgumentParser(description="Symbolic Compressor Training")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry run mode to see generated contexts without training.")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-4B", help="The path to the model to train.")
    parser.add_argument("--n", type=int, default=10, help="How many dataset rows to process with the loom.")
    args = parser.parse_args()

    # Clear screen for full-screen experience
    # logc()
    log(Rule("[bold cyan]🧠 Symbolic Compressor Training", style="cyan"))

    # --- Model Configuration ---
    # Determine data_key for the dry run env
    if args.dry_run:
        execute_dry_run(args.n)
        return

    # --- Normal Training Flow ---
    # Starting model
    log(f"[dim]Base model: {args.model_path}[/]")
    log(Rule(style="dim"))
    log()

    os.makedirs("outputs", exist_ok=True)

    try:
        final_model_path = train_compressor(args.model_path)

        # Final completion with full-screen effect
        log(Rule("[bold green]🏆 TRAINING COMPLETED", style="green"))
        log(f"[bold green]🏆 Training finished successfully![/]")
        log(f"[dim]Final model saved to: {final_model_path}[/]")
        log(f"[dim]Usage: model, tokenizer = vf.get_model_and_tokenizer('{final_model_path}')[/]")
        log(Rule(style="green"))

    except Exception as e:
        log(Rule("[red]❌ Training Failed", style="red"))
        log(f"[bold red]❌ An error occurred during training: {e}[/]")
        log(Rule(style="red"))
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
