"""Symbolic Compression Training Loop

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
from errloom.main import main

logger = logging.getLogger("main")

if __name__ == "__main__":
    main(
        "🧠 Symbolic Compressor Training",
        default_model="",
        default_data=""
    )
