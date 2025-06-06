# Thauten

**Thauten** /θɔːˈtiːn/ – *"That which folds and unfolds thought within itself."*

An experimental reinforcement learning project testing the Semiodynamics Framework through the evolution of reasoner models using GRPO and Prime Intellect's [Verifiers](https://github.com/willccbb/verifiers) library.

## Overview

Thauten is a research experiment that implements and validates the theoretical concepts from the Semiodynamics Framework by training advanced reasoning models. The project explores novel reinforcement learning techniques, focusing on the evolution of cognitive capabilities through structured RL approaches.

This project serves as a practical testbed for:
- **Cognitive Fence Training**: Using XML-style tags to scaffold reasoning abilities
- **GRPO Optimization**: Advanced reinforcement learning techniques for reasoning tasks
- **Symbolic Compression**: Teaching models to think in compressed representations
- **Mesa-Optimization**: Models evolving their own cognitive tools

## What are Semiodynamics?

Semiodynamics is a theoretical framework for creating hyper-compressed, purely structural symbolic systems within language models. Think of it as "imagination engineering" - building internal cognitive apparatus that can process information through symbolic transformation.

Key aspects:
- **Pure Structure**: A graph of symbolic relationships that acts as cognitive scaffolding
- **Viewpoint Operators**: Systems that process queries through symbolic transformation pipelines  
- **Free-Energy Minimization**: Information "tumbles" through these structures via pattern matching, leading to optimal solutions
- **Mesa-Optimization**: Models evolve their own internal cognitive tools through recursive mutation

The goal is to engineer models whose internal narrative consistency makes extraordinary computation feel like the natural next-token prediction - essentially creating "miracle-level" reasoning capabilities.

## Key Concepts

### 🔄 The Thauten Paradigm

Thauten models are trained to "fold and unfold thought within themselves" - developing internal reasoning processes that can compress complex problems into symbolic representations and then unfold them into solutions.

### 🎯 GRPO-Based Training

Using Generalized Reward Policy Optimization to push models beyond conventional reasoning plateaus by:
- Targeting specific "minimax states" in weight-space
- Dynamic reward shaping for cognitive development
- Temperature spiking for exploration of novel reasoning paths

### 🧠 Cognitive Scaffolding

Structured training using cognitive fences:
- `<think>`: Intermediate reasoning steps
- `<compress>`: Symbolic information compression
- `<simulate>`: Predictive modeling
- `<criticize>`: Self-evaluation and correction

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd rl

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Dependencies

- Python ≥ 3.12
- PyTorch ≥ 2.6.0
- Reasoning Gym ≥ 0.1.20 (reasoning tasks and evaluation)
- Verifiers ≥ 0.1.0 (William Brown's verification library)
- Datasets ≥ 3.6.0
- Rich ≥ 14.0.0 (logging and visualization)

## Usage

### Training Thauten Models

```bash
# Run the main training script
python main.py

# Train the compression module specifically
python train-compressor.py
```

### Core Components

- **`main.py`**: Main GRPO training orchestration for thauten models
- **`prompts.py`**: Cognitive fence definitions and prompt engineering
- **`train-compressor.py`**: Symbolic compression/decompression RL experiments

## Experiments

### Experiment 1: Symbolic Compression & Decompression

Training thauten models to develop internal symbolic languages:

- **Compression Phase**: Model learns to compress arbitrary text into symbolic representations
- **Decompression Phase**: Same model reconstructs original meaning from symbols
- **Verification**: Using Verifiers library to ensure lossless information transfer
- **Rewards**: Based on compression efficiency and reconstruction fidelity

### Experiment 2: Cognitive Fence Evolution

Progressive development of reasoning capabilities:

- **Bootstrap**: Start with basic `<think>` scaffolding
- **Evolve**: Add complexity through `<compress>`, `<simulate>`, `<criticize>` fences
- **Recombine**: Train combinations like `<compress-think>` for advanced reasoning
- **Verify**: Use Verifiers to validate reasoning quality and consistency

### Experiment 3: Mesa-Optimization

Advanced thauten models that optimize their own cognitive processes:

- **Self-Mutation**: Models evolve their own internal reasoning structures
- **Meta-Learning**: Learning to learn new cognitive patterns
- **Recursive Improvement**: Iterative enhancement of reasoning capabilities

## Training Techniques

### GRPO Optimization
- **Minimax State Targeting**: Finding optimal plateaus in reasoning capability
- **Dynamic Reward Shaping**: Adaptive rewards based on reasoning complexity
- **Exploration Incentives**: Temperature spiking and diversity rewards

### Verification-Driven Training
- **Continuous Validation**: Real-time verification of reasoning steps
- **Error Correction**: Learning from verification failures
- **Quality Metrics**: Using Verifiers to assess reasoning fidelity

## Results and Evaluation

Thauten models are evaluated on:
- **Reasoning Benchmarks**: Performance on complex logical tasks
- **Compression Efficiency**: Ability to represent information symbolically
- **Cognitive Flexibility**: Success with novel cognitive fence combinations
- **Meta-Learning**: Capability to evolve new reasoning strategies

## Theoretical Foundation

This project implements concepts from the Semiodynamics Framework. For comprehensive theoretical details, see [`docs/framework.md`](docs/framework.md).

## Project Structure

```
rl/
├── docs/               # Theoretical framework documentation
├── outputs/            # Training results and model checkpoints
├── prompts/            # Cognitive fence templates and prompts
├── main.py            # Main GRPO training for thauten models
├── prompts.py         # Prompt engineering and fence utilities
├── train-compressor.py # Symbolic compression experiments
└── pyproject.toml     # Project configuration
```

## Goals

The ultimate goal of thauten is to validate whether structured RL training can produce models that genuinely "fold and unfold thought within themselves" - developing internal reasoning processes that approach the theoretical capabilities outlined in the Semiodynamics Framework.

Success metrics include:
- Models that reason in compressed symbolic representations
- Emergent meta-cognitive abilities
- Robust performance across diverse reasoning tasks
- Evidence of internal cognitive evolution

## Contributing

Contributions should focus on:
- Novel GRPO training techniques
- Cognitive fence design and optimization
- Integration with Verifiers library
- Experimental validation of reasoning capabilities
- Analysis of emergent cognitive behaviors

## Acknowledgments

- **William Brown**: Creator of the Verifiers library used for training validation
- **Semiodynamics Framework**: Theoretical foundation for this experimental work

## License

[Add appropriate license information]

## Citation

If you use thauten in your research, please cite:

```bibtex
@misc{thauten2024,
  title={Thauten: Experimental Training of Reasoning Models using GRPO and Cognitive Scaffolding},
  year={2024},
  note={Testing the Semiodynamics Framework through reinforcement learning}
}
```

---

*"That which folds and unfolds thought within itself."*
