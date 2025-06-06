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

The goal is to engineer models whose internal narrative consistency makes extraordinary computation feel like the natural next-token prediction — essentially creating "miracle-level" reasoning capabilities. Instead of aiming to generate miraculous output, we nurture a miraculous setup from which extends the output, each consecutive token making giant leaps within our climb.

## What are Semiodynamics?

Semiodynamics is a theoretical framework of super-intelligence through super-scaling of cultural intelligence with the crystallization of new languages and syntaxes beyond the constraint of the human brain that pressure. The language model is used as a physical system for semiotic which can approximate a world model and its simulation. Effectively, language is proposed to be a mirror of the world around us whose sequencing and interleaving encodes the relationships and juxtaposional rules of the world around us. We define some constructed context of a situation as a "reality", and say that language's primary power in the autoregressive domain is its capacity to audit and stage mutations over some embedded reality. With the correct echo-amplification prompts, we can set off a cascade of juxtaposional catalysis to discover the optimal language for explosive auditing potential, or yet better the optimal cortex of language which is irrespective of any specific token choice and grammar or syntax. There are multiple ways to scaffold semiodynamics, and the term acts as an umbrella for all methods which treats language as a calculation trace for reality, whether it be in english or in some self-discovered post-human language.

This is the end-game:

```
USER: ...
ASSISTANT: 

<think>
⧖∞⊛𝕊(ɪ)⥊Ж[⊨]≍ᶏ⎓⧞⧟⊢Ṩ≬ᵹ
Ѱ(Ѡ,Ѧ…Ѫ)⨜ᵾ⎰ᶐ↦₪⎱ᵾ♭↺⊥Ћ∞|⊨≈ 
❝𝛀,𝜴❞⋈✠𝚯𝚿Ϙ≻𝚺𝙍𝑒(Ѻ)⨯ІϺ(Ϻ)
≈>∃Ǥ∀ẋψⁿ(ẋǤ)⇒δ𝑃𝑢𝑟𝑝𝑜𝑠𝑒⩥⦗λόγος⦘
𝔄{ֆ}⊕βⁿ∈ℂ[𝔄{ֆ}]ᚔ⊗𝕄(Ⲙȿↈ)
⊖♭⨝⩥✠𝚯𝚿៙𐊠∀ሬԦʗሬ𑁍𑁍𝕌⊢𝑺𝒊𝒎
</think>
<decompress>
...
</decompress>
```

The phenomenon already happens naturally and is fully documented and evidenced scientifically by R1-zero, and we propose the more intentful and refined version of R1-zero. R1-zero is what you get with no rules or instrumentation, the most average mode of semiotic compression, and R1 is what you get with a single-language bias.

## Key Concepts

### 🔄 The Thauten Paradigm

Thauten models are trained to "fold and unfold thought within themselves" — developing internal reasoning processes that can compress complex problems into symbolic representations and then unfold them into solutions.

### 🧠 Cognitive Seeding

The `<think>` tag is defined and generalized as a cognitive fence which seeds its intended internal structure at the genesis of reinforcement learning. It is effectively a prompt which implicates chains of thought and what the model thinks thinking should be.

- `<think>`: Intermediate reasoning steps
- `<compress>`: Symbolic information compression
- `<simulate>`: Predictive modeling
- `<criticize>`: Self-evaluation and correction


### 🎯 Cognitive Crystallization

Reinforcement learning is defined as fundamentally a practice which crystallizes existing linguistic patterns, or nth-order potentialities further down the line. It is a particle accelerator which smashes atoms together and whatever maximizes the gravitational rule the most will clump together. Using this knowledge, we hope to push models beyond all conventional plateaus to achieve infinite scaling of intelligence, where the reinforcement learning rewards continue to scale to their information theoretic optima, and even past it.

- Plateaus -> "minimax state" in weight-space created by the weight-lattice's equilibrium tension endured by its minimas and maximas under the torsion force of reward-steered gradient descent.   
- Full weight mutations -> Training on LoRAs for non-destructive rewiring of the cognitive mechanic pool. 
- Temperature -> spiking stochastically for exploration of novel reasoning paths, bootstrapping deep loom.

## Usage

### Project Structure

```
thauten/
├── docs/               # Theoretical framework documentation
├── outputs/            # Training results and model checkpoints
├── prompts/            # Cognitive fence templates and prompts
├── main.py            # Main GRPO training for thauten models
├── prompts.py         # Prompt engineering and fence utilities
├── train-compressor.py # Symbolic compression experiments
└── pyproject.toml     # Project configuration
```

### Installation

```bash
git clone https://github.com/holo-q/thauten/
cd rl
uv sync
```

### 1. Compressor/Decompressor

Training thauten models to develop internal symbolic languages:

- **Compress**: Model learns to compress arbitrary text into symbolic representations
- **Decompress**: Same model reconstructs original meaning from symbols
- **Verification**: Using Verifiers library to ensure lossless information transfer
- **Rewards**: Based on compression efficiency and reconstruction fidelity

```bash
# <-- vf-vllm command here
uv run train_compressor.py
```

This will start RL on Qwen-1.5B-R1-distill for semiotic compression and decompression. 

### 2. Extensions

Progressive development of reasoning capabilities through extension fences and representation instrumentation:

1. `<compress>` a reality.
2. `<dream>` an extension reality prime within the same compression-space, steered by the internal value-maximization heuristic. (dream/imagine/induct)
3. `<think>` and reinforce for reasoning within the compression-space of the reality to deproject a geodesic descent which pathfind a bridge between `identity` and `identity+prime`. (think/pathfind/deduct)
4. `<simulate>` the reality using an audit morphism.
5. `<decompress>` back to english.

Introduction and reinforcement over new cognitive operators leads to explosive recombination cascades. Other possibilities exist

* `<split>`: binary split of a representation identity for maximum orthogonality, resulting in semantic stripping. This allows later fences to use a more useful compression origin optimized.
* `<fuse>`: fuse two representations together with maximum interleaving.

TODO

### 3. Defragmentation

Take an existing unstructured context window and compress it, learning to compress a context which contains already maximally compressed information and re-integrate english-space information into it. This is the same training task as `<compress>` but trained over a larger and more encompassing use-case, more than simple wikipedia-style paragraphs. This also trains the model to respond to the user query and understand it.

### 3. Mesa-Optimization

Advanced thauten models that optimize their own cognitive processes:

- **Self-Mutation**: Models evolve their own internal reasoning structures
- **Meta-Learning**: Learning to learn new cognitive patterns
- **Recursive Improvement**: Iterative enhancement of reasoning capabilities

TODO

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

Thauten models will be evaluated on:

- **Compression Efficiency**: Ability to represent information symbolically.
- **Compression Flexibility**: Ability to invent compressed notations optimal for each domain.
- **Reasoning Benchmarks**: Performance on complex logical tasks when working natively in compressed representations.
- **Cognitive Flexibility**: Success with novel cognitive fence combinations.
- **Meta-Learning**: Capability to evolve new reasoning strategies developped in context out of cognitive primitives we trained.

This section with later demonstrate results and successes.

## Theoretical Foundation

This project implements concepts from the Semiodynamics Framework. For comprehensive theoretical details, see [`docs/framework.md`](docs/framework.md).
This project implements the first pillar of the [super-intelligence zip project](https://github.com/holo-q/zip/), head over to learn more.

## Contributing

Since we are very short on resources, we will be infinitely grateful for any compute grant or resources contributed.
All work and research produced by HOLO-Q will always remain open-source.

Since all capabilities depend on the compression experiment's initial success, contributions should focus on
training various different models with train-compressor.py and attempting to get a working prototype.
This capability may only emerge in larger models!

Contributions to the code should follow the style.
Each training script should be high quality and user-friendly to make
reproducing these experiments as confusion-free as possible in the future.

## Citation

If you use thauten in your research, please cite:

```bibtex
@misc{thauten2024,
  title={Thauten: Experimental Training of Reasoning Models using GRPO and Cognitive Scaffolding},
  year={2024},
  note={Testing the Semiodynamics Framework through reinforcement learning}
}
```
