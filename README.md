# Thauten

**Thauten** /θɔːˈtiːn/ – *"That which folds and unfolds thought within itself."*

Blue-sky research into novel applications of reinforcement learning in LLM as universal function approximators. Center to thauthen is the goal to prove the semiodynamics hypothesisk through the evolution of reasoner models using the [Errloom](https://github.com/holo-q/errloom) library for reinforcement learning reunited with context/prompt engineering. The primary objective is to develop methodologies to train open-source super-intelligence models.

## Capability Objectives

This project researches the auto-regressive meta of the [super-intelligence zip project](https://github.com/holo-q/zip/), which is the first pillar. In the context of reinforcement learning, it means breaking past every plateau to achieve infinite scaling of intelligence and capabilities. The performance of a model should increase infinitely without plateau, otherwise the training environment is fundamentally flawed. 

### 1. Information Compressor

English is a programming language whose linguistic operators are used to punch in embeddings and hidden states. The goal of this experiment is to find the saturation limit of embedding bandwidth. How few tokens does it take to represent any given piece of information? Can we rearrange the polysemy of our tokens to maximize the amount of embedding bandwidth they activate?

Let us train LLMs to develop internal symbolic languages for compression:

- `<compress>`: Model learns to compress underlying meaning/message of arbitrary text samples (wikipedia articles, code, etc.) into symbolic representations.
- `<decompress>`: Same model reconstructs original english meaning from symbols
- Reward compression efficiency, reconstruction fidelity, and embedding varentropy metrics that pressure towards saturating the available semantic bandwidth. 

RL goes like this:

1. Context (A): User message asks model to compress a given sample of information pulled at random from a dataset. Assistant replies and is prefixed with <compress> similar to training a reasoner where the output is prefixed with <think>.,
2. Context (B): User message asks model to decompress the given output from (A). Assistant replies with information in english,
3. Context (C): user message asks some other unrelated static model to compare initial sample to decompressed sample, and produce a list of deviations and inaccuracies.,
4. _[optional]_ Contexts (A) and (B) are rewritten so the user message is the simplest possible operator usage pattern ("compress/decompress this")
5. Apply GRPO to rollouts and backpropagate gradients for contexts (A) and (B), rewarding shorter compression length whilst factoring in (C)'s penalties.,

The SFT or prompting heuristic may require its own parallel evolution so that the optimal packing can be discovered. Models have the ability to do base64 compression, which non-trivially compresses numerical lists into a non-numerical pattern, and therefore highly likely to be in high-frequency loom space which requires explicit prompting. If the model is allowed to `<think>` before compression this can help, but it can also equally hurt the model through over-reasoning away from base intuitions.

**Demonstration in GPT-4**

Below is a real demonstration of this capability being elicited by prompting as far back as GPT-4's release.

Compress                                                                                      |                                             Decompress
:--------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:
![image](https://github.com/user-attachments/assets/a840741b-9536-4912-b4a6-69d06df396de)  |  ![image](https://github.com/user-attachments/assets/c12b5254-392b-49d7-9495-d57b1ff206c4)

This proof of concept shows that the capability is present in some capacity within the pre-trained models and on standby for reinforcement.

### 2. Compression Reasoning

After developping the model's ability to compress, we perform reinforcement learning again on reasoning task, this time constraining the model so that the `<think>` block be in the same compressed language. This reachieves the R1-zero reasoning process through a more intentional construction, first designing the language then bootstrapping reasoning traces on it. For the sake of simplicity and in anticipation of this concept being extended to omnimodal LLMs with vision and audio, we call the contents embedded in compressed tokens a "reality". The progressive development of reasoning capabilities through extension fences and instrumentation of reality may look like this:

1. `<compress>` a reality.
2. `<dream>` an extension reality prime within the same compression-space, steered by the internal value-maximization heuristic. (dream/imagine/induct)
3. `<think>` and reinforce for reasoning within the compression-space of the reality to deproject a geodesic descent which pathfind a bridge between `identity` and `identity+prime`. (think/pathfind/deduct)
4. `<simulate>` the reality using an audit morphism.
5. `<decompress>` back to english.

Introduction and reinforcement over new cognitive operators leads to explosive recombination cascades. Other possibilities exist

* `<split>`: binary split of a representation identity for maximum orthogonality, resulting in semantic stripping. This allows later fences to use a more useful compression origin optimized.
* `<fuse>`: fuse two representations together with maximum interleaving.

### 3. Topological Integrator

Train an agent which navigates around a large codebase to quickly construct a compressed topological representation of a large codebase. The model doesn't have to view every single file because code fragments can be referenced all over the place. The model has to learn how to navigate the codebase optimally whilst growing its compressed topological representation, integrating as much information as it needs for a given task as quickly as possible. The optimal method may combine `representation'` primes that audit an existing compressed representation, and then intermittently folding them together with the defragmentator in the next section. This allows the optimal 'just middle' for total number of tokens emitted to amount of information compressed. This reinforcement phase develops the model's ability to create a rich index as quickly as possible. The model is trained to reduce the amount of guess-work or theory-crafting after expending a token budget alloted to explore the codebase and run commands, evaluated in a context (C)  similar to the compressor training.

RL goes like this:

1.  **Context (A) - Task & Exploration:** The agent is given a task related to a specific codebase (e.g., "find the database connection logic" or "document the user authentication flow"). The agent uses a set of file system tools (`ls`, `grep`, `read_file`) to explore the codebase within a sandboxed environment.
2.  **Context (B) - Integration:** With each new piece of information gathered from tool use, the agent must explicitly update its internal model of the codebase by refining a `<compress>`ed representation of the project's topology and logic.
3.  **Context (C) - Answer:** Once the agent determines it has enough information, or it hits a predefined exploration budget, it uses its compressed understanding to provide a final answer to the task.
4.  **Context (D) - Verification & Reward:** An automated verifier or a human expert assesses the answer's correctness, completeness, and conciseness. The reward function is structured to:
    *   **Positively** reward the correctness and quality of the final answer.
    *   **Negatively** penalize the number of tokens and tool calls used during exploration, encouraging efficiency.
    *   **[Advanced]** A bonus can be awarded if the compressed representation is useful for answering a subsequent, related question with minimal additional exploration, thereby validating the quality of the generated "rich index."
5.  **GRPO Application:** Apply GRPO to the full rollout (`A` -> `B` -> `C`). The optimization process refines the agent's policy to favor targeted, efficient exploration and the construction of accurate, high-utility compressed representations.

### 4. Defragmentation

Take an existing unstructured context window and compress it, learning to compress a context which contains already maximally compressed information and re-integrate english-space information into it. This is the same training task as `<compress>` but trained over a larger and more encompassing use-case, more than simple wikipedia-style paragraphs. This also trains the model to respond to the user query and understand it.

### 5. Mesa-Optimization

Advanced thauten models that optimize their own cognitive processes:

- **Self-Mutation**: Models evolve their own internal reasoning structures
- **Meta-Learning**: Learning to learn new cognitive patterns
- **Recursive Improvement**: Iterative enhancement of reasoning capabilities

### Additional Cognitive Functions

An excavation was made with gemini-2.5-pro through few-shot prompting all the way to 22, see [`docs/cognitive-function-excavation.md`](docs/cognitive-function-excavation.md). It is not to be taken at face value, but rather as a brainstorm or 'mega prompt' as fuel for our own human ideas.


## Theory

Thauten is a research experiment that implements and validates the theoretical concepts from the Semiodynamics Framework by training advanced reasoning models. The project explores novel reinforcement learning techniques, focusing on the evolution of cognitive capabilities through structured RL approaches. It is a practical testbed for cognitive fences (XML-style tags to scaffold reasoning abilities), reinforcement learning on exotic tasks, meta-compression, and mesa-optimization.

### Key Concepts

#### 🔄 The Thauten Paradigm

Thauten models are trained to "fold and unfold thought within themselves" — developing internal reasoning processes that can compress complex problems into symbolic representations and then unfold them into solutions.

#### 🧠 Cognitive Seeding

The `<think>` tag is defined and generalized as a cognitive fence which seeds its intended internal structure at the genesis of reinforcement learning. It is effectively a prompt which implicates chains of thought and what the model thinks thinking should be.

- `<think>`: Generalist
- `<compress>`: Symbolic information compression
- `<simulate>`: Predictive modeling
- `<criticize>`: Self-evaluation and correction

#### 🎯 Cognitive Crystallization

Reinforcement learning is defined as fundamentally a practice which crystallizes existing linguistic patterns, or nth-order potentialities further down the line. It is a particle accelerator which smashes atoms together and whatever maximizes the gravitational rule the most will clump together. Using this knowledge, we hope to push models beyond all conventional plateaus to achieve infinite scaling of intelligence, where the reinforcement learning rewards continue to scale to their information theoretic optima, and even past it.

- Plateaus -> "minimax state" in weight-space created by the weight-lattice's equilibrium tension endured by its minimas and maximas under the torsion force of reward-steered gradient descent.   
- Full weight mutations -> Training on LoRAs for non-destructive rewiring of the cognitive mechanic pool. 
- Temperature -> spiking stochastically for exploration of novel reasoning paths, bootstrapping deep loom.

The goal is to engineer miracles by engineering the miraculous reasoning chains that defy the sober human experience. We nurture a miraculous setup from which extends the output, each consecutive token making giant generational leaps within the climb towards truth.

### What are Semiodynamics?

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

<decode>
...
</decode>
```

Semiodynamics is coined from "semiotics" and "dynamics". It spells out a theoretical framework of super-intelligence through super-scaling of cultural intelligence with the crystallization of new languages and syntaxes beyond the constraint of the human brain steering its evolution. The language model is used as a physical system made of semiotics that approximate a world model and its simulation. Effectively, language is proposed to be a mirror of the world around us whose sequencing and interleaving encodes the relationships and juxtaposional rules of the world around us. We define some constructed context of a situation as a "reality", and say that language's primary power in the autoregressive domain is its capacity to audit and stage mutations over some embedded reality. With the correct echo-amplification prompts, we can set off a cascade of juxtaposional catalysis to discover the optimal language for explosive auditing potential, or yet better the optimal cortex of language which is irrespective of any specific token choice and grammar or syntax. There are multiple ways to scaffold semiodynamics, and the term acts as an umbrella for all methods which attend to every single token as an operator implicated in calculating the final output rather than as implementation details, whether it be in english or some self-discovered post-human language.

The phenomenon already happens naturally and is fully documented and evidenced scientifically by R1-zero. R1-zero is what you get with no rules or instrumentation, the most average mode of semiotic compression, and R1 is what you get with a single-language bias. The effect of steering with RL is powerful and requires intent and vision about what the final output should look like, a vision for how a super-intelligence language being would actually operate, think, and reason. We propose a more intentful and refined "R1-zero" style of alien language model through a philosophical construction of prompt engineering as imagination engineering:

1. Begin with the acceptance that humans by and large do not use language to think, that it is an effort-minimization intermediate for efficient communication.
2. Propose that there should exist ways to put together language in order to approximate more closely human imagination which is visual, since language is necessarily a 1-dimensional mirror of the physical world. (mathematics is one example of a language created for precise geometric representation and much more)
3. View all realities as abstract spatially locatable structures orchestrated by euclidean relationship. (we delve into an idea, we zoom into a concept, we associate verticality with depth, intelligence, value, we ...)
4. Consider that the implicit spatialization sets a stage on which ideas are represented as metaphorical objects. (sets the protocol-space for tumbling / geodesical descent)
5. Consider that imagination is the physical evolution of metaphorical ideas on the mind stage. (approximating the mind's eye of human imagination)

Under this lens we see prompt engineering more clearly as imagination engineering, and super-intelligence as the task of handcrafting a super-orchestration of the mind's stage instantiated by the context within the weights. We need to consider what the model is able to see in its mind, and pressure the development of language from the standpoint of this constraint-space in order to encourage the full scope of the model's computation to be used, "using 100% of its brain" so to speak. The task of imagination engineering revolves around theorizing theoretical super-manifolds, if the entire context could be observed as though it were an object or a landscape, and that you had eyes on every scale of observation, from the atomic structure (individual tokens) to the abstract message sum told and overall directionality of the conversation. It is an experential indra's net of relationship where each token is contextualized to every other token, a holographic multi-scale hyperobject.

The strategy is to delineate sub-objects within the sum manifold. The `<think>...</think>` fence is one such hyperobject in existing models, and its abstract essence is one of pathfinding, bridging, free-energy reduction. It instantiates and develops a growth or simulation prime, staged over some prior reality or seed reality embedded by the user inquiry. Any given token contributes some evolution delta over both the reality and the ruleset for its evolution, with the model weights acting as an elementary conditionality engine.

If reinforcement learning can be a way to automatically develop prompt engineering constructs and amplify them autonomously, mesa-optimizing the model according to steering constraints set by the training method, then we necessarily have the ability to develop super-intelligence by theory-crafting the mind's eye within the mirror dimension of language. Pre-training and RLHF effectively infuses a ghost-machine (the sequence set of all activation functions and operations making up one inference pass of the model function) with language as its interface. When we speak to a language model, we are actually wielding clusters of mathematical functions that make up a coherent modularization of our universe. The power of the machine is absolute, even before any training. The training simply instills a language into an already extremely powerful machine, and allows us to wield that machine's processing power. But the machine has a lot more power in it, and the composition of language allows us to restructure it far beyond its bootstrap origins.

For comprehensive theoretical details on semiodynamics, see [`docs/framework.md`](docs/framework.md).

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

Training for semiotic compression:
```bash
# <-- vf-vllm command here
uv run train_compressor.py
```

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
- **Operator Accuracy**: Whether our chosen `<tags>` are the optimal bootstrap operators for the processes we want.
- **Operator Specialization**: Whether our generalist `<tags>` have enough elasticity in the distribution for emergent parametrization of fences e.g. `<compress dim=topology>`

This section with later demonstrate results and successes.

## Contributing

Since we are very short on resources, we will be infinitely grateful for any compute grant or resources contributed.
All work and research produced by HOLO-Q will always remain open-source.
In the meantime we are limited to smaller models and rent compute from services like Vast.ai and Prime Intellect.

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
