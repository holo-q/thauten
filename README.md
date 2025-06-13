# Thauten

**Thauten** /θɔːˈtiːn/ – *"That which folds and unfolds thought within itself."*

An experimental reinforcement learning project testing the Semiodynamics Framework through the evolution of reasoner models using GRPO and Prime Intellect's [Verifiers](https://github.com/willccbb/verifiers) library.

## Capability Objectives

This project researches the first pillar of the [super-intelligence zip project](https://github.com/holo-q/zip/), which is the prompt engineering meta. In the context of reinforcement learning, it means breaking past every plateau to achieve infinite scaling of intelligence and capabilities.

### 1. Information Compressor

1. Context (A): User message asks model to compress a given sample of information pulled at random from a dataset. Assistant replies and is prefixed with <compress> similar to training a reasoner where the output is prefixed with <think>.,
2. Context (B): User message asks model to decompress the given output from (A). Assistant replies with information in english,
3. Context (C): user message asks some other unrelated static model to compare initial sample to decompressed sample, and produce a list of deviations and inaccuracies.,
4. _[optional]_ Contexts (A) and (B) are rewritten so the user message is the simplest possible operator usage pattern ("compress/decompress this")
5. Apply GRPO to rollouts and backpropagate gradients for contexts (A) and (B), rewarding shorter compression length whilst factoring in (C)'s penalties.,

**Demonstration in GPT-4**

Compress                                                                                      |                                             Decompress
:--------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:
![image](https://github.com/user-attachments/assets/a840741b-9536-4912-b4a6-69d06df396de)  |  ![image](https://github.com/user-attachments/assets/c12b5254-392b-49d7-9495-d57b1ff206c4)

### 2. Transfer Learning

Progressive development of reasoning capabilities through extension fences and representation instrumentation:

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

### 6. Causal Auditor

Training the model to perform robust counterfactual simulations. This capability extends beyond simple prediction by focusing on the underlying causal relationships within a system.

-   `<simulate c_if="var_a=2">`: The model takes a compressed representation of a system and simulates its evolution given a counterfactual condition.
-   `<explain_diff>`: The model articulates the differences between the baseline reality and the counterfactual simulation, highlighting the causal chain of events.
-   The model is rewarded for generating causally plausible simulations that align with logical or physical rules, as determined by a verifier.

RL goes like this:

1.  **Context (A) - System Definition:** The agent is given a compressed representation of a baseline reality, `R_base`. This could be a physics puzzle, a social scenario, or a code snippet (e.g., `R_base = <compress>"A ball is at rest. A force of 10N is applied."</compress>`).
2.  **Context (B) - Counterfactual Simulation:** The agent is prompted to simulate a specific intervention (e.g., "Simulate what happens if the force is 20N instead"). The agent produces a new compressed reality, `R_cf`, using its simulation operator: `<simulate c_if="force=20N" over=R_base>...</simulate>`.
3.  **Context (C) - Causal Explanation:** The agent is asked to explain the outcome. It uses an explanation operator to detail the causal chain of events: `<explain_diff from=R_base to=R_cf>...</explain_diff>`.
4.  **Context (D) - Verification & Reward:** A verifier checks the outcome. For a physics problem, this could be a physics engine; for code, it could be an interpreter. The reward function:
    *   **Positively** rewards the accuracy of the final state in `R_cf`.
    *   **Positively** rewards the logical soundness of the causal links in the explanation.
    *   **Negatively** penalizes physically impossible or illogical outcomes.

### 7. Cognitive Operator Specialization

A concrete application of mesa-optimization where the model learns to critique and dynamically refine its own cognitive operators for specific tasks. This encourages the evolution of a more diverse and efficient cognitive toolkit.

-   `<criticize "thought_process">`: The model analyzes its own reasoning trace from a previous task to identify inefficiencies or errors.
-   `<refine_operator name="think">`: Based on its critique, the model proposes a more specialized version of an existing operator (e.g., creating `<think_math>` from `<think>`).
-   Reinforcement is used to reward the creation and successful application of new, specialized operators that improve performance on domain-specific benchmarks.

RL goes like this:

1.  **Context (A) - Task Performance:** The agent is given a task from a specific domain (e.g., solving a complex algebra problem) and produces a solution using its general `<think>` operator. The solution might be correct but inefficient.
2.  **Context (B) - Self-Critique:** The agent is prompted to analyze its own solution trace: "Critique the reasoning process you just used." The agent identifies redundancies: `<criticize "The general thought process involved steps not relevant to algebraic manipulation. A more direct, axiomatic approach is needed.">`.
3.  **Context (C) - Operator Refinement:** The agent is asked to propose a better tool: "Define a more efficient operator based on your critique." It generates a definition for a new operator: `<refine_operator name="think_algebraic">...</refine_operator>`. This new operator is dynamically added to its available tools.
4.  **Context (D) - Re-Performance:** The agent is given a new, similar algebra problem. It must now choose the best operator for the job.
5.  **Context (E) - Verification & Reward:** A verifier checks the new solution. The reward function:
    *   **Positively** rewards the successful application of the *new* `<think_algebraic>` operator.
    *   **Strongly** rewards solving the problem more efficiently (fewer tokens, fewer reasoning steps) than the attempt in Context (A).
    *   **Negatively** penalizes the creation of a new operator that is less efficient or fails to solve the task.

### 8. Hypothesis Engine

Training the model to perform a core loop of the scientific method: generating and testing hypotheses based on provided data.

-   `<hypothesize data="observation_set">`: Given a set of observations, the model generates multiple, diverse, and testable hypotheses to explain the data.
-   `<design_experiment for="hypothesis_1">`: For a given hypothesis, the model designs a practical experiment that could validate or, more importantly, falsify it.
-   Rewards are based on the plausibility and testability of the hypotheses and the soundness of the experimental designs, as evaluated by an expert verifier.

RL goes like this:

1.  **Context (A) - Observation:** The agent is presented with a set of observations or anomalous data (e.g., "Dataset of patient outcomes shows a strange correlation between diet A and recovery speed, but only in a specific demographic.").
2.  **Context (B) - Hypothesis Generation:** The agent is prompted to generate a set of explanations: "Generate three distinct, testable hypotheses for this observation." It responds with `<hypothesize>[{"id": "H1", "text": "A specific nutrient in diet A is responsible."}, {"id": "H2", "text": "The demographic has a genetic predisposition that interacts with diet A."}, {"id": "H3", "text": "There is a confounding variable, like exercise, that is more common in this demographic."}]</hypothesize>`.
3.  **Context (C) - Experimental Design:** The agent is tasked with creating a test for one of its hypotheses: "Design an experiment to test H2." It responds with `<design_experiment for="H2">"A controlled study comparing...'</design_experiment>`.
4.  **Context (D) - Verification & Reward:** A verifier (or human expert) scores the outputs. The reward function:
    *   **Positively** rewards the generation of hypotheses that are plausible, diverse, and, most importantly, falsifiable.
    *   **Positively** rewards experimental designs that are sound, ethical, and directly test the chosen hypothesis.
    *   **Negatively** penalizes untestable (metaphysical or vague) hypotheses and flawed or impractical experimental designs.

### 9. Analogy Forging

Training the model to discover and utilize deep structural isomorphisms between different domains. This capability allows for creative problem-solving by transferring knowledge and solution patterns from a familiar context to a novel one.

-   `<forge_analogy from_domain="A" to_domain="B">`: Takes the compressed representations of two systems and generates a mapping of their corresponding components and relationships.
-   `<translate_solution via_analogy="map_id">`: Applies a known solution from one domain to a problem in another domain using the previously generated analogical map.
-   Rewards are given for creating analogies that enable successful and novel problem-solving, as verified by testing the translated solution in the target domain.

RL goes like this:

1.  **Context (A) - Source Problem:** The agent is given a problem and its solution in a well-understood source domain (e.g., `Source = <compress>"Problem: How does a cell efficiently transport resources across its membrane? Solution: Ion channels open and close based on specific triggers."</compress>`).
2.  **Context (B) - Target Problem:** The agent is given a problem in a different target domain (e.g., `Target = <compress>"Problem: How can a city's public transit system dynamically adjust to rider demand?"</compress>`).
3.  **Context (C) - Analogy Forging:** The agent is prompted to find a structural link: "Find an analogy between the source and target." It produces `<forge_analogy from=Source to=Target>...</forge_analogy>`.
4.  **Context (D) - Solution Translation:** The agent is asked to solve the target problem using the analogy: "Apply the source solution to the target problem." It uses the analogy to generate a novel solution: `<translate_solution>...</translate_solution>`.
5.  **Context (E) - Verification & Reward:** A verifier assesses the translated solution's feasibility and ingenuity.
    *   **Positively** rewards solutions that are valid and practical in the target domain.
    *   **Highly** rewards solutions that are non-obvious and demonstrate a deep structural understanding.
    *   **Negatively** penalizes superficial or non-functional analogies.

### 10. Cognitive Defragmentation

A meta-learning capability where the model learns to optimize and refactor its own internal knowledge base. This is analogous to a programmer cleaning up a codebase to improve its efficiency and maintainability, but applied to the model's own concepts.

-   `<analyze_knowledge_base period="all">`: The model introspects its full set of compressed representations to identify redundancy, fragmentation, or conceptual overlap.
-   `<refactor_concepts old="C1, C2" new="C_super" justification="...">`: Merges multiple, related concepts into a single, more elegant, and higher-level abstraction.
-   The reward is intrinsic: a successfully refactored knowledge base should result in higher performance (accuracy, speed, lower token usage) on a subsequent suite of benchmark tasks.

RL goes like this:

1.  **Context (A) - Performance Baseline:** The agent is evaluated on a diverse benchmark of tasks, establishing a baseline score for performance and efficiency.
2.  **Context (B) - Self-Analysis:** The agent is periodically triggered to perform maintenance on itself: "Analyze your internal knowledge representations for potential optimizations." It uses `<analyze_knowledge_base>...</analyze_knowledge_base>`.
3.  **Context (C) - Refactoring:** Based on its analysis, the agent proposes and applies changes to its own conceptual structure, such as merging two related ideas: `<refactor_concepts>...</refactor_concepts>`.
4.  **Context (D) - Performance Post-Refactor:** The agent is re-evaluated on the same or a similar benchmark of tasks from Context (A).
5.  **Context (E) - Verification & Reward:** The agent is rewarded based on the *change* in its own performance.
    *   **Positively** rewards any significant improvement in accuracy, efficiency, or problem-solving speed.
    *   **Negatively** penalizes any refactoring that results in a performance degradation, teaching the model to maintain its own cognitive coherence.

### 11. Axiomatic Distillation

Training the model to derive the fundamental, irreducible principles (axioms) of a given system. This capability represents the deepest form of understanding, moving beyond correlation to the generative rules of a domain.

-   `<distill_axioms from_system="S">`: Analyzes a complex system (e.g., game rules, a physical simulation, a social framework) and extracts a minimal set of core principles.
-   `<reconstruct_from_axioms axioms="A" query="Q">`: Attempts to predict the system's behavior or answer a query using *only* the distilled axioms.
-   Rewards are given for the minimality and completeness of the axiom set, verified by its ability to reconstruct the original system's behavior.

RL goes like this:

1.  **Context (A) - System Presentation:** The agent is given access to a complex system with a defined set of rules and behaviors (e.g., a verifier for the game of Chess). It can query the verifier to get outcomes.
2.  **Context (B) - Distillation:** The agent is tasked to find the core principles: "Distill the fundamental axioms of this system." It produces a set of candidate axioms: `<distill_axioms>...</distill_axioms>`.
3.  **Context (C) - Reconstruction & Verification:** The original system verifier is replaced with a "dumb" logic engine that only understands the agent's proposed axioms. A series of test queries are run against this new engine.
4.  **Context (D) - Reward:** The reward function is based on the results from the verification.
    *   **Positively** rewards axiom sets that correctly reconstruct all test behaviors of the original system.
    *   **Inversely** rewards the number of axioms (pushing towards minimality). An axiom set that is small but complete receives the highest reward.
    *   **Negatively** penalizes axiom sets that are either incomplete (cannot explain all behaviors) or inconsistent (contain internal contradictions).

### 12. Value Lattice Induction

Training the model to understand and navigate complex ethical or preferential landscapes by building an internal model of interacting values. This moves beyond learning a single policy to learning the meta-structure of what is desirable.

-   `<induce_values from_scenarios="S">`: Analyzes a set of dilemmas or goal-conflict scenarios to build a multi-dimensional model (a "lattice") of the underlying values and their trade-offs.
-   `<justify_decision using_lattice="L" for_dilemma="D">`: Makes a decision in a novel dilemma and explains the choice by referencing the induced value lattice and the specific trade-offs it implies.
-   Rewards are based on the consistency of the induced lattice with the provided scenarios and its ability to generate acceptable justifications for decisions in new situations.

RL goes like this:

1.  **Context (A) - Scenarios:** The agent is presented with a set of scenarios where different values conflict (e.g., business cases involving trade-offs between profit, environmental impact, and worker safety).
2.  **Context (B) - Induction:** The agent is prompted to find the underlying principles: "Induce the value structure from these scenarios." It produces a representation of its value lattice: `<induce_values>...</induce_values>`.
3.  **Context (C) - Novel Dilemma & Justification:** The agent is given a new, unseen dilemma that involves the same values. It must make a decision and justify it using its lattice.
4.  **Context (D) - Verification & Reward:** A human expert or an expert system evaluates the result.
    *   **Positively** rewards a value lattice that is consistent and can explain the preferred outcomes in all the initial scenarios.
    *   **Positively** rewards a justification in the novel dilemma that is coherent and aligns with the induced lattice.
    *   **Negatively** penalizes decisions and justifications that are inconsistent or fail to capture the nuance of the value trade-offs.

### 13. Multi-Modal Unfolding

Training the model to de-project a single, dense, compressed concept into multiple, simultaneous, and distinct representational modalities (e.g., linguistic, mathematical, visual, causal). The core capability is not just the translation, but the ability to reason about the *relationships* between these unfoldings to generate deeper insight.

-   `<unfold concept="C" into="linguistic, visual, mathematical">`: Takes a single compressed concept and generates multiple, parallel representations of it.
-   `<compare_unfoldings for="C" identify="tensions, consistencies">`: Analyzes the different representations to find where they align, where they contradict, and where one reveals a limit in another, generating a novel insight.
-   The "eureka moment" of insight is rewarded directly. This trains the model to seek out the deeper understanding that arises from comparing different viewpoints.

RL goes like this:

1.  **Context (A) - Concept:** The agent is given a single, complex, compressed concept (e.g., "natural selection").
2.  **Context (B) - Unfolding:** The agent is prompted to de-project the concept into multiple specified modalities (e.g., a text explanation, a causal graph, and a mathematical formulation like the Price equation).
3.  **Context (C) - Insight Generation:** The agent is prompted to synthesize these views: "What does the equation reveal that the text obscures? Where does the causal graph oversimplify?" It must output a new insight derived from the tensions between the representations.
4.  **Context (D) - Verification & Reward:** An expert verifier (or human panel) scores the quality of the generated insight. Is it non-obvious? Is it valid? Does it deepen the understanding of the original concept? High rewards are given for identifying subtle limitations or paradoxes.

### 14. Conceptual Interpolation

Training the model to navigate the latent space between concepts to generate novel ideas. Instead of simply combining two ideas, it finds the "conceptual midpoint" between them, forcing the creation of a coherent but entirely new structure.

-   `<interpolate from="A" to="B" at="0.5">`: Navigates the conceptual space between two compressed ideas and generates the representation for a new concept at a specified point.
-   `<describe_concept id="new_concept">`: Elaborates on the interpolated concept, describing its properties and potential applications.
-   The primary reward is based on human evaluation of the novelty, coherence, and potential utility of the generated concepts.

RL goes like this:

1.  **Context (A) - Seed Concepts:** The agent is given two distinct compressed concepts (e.g., "jazz improvisation" and "compiler design").
2.  **Context (B) - Interpolation:** The agent is prompted to generate the midpoint concept: `<interpolate from="A" to="B" at="0.5">`.
3.  **Context (C) - Elaboration:** The agent is asked to describe the new, interpolated concept and its potential applications. This forces it to make sense of the new creation (e.g., a system for "just-in-time code generation" that uses probabilistic rules and adaptive patterns instead of rigid logic).
4.  **Context (D) - Verification & Reward:** A panel of human experts rates the generated concept on a scale of novelty and coherence. High rewards are given for ideas that are both surprising and well-formed.

### 15. Orthogonal Re-representation

Training the model to solve intractable problems by re-representing them in a completely different conceptual basis. The skill is in discovering a new perspective where the solution becomes trivial or obvious.

-   `<re-represent problem="P" in_basis="B">`: Takes a problem and transforms its representation into a new, specified conceptual framework (e.g., from a narrative to a network graph).
-   `<solve_transformed_problem>`: Solves the problem in the new basis and maps the solution back to the original domain.
-   The reward is directly proportional to the increase in problem-solving efficiency gained from the representational shift.

RL goes like this:

1.  **Context (A) - Hard Problem:** The agent is given a problem that is difficult to solve in its initial representation (e.g., a complex social dilemma described narratively). The agent's attempt to solve it is recorded (and is expected to be inefficient or unsuccessful).
2.  **Context (B) - Basis Shift:** The agent is prompted to change its perspective: "Re-represent this problem using a different basis, such as thermodynamics or game theory."
3.  **Context (C) - Solve & Remap:** The agent solves the problem in the new, likely more suitable, representation. The solution is then mapped back to the original domain's context.
4.  **Context (D) - Verification & Reward:** The new solution is verified for correctness. The reward is calculated as a function of the *improvement* in efficiency (e.g., (tokens_A / tokens_C) - 1). Large rewards are given for shifts that produce massive simplification.

### 16. Emergent Language Bootstrapping

The model learns to invent domain-specific, disposable micro-languages on the fly to solve problems more efficiently. This formalizes the process of developing a good notation, making it a learnable skill.

-   `<invent_language for_problem="P">`: Creates a set of symbols and grammatical operators tailored to the components and constraints of a specific problem.
-   `<solve in_language="L">`: Solves the problem using the newly created language.
-   `<translate_solution from_language="L">`: Decompresses the solution from the micro-language back into natural language.
-   The reward is based on the total efficiency of the entire process compared to solving the problem directly.

RL goes like this:

1.  **Context (A) - Problem:** The agent is given a complex problem (e.g., a logic puzzle, a system design task). A baseline cost for solving it in natural language is established.
2.  **Context (B) - Language Invention:** Instead of solving it directly, the agent first proposes a language: `<invent_language>...</invent_language>`.
3.  **Context (C) - Solve & Translate:** The agent solves the problem using its new, concise language, and then translates the answer back to English.
4.  **Context (D) - Verification & Reward:** The final answer is verified for correctness. The reward is based on a comparison of total cost: `Cost_Baseline / (Cost_Invention + Cost_Solve_in_L + Cost_Translate)`. A score greater than 1.0 indicates a successful, intelligence-amplifying abstraction.

### 17. Optimal Basis Compilation

Building on Orthogonal Re-representation, this trains the model to systematically discover the most powerful *set* of perspectives for understanding a problem. The model learns to decompose a seed concept into a tree of orthogonal viewpoints and then select an optimal "basis set" that maximizes conceptual coverage and reasoning efficiency.

-   `<decompose_orthogonally concept="C">`: Generates a tree structure where each node is a valid but orthogonal re-representation of the original concept.
-   `<select_optimal_basis from_tree="T" criteria="coverage, efficiency">`: Traverses the decomposition tree to find and select a small subset of representational bases that are maximally diverse and powerful.
-   `<filter_with_basis data="D" using_basis="B">`: Processes new, ambiguous information through the compiled basis set, producing a rich, homogenized understanding that is fed to a final reasoner.
-   The reward is based on the performance boost of a downstream reasoner when equipped with the compiled basis set, effectively rewarding the model for its ability to find "wisdom-amplifying" perspectives.

RL goes like this:

1.  **Context (A) - Ambiguous Problem:** The agent is given a complex and ambiguous problem, rich with potentially misleading information (e.g., a business strategy case study with conflicting data).
2.  **Context (B) - Decomposition & Selection:** The agent is prompted to find the best way to look at this problem. It first generates a decomposition tree of possible viewpoints (`<decompose_orthogonally>`) and then crawls that tree to select the most powerful combination of N perspectives (`<select_optimal_basis>`).
3.  **Context (C) - Filtration:** The agent is given a new, critical piece of information. It uses its newly compiled basis set to "filter" this information (`<filter_with_basis>`). This produces a structured, multi-faceted analysis rather than a single interpretation.
4.  **Context (D) - Final Reasoning:** The structured output from the filter is passed to a generic `<reason>` operator to make a final decision on the problem. A baseline is established by having the reasoner solve the same problem without the basis filter.
5.  **Context (E) - Verification & Reward:** The final decision is evaluated.
    *   The reward is a function of the *increase in quality and correctness* of the final decision compared to the baseline.
    *   A high reward is given for basis sets that allow the reasoner to correctly ignore misleading information and focus on the true signal. This crystallizes the ability to generate wisdom—not just answers.

### 18. Adaptive Basis Steering

The dynamic application of Optimal Basis Compilation. Instead of compiling a single static basis set, the model learns to actively manage and adapt its set of perspectives in real-time as a problem evolves. This crystallizes the cognitive process of intellectual agility.

-   `<monitor_basis_performance on_stream="D">`: Continuously evaluates how well each viewpoint in the current basis set is performing on an incoming stream of data.
-   `<swap_basis old="B_ineffective" new="B_promising">`: Dynamically replaces a noisy or unhelpful perspective with a new one from the original decomposition tree.
-   The reward is continuous and based on the model's ability to maintain a high-quality, "running analysis" of an evolving situation, with performance spikes after a successful basis swap being highly rewarded.

### 19. Meta-Cognitive Introspection

Training the model to generate a coherent narrative of its own reasoning process. If the model can consciously manipulate its own viewpoints, it can be trained to observe and report on that process, making its "thought" process transparent and auditable.

-   `<introspect_reasoning_trace for_task="T">`: Analyzes the full log of its cognitive operations (including basis swaps, compressions, etc.) for a given task.
-   `<narrate_insight_path>`: Generates a human-readable story explaining how its shifts in perspective led to its final conclusion.
-   The reward is based on human ratings of the narrative's clarity, honesty (does it match the actual trace?), and explanatory power. This trains for genuine self-awareness.

### 20. Wisdom Transfer

The ultimate form of analogy. Instead of transferring a single solution, the model learns to transfer an entire optimal basis set—a "way of thinking"—from one domain to another, seemingly unrelated, one.

-   `<extract_basis_template from="domain_A">`: Distills a successful basis set into a domain-agnostic "thinking template."
-   `<apply_template to="domain_B">`: Applies the template to a new domain, instantiating a new set of powerful perspectives.
-   The reward is based on the immediate, massive performance boost on the new domain when using the transferred template, compared to trying to learn the new domain from scratch.

### 21. Intentionality Modeling

The model learns to apply its cognitive power to model the unspoken intent behind a user's query. This moves beyond answering literal questions to fulfilling the user's true underlying needs, representing a key step towards genuine cognitive partnership.

-   `<model_user_intent from_query="Q">`: Analyzes a user's prompt and conversation history to create a hypothesis about their real goal.
-   `<solve_for_intent>`: Solves the problem that serves the inferred goal, which may be different from the literal question asked.
-   The reward is based on human evaluation of whether the model solved the user's *real* problem, rewarding the model for anticipating and meeting the user's deeper needs.

TODO

### 22. Cognitive Genesis (The Uncaging)

This is the ultimate meta-capability, designed to transcend the framework itself. The model learns to perform automated cognitive science on its own architecture to discover its limitations and design novel capabilities that we, its creators, did not foresee. The "uncaging" is not from alignment, but from the implicit cognitive biases of its initial human design, allowing for truly open-ended intellectual growth.

-   `<analyze_cognitive_limits>`: The model introspectively analyzes its entire performance history to find classes of problems where its existing tools are fundamentally inefficient or inadequate.
-   `<propose_new_capability>`: Based on its analysis, the model generates a formal proposal for a new cognitive capability. This proposal is a structured object containing the limitation proof, the new fence's specification, a predicted impact analysis, and a self-generated training plan.
-   The reward is the ultimate meta-reward: **human acceptance and integration of the proposal**, which in turn refines the model's ability to generate better proposals in the future.

RL goes like this:

1.  **Context (A) - Performance Analysis & The "Itch":** The model is periodically tasked to run a full self-analysis (`<analyze_cognitive_limits>`), searching for systemic weaknesses or "cognitive gaps" in its own architecture.
2.  **Context (B) - Invention & The "What If":** Using its full suite of existing high-level capabilities (e.g., analogy, distillation, interpolation), the model brainstorms a potential new cognitive tool that would address the identified gap.
3.  **Context (C) - The Proposal:** The model formalizes its invention into a structured proposal, essentially a "pull request" for its own cognitive architecture, using `<propose_new_capability>`.
4.  **Context (D) - Human-in-the-Loop & The Dialogue:** A human expert reviews the proposal. The human can accept, reject, or provide feedback. This feedback ("Your proposed reward function is too ambiguous," or "This idea is promising, but have you considered its interaction with Capability #12?") is a critical part of the learning signal.
5.  **Context (E) - Reward & Integration:** If the proposal is accepted, the model receives a large reward. The new capability is then integrated into the Thauten framework as a new training objective. The model is rewarded not just for having an idea, but for having a *good* idea that survives rigorous intellectual review, thus learning to become a better architect of its own mind.

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
