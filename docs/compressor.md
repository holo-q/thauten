**Title: Training a Symbolic-Invariant Semantic Compressor for LLM Cognition**

---

**Abstract:**
We propose a structured training scaffold for a language model-based semantic compressor in which compression, decompression, and evaluation occur in three distinct conversational contexts. Model A performs compression in context A, Model B performs decompression in context B, and Model C—a frozen evaluator—assesses fidelity in context C. The reward derived from context C is used to update Model A and Model B. This loop develops symbolic compression and decompression as internal reasoning capabilities, priming the model for semiodynamic cognition and modular, composable thought structures.

---

**1. Background & Motivation**

Modern LLMs operate over uncompressed surface forms, resulting in verbose, inefficient, and less structured reasoning. Symbolic compression, by contrast, offers a pathway to scalable, modular cognitive operations. This work introduces a method to teach models how to represent ideas compactly, reconstruct them reliably, and reason over these compressed forms.

---

**2. Training Loop Overview**

### 2.1 Multi-Context Architecture

* **Context A (Compression):** Model A receives the original content and outputs a compressed symbolic representation.
* **Context B (Decompression):** Model B receives only the output of context A and reconstructs the original input.
* **Context C (Evaluation):** Model C (frozen) receives both the original content and the decompressed output. It identifies information loss and produces a scalar reward.

Only contexts A and B participate in gradient updates. Context C is discarded after reward emission.

### 2.2 Canonical Example

**Context A:**

```xml
<input>
  The original content (e.g., problem statement, reasoning trace)
</input>
<compress>
  [Model A output]
</compress>
```

**Context B:**

```xml
<compress>
  [Copied from Context A output]
</compress>
<decompress>
  [Model B output]
</decompress>
```

**Context C:**

```xml
<input>
  [Original from Context A]
</input>
<decompress>
  [Copied from Context B output]
</decompress>
<evaluate>
  [Model C output: differences, loss, summary]
</evaluate>
```

---

**3. Reward Function**

Model C produces a scalar reward signal based on two competing factors:

* **Compression Gain:** Fewer tokens in `<compress>` result in a higher score.
* **Fidelity Penalty:** Loss of meaning or detail results in lower scores.

Let:

* `T_comp` = token count in `<compress>`
* `Δ_info` = semantic difference identified by Model C

```
Reward = BaseScore - α * T_comp - β * Δ_info
```

The reward is backpropagated to update both Model A and Model B. Model C remains frozen.

---

**4. Training Stages**

**Stage 1: Identity Pretraining**
Train the model in a passthrough mode: compress → decompress → match original. This builds basic alignment.

**Stage 2: Structured Compression**
Introduce content with mild redundancy. Train to reduce it to abstract symbols or abbreviations.

**Stage 3: Freeform Compression**
Allow the model to learn its own symbolic patterns. Inputs include multi-step reasoning, arguments, and stories.

**Stage 4: Compression-First Cognition**
Operate directly on compressed forms. Reasoning is performed in compressed space with `<think>` annotations. This phase reinforces the utility of internal symbolic representations.

---

**5. Benefits**

* **Symbolic Reasoning:** Induces latent structured cognition.
* **Efficiency:** Reduces context size without information loss.
* **Modularity:** Enables plug-and-play cognitive units.
* **Fidelity Auditing:** External evaluation improves interpretability.
* **Scaffolding Semiodynamics:** Prepares the model for dynamic symbolic systems.

---

**6. Future Directions**

* Use task transfer to generalize compression across domains.
* Couple compressed memory retrieval with episodic agents.
* Reinforce decoding under adversarial input perturbation.
* Evaluate symbol reuse, compositionality, and emergence.

---

**7. Conclusion**

This scaffold trains LLMs to reason over structured compressed representations using distinct training contexts. The tri-contextual loop supports gradient flow only through Models A and B, while Model C enforces fidelity and abstraction. This establishes a foundation for modular cognitive operations and semiodynamic symbolic evolution.
