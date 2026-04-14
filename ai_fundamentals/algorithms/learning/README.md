# On Becoming Dangerous
### A personal doctrine for AI research

---

## The Mission

Build the mathematical and algorithmic tools to contribute to AI systems that are:
- **Energy efficient** — doing more with less computation
- **Human aligned** — modeling what humans actually value
- **Universally communicative** — capable of representing meaning across any possible system of expression

---

## The North Star

> **What is the minimum mathematical structure required for any intelligent system to represent, optimize, and communicate meaning?**

At every level of study, return to these three questions:
1. What is being represented?
2. What is being optimized?
3. What uncertainty is being reduced?

---

## The Learning Loop
*Every concept. Every time. No exceptions.*

```
FORMALIZE   →   What is the objective function?
     ↓
DERIVE      →   Work through the math by hand
     ↓
IMPLEMENT   →   Code it from scratch
     ↓
REFLECT     →   What assumption is hidden?
                What would break this?
                What does this look like in a different representational system?
     ↓
(return to FORMALIZE with a harder question)
```

**Be rigid about this method. Be flexible about schedule.**

---

## The Research Vision (to be sharpened)

> **Combinatorial structure is underutilized as a mathematical foundation for efficient and general AI.**

Connects to: Kolmogorov complexity, algorithmic information theory, information bottleneck,
tokenization efficiency, universal communication structure.

*Journal every question that surfaces. Especially the ones that seem too big.*

---

## The Note-Taking Doctrine

Write every note as if someone will learn from it after you are gone.

This is not sentiment. It is a technical discipline. It forces:
- Precision over shorthand
- Explanation over assumption
- Insight over transcription

Every notebook is a contribution, not a personal reference. Treat it accordingly.

---

## Honest Self-Assessment

| Area | Current State | Priority |
|---|---|---|
| Probability Theory | **Strongest area.** Multiple notebooks. Currently consolidating into one definitive notebook. | Deepen and formalize |
| Linear Algebra | **Weakest area.** Used LA in quantum mechanics but without true foundational understanding. | Most urgent rebuild |
| Statistics / Data Science | Weak but fast learner here. Gap is applied statistical thinking, not mathematical maturity. | Learn in parallel with ML |
| Algorithms / CS | Moderate. Weak on formal complexity analysis. | Build deliberately via CLRS |
| ML Implementation | Competent with libraries. Insecure from scratch — insecurity is a reps problem, not a knowledge problem. | Solve by doing |

---

---

# THE WORK

---

## TRACK A — Foundations

*Do not consider a milestone complete until it has passed through the full learning loop.*
*Completion is not finishing a chapter. See Exit Criteria.*

---

### 🎲 Probability — ACTIVE (your strongest area, currently consolidating)

**Current task:** Consolidate existing notebooks into one definitive, teach-forward notebook.
Write every entry as if it is the last clear explanation of that concept that will ever exist.

- [ ] Consolidate all existing probability notebooks into one master notebook
- [ ] **Ross** — Foundations
  - Combinatorics and counting *(connects directly to the research vision — pay attention here)*
  - Axioms of probability
  - Conditional probability and independence
  - Bayes' theorem — derive from axioms, implement, reflect on what it means for learning

- [ ] **Ross** — Random Variables
  - Discrete and continuous distributions
  - Expectation, variance, and moments
  - Key distributions: Bernoulli, Binomial, Poisson, Gaussian, Exponential

- [ ] **Ross** — Limit Theorems
  - Law of Large Numbers
  - Central Limit Theorem — understand what "convergence" actually guarantees

- [ ] **Ash** — Structural depth
  - Measure-theoretic foundations (build intuition before formalism)
  - Deeper treatment of expectation and distributions

- [ ] **Casella & Berger** or **Rice** — Statistical Inference
  - Maximum Likelihood Estimation — derive it, implement it
  - Bayesian inference — the bridge from probability to learning
  - Hypothesis testing, confidence intervals, sufficient statistics

- [ ] **Jaynes** — Probability as Logic *(after Ross and Ash feel natural)*
  - Probability as a framework for reasoning in any system
  - Directly relevant to alignment and universal communication

- [ ] **Feller** — Deep Structure *(last — save until Jaynes feels comfortable)*

---

### 📐 Linear Algebra — URGENT (your weakest area)

*You used LA in quantum mechanics without true foundations. That gap closes here.*
*Do not rush this. Do not skip geometric intuition to get to abstraction faster.*

- [ ] **Anton** — Rebuild from the ground up
  - Vectors and vector spaces — what they actually are geometrically
  - Matrix operations — implement every one from scratch
  - Determinants — derive, don't memorize
  - Eigenvalues and eigenvectors — derive, implement, visualize
  - What does a linear transformation *do* to space? Answer this geometrically.

- [ ] **Hoffman & Kunze** — Abstract foundations
  - Abstract vector spaces beyond Rn
  - Linear maps and dual spaces
  - Inner product spaces
  - This is what lets you read ML papers as a native speaker, not a translator

---

### 📉 Vector Calculus — Actively studied, not just referenced

*Gradients, Jacobians, and Hessians are the engine of optimization.
You cannot design new optimizers without deep comfort here.*

- [ ] **Baxandall & Liebeck**
  - Partial derivatives and gradients
  - Multivariate chain rule — this IS backpropagation, mathematically
  - Jacobians and Hessians
  - Optimization conditions (when is a critical point a minimum?)

---

### 📊 Statistics for Data Science — Weak but fast learner; learn in parallel with ML

*The gap here is applied statistical thinking, not mathematical maturity.
Pull from Casella & Berger and Rice; supplement with hands-on data work.*

- [ ] Experimental design and sampling
- [ ] Regression — derive OLS from first principles
- [ ] Model evaluation: bias-variance tradeoff, cross-validation
- [ ] Bayesian vs. frequentist framing — understand both, know when each applies
- [ ] Apply to a real dataset (connect to Track B projects)

---

### 🔢 Algorithms — CLRS

*Efficiency is an algorithmic complexity problem before it is an ML problem.*

- [ ] Asymptotic analysis — make O(n), O(n log n), O(n^2) feel visceral, not symbolic
- [ ] Sorting and searching
- [ ] Dynamic programming
- [ ] Graph algorithms
- [ ] NP-completeness — why some problems are hard by nature, not by implementation

---

## TRACK B — Implementation

*Always have one active project.*
*From scratch when it reveals structure; libraries when structure is already understood.*

- [ ] Matrix operations from scratch (no NumPy) — do this early, while doing Anton
- [ ] Sample from a Gaussian distribution from scratch
- [ ] Gradient descent from scratch
- [ ] Logistic regression: derive MLE, implement, analyze time complexity
- [ ] Backpropagation from scratch
- [ ] Single hidden-layer neural net from scratch
- [ ] Demonstrate catastrophic forgetting on two sequential tasks — make it visual
- [ ] Implement Elastic Weight Consolidation (EWC) as a baseline mitigation
- [ ] A project operationalizing the combinatorics hypothesis *(define as vision sharpens)*

---

## TRACK C — Research Taste

*Weekly. This is what turns study into research.*

- [ ] Week 1: Read 1 abstract in continual learning. Write 1 question it opens.
- [ ] Week 2: Identify the objective function of any model you encounter. Write it out formally.
- [ ] Week 3: Read 1 abstract in information theory or combinatorics. Write 1 "what if X?" question.
- [ ] *(continue indefinitely — rotate topics, keep writing questions, keep the journal)*

---

---

## Exit Criteria

*A milestone is complete when you can do the following without looking anything up.*

**Probability (Ross):** Derive Bayes' theorem from the axioms. Sample from three different distributions in code. Explain what the Central Limit Theorem actually guarantees and where it breaks down.

**Linear Algebra (Anton):** Implement matrix multiplication from scratch. Compute eigenvalues and eigenvectors of a 2x2 matrix by hand. Explain geometrically what any linear transformation does to space.

**Linear Algebra (H&K):** Define a vector space from the axioms. Explain what a dual space is and why it matters. Read a paper that uses kernel methods and understand the function space it implies.

**Vector Calculus:** Compute the gradient of a loss function by hand. Apply the multivariate chain rule to a two-layer network. Explain what the Hessian tells you about an optimization landscape.

**Algorithms:** Analyze the time complexity of something you wrote. Explain why a problem is NP-hard. Implement a dynamic programming solution without looking at the pattern first.

**Full Research Gate:** Read an arXiv abstract in your area and write a one-paragraph research question it opens up — one that isn't already in the paper.

---

## Portfolio Rule

Every public GitHub project must answer:
> *"What does this person understand that most ML engineers don't?"*

Projects orbit the research vision. Generic tutorials are not portfolio items.

---

## Ground Rules

- The goal is not to finish books. The goal is to ask questions nobody has asked.
- Linear algebra is the most urgent rebuild. Do not let probability strength become an excuse to avoid it.
- Insecurity in implementation is a reps problem. Solve it by doing, not by waiting.
- Write every note as if it will outlast you. That standard produces real understanding.
- Stay at altitude on the vision. Land on the ground for the work.
- Journal the questions. Especially the ones that seem too big.

---

*The tools exist. The vision exists. The work is making them meet.*
