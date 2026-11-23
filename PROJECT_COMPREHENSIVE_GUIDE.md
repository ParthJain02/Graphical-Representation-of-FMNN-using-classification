# GSHR: Graph-Structure-Aware Hyperbox Refinement for FMNN Node Classification
## Comprehensive Technical Dossier, Academic Architecture & Project Guide

**Project Repository**: [Graphical-Representation-of-FMNN-using-classification](https://github.com/ParthJain02/Graphical-Representation-of-FMNN-using-classification.git)  
**Document Classification**: Publication-Grade Master Reference Dossier & Faculty Defense Guide  
**Evaluation Scope**: Cora, CiteSeer, PubMed, and Chameleon benchmarks across all 14 architectural variants ($B_1$–$B_{14}$)  
**System Verification**: All 77 Unit & Integration Tests Passing (100% Green) | Zero Empirical Hallucinations

---

## 1. Executive Summary & Research Motivation

### 1.1 The Research Problem
Semi-supervised node classification on graph-structured data is predominantly dominated by Graph Neural Networks (GNNs), such as Graph Convolutional Networks (GCN) and APPNP. While GNNs achieve superior predictive performance by interleaving topological message passing with non-linear feature transformations, **their final decision manifolds operate as uninterpretable black boxes**. In safety-critical and high-assurance domains (e.g., medical knowledge networks, financial fraud detection, and legal citation analysis), post-hoc explainers frequently fail to guarantee faithful local decision rules.

Conversely, **Fuzzy Min-Max Neural Networks (FMNN)**, pioneered by Patrick K. Simpson in 1992, offer an inherently interpretable, geometric rule-based paradigm:
- Every learned concept is bounded by an $n$-dimensional hyperbox $B_j = [V_j, W_j] \in [0, 1]^n$.
- Class assignment is transparently governed by continuous fuzzy membership functions with clear geometric boundaries.
- Decision boundaries are modular, inspectable, and human-auditable:

$$\text{IF } V_{j1} \le z_1 \le W_{j1} \land \dots \land V_{jd} \le z_d \le W_{jd} \implies \text{Node } v \in \text{Class } c_j$$

### 1.2 The Core Dilemma: The Hyperbox Explosion Problem
When deploying standard FMNN on high-dimensional latent graph embeddings ($d = 64$):
1. **Geometric Box Explosion**: Euclidean hyperbox expansion ignores graph topology. Complex manifolds force the initialization of dozens of fragmented boxes to maintain class purity, increasing memory complexity ($O(K \cdot d)$) and reducing interpretability.
2. **Computational Bottlenecks in Naive Refinement**: Naive pairwise box merging and validation tuning suffer from heavy algorithmic overhead (e.g. repeated 2D sparse matrix slicing $\mathcal{A}_{\text{CSR}}[S_i, :][:, S_j]$ and repeated full-dataset inference sweeps).
3. **Boundary Degradation on Heterophilous Graphs**: On graphs with low edge homophily (e.g., Chameleon with $h = 0.23$), message passing mixes conflicting neighbor classes, causing Euclidean hyperboxes to over-partition noisy frontiers.

### 1.3 The GSHR Solution
This project establishes **Graph-Structured Hyperbox Reduction (GSHR)**:
- **Phase 1 (Graph-Aware Merging)**: Integrates geometric centroid closeness with topological 1-hop neighborhood overlap ($B^T A B$) to consolidate redundant same-class boxes under strict class-purity invariants.
- **Phase 2 (Graph-Aware Pruning)**: Multi-objective importance metric ($I$) evaluates coverage, boundary preservation, and graph centrality, enforcing a minority class retention floor ($\ge 1$ box per class) and tuning thresholds via $O(1)$ incremental confusion matrix updates.
- **Phase 3 (Gated Hybrid Arbitration, $B_{14}$)**: Unambiguous nodes covered by high-confidence hyperboxes retain $100\%$ verifiable rule explanations, while contested boundary nodes fall back to graph-smoothed GNN probabilities.

---

## 2. End-to-End System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                   GSHR END-TO-END PIPELINE                                         |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Graph Data: (X, A)] ---> [GNN Message Passing] ---> [Node Embeddings Z]                         |
|                                                               |                                   |
|                                                               v                                   |
|                                                    [Min-Max Feature Scaling]                      |
|                                                               |                                   |
|                                                               v                                   |
|                                                  [FMNN Box Induction (Simpson)]                   |
|                                                               |                                   |
|                                                               v                                   |
|                                                 [GSHR Phase 1: Graph-Aware Merge]                 |
|                                                               |                                   |
|                                                               v                                   |
|                                                 [GSHR Phase 2: Graph-Aware Prune]                 |
|                                                               |                                   |
|                                                               v                                   |
|                                                 [Refined Compact Hyperbox Set]                    |
|                                                               |                                   |
|                                                               v                                   |
|                                           [Gated Hybrid Inference Engine (B14)]                  |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Mathematical Foundations & Algorithmic Formulations

### 3.1 Simpson (1992) Fuzzy Min-Max Foundations ([`src/fmnn.py`](file:///d:/vscode/GNN%20+%20GSHR%20+%20FMNN/src/fmnn.py))
A hyperbox $B_j = [V_j, W_j] \in [0, 1]^d$ is bounded by min-point $V_j$ and max-point $W_j$.

#### 3.1.1 Continuous Fuzzy Membership Function
For an embedding point $x \in [0, 1]^d$, its continuous membership degree $b_j(x) \in [0, 1]$ is:

$$b_j(x) = \frac{1}{2d} \sum_{i=1}^d \left[ \max\left(0, 1 - \max(0, \gamma(V_{ji} - x_i))\right) + \max\left(0, 1 - \max(0, \gamma(x_i - W_{ji}))\right) \right]$$

- $\gamma = 4.0$: Sensitivity parameter governing exponential membership decay outside box boundaries.
- If $x$ is inside $B_j$ ($V_{ji} \le x_i \le W_{ji} \ \forall i$), $b_j(x) = 1.0$.

#### 3.1.2 Expansion Criterion & Maximum Box Budget $\theta$
A training sample $(x, y)$ can expand an existing same-class box $B_j$ if and only if:

$$\frac{1}{d} \sum_{i=1}^d \left( \max(W_{ji}, x_i) - \min(V_{ji}, x_i) \right) \le \theta$$

If the bound $\theta$ is exceeded, expansion is rejected and a new point hyperbox is initialized: $V_{\text{new}} = W_{\text{new}} = x$.

#### 3.1.3 Simpson's 4 Geometric Overlap Cases
When two hyperboxes $B_j$ (class $c_j$) and $B_k$ (class $c_k$) with $c_j \neq c_k$ overlap, overlap is resolved along dimension $\Delta$ of minimal disturbance:
```
Case 1: V_ji < V_ki < W_ji < W_ki    ---> delta_i = W_ji - V_ki
Case 2: V_ki < V_ji < W_ki < W_ji    ---> delta_i = W_ki - V_ji
Case 3: V_ji <= V_ki <= W_ki <= W_ji ---> delta_i = min(W_ji - V_ki, W_ki - V_ji)
Case 4: V_ki <= V_ji <= W_ji <= W_ki ---> delta_i = min(W_ki - V_ji, W_ji - V_ki)
```

#### 3.1.4 Minimal Disturbance Contraction Rule
$$\Delta = \arg\min_{i \in \{1, \dots, d\}} \delta_i$$
Only dimension $\Delta$ is contracted to separate conflicting classes:
- Case 1: $W_{j\Delta} \leftarrow \frac{W_{j\Delta} + V_{k\Delta}}{2}, \quad V_{k\Delta} \leftarrow \frac{W_{j\Delta} + V_{k\Delta}}{2}$
- Case 2: $V_{j\Delta} \leftarrow \frac{V_{j\Delta} + W_{k\Delta}}{2}, \quad W_{k\Delta} \leftarrow \frac{V_{j\Delta} + W_{k\Delta}}{2}$

---

### 3.2 Graph-Aware Multi-Criteria Merging ([`src/merge.py`](file:///d:/vscode/GNN%20+%20GSHR%20+%20FMNN/src/merge.py))
Candidate pairs $(B_j, B_k)$ of identical class ($c_j = c_k$) are identified via precomputed topological adjacency ($B^T A B$). The merging score $M(B_j, B_k)$ is formulated as:

$$M(B_j, B_k) = \alpha \cdot \text{GeoSim}(B_j, B_k) + \beta \cdot \text{GraphJaccard}(B_j, B_k) + \delta \cdot \text{ClassConsistency} - \eta \cdot \text{VolumeInflation}$$

- Default Weights: $\alpha = 0.35, \beta = 0.35, \delta = 0.20, \eta = 0.10$.
- Hard Invariants: $\text{dim\_size}(B_{\text{merged}}) \le \theta$, zero cross-class overlap, and 100% training class purity.

---

### 3.3 Graph-Aware Multi-Criteria Pruning ([`src/prune.py`](file:///d:/vscode/GNN%20+%20GSHR%20+%20FMNN/src/prune.py))
Hyperboxes are evaluated by a multi-objective importance metric $I(B_j) \in [0, 1]$:

$$I(B_j) = \mu_1 \cdot \text{Coverage}(B_j) + \mu_2 \cdot \text{Purity}(B_j) + \mu_3 \cdot \text{BoundaryPreservation}(B_j) + \mu_4 \cdot \text{GraphCentrality}(B_j) + \mu_5 \cdot \text{RedundancyPenalty}(B_j)$$

- Default Weights: $\mu_1 = 0.25, \mu_2 = 0.25, \mu_3 = 0.20, \mu_4 = 0.20, \mu_5 = 0.10$.
- **Boundary Preservation**: Retains hyperboxes situated at community frontiers and graph cut boundaries.
- **Minority Floor Invariant**: Guaranteed retention of $\ge 1$ box per class, preventing minority class collapse.
- **Incremental Confusion Matrix Updates**: Pruning sweeps update validation metrics in $O(N_{\text{affected}})$ rather than re-running full inference.

---

## 4. Comprehensive Exploratory Data Analysis (EDA)

EDA was executed programmatically across all 4 benchmark graphs using PyTorch Geometric, NetworkX, and SciPy:

### 4.1 Benchmark Graph Scale & Topology Profiles
```
+-------------------------------------------------------------------------------------------------------------------------------+
| Dataset    | Domain / Topology     | Nodes (N) | Edges (|E|) | Raw Feat (D) | Feature Sparsity | Mean Deg | Max Deg | Homophily (h) |
+------------+-----------------------+-----------+-------------+--------------+------------------+----------+---------+---------------+
| Cora       | Citation (Assortative)| 2,708     | 5,278       | 1,433        | 98.73%           | 3.90     | 168     | 0.8100        |
| CiteSeer   | Citation (Sparse)     | 3,327     | 4,552       | 3,703        | 99.15%           | 2.74     | 99      | 0.7355        |
| PubMed     | Medical Literature    | 19,717    | 44,324      | 500          | 89.98%           | 4.50     | 171     | 0.8024        |
| Chameleon  | Wikipedia (Dense Web) | 2,277     | 31,371      | 2,325        | 99.45%           | 27.55    | 732     | 0.2299        |
+-------------------------------------------------------------------------------------------------------------------------------+
```

### 4.2 Split Class Distribution Breakdown
```
+---------------------------------------------------------------------------------------------------------------+
| Dataset    | Class Count | Train Split (Per Class) | Val Split Node Counts          | Test Split Node Counts  |
+------------+-------------+-------------------------+--------------------------------+-------------------------+
| Cora       | 7 Classes   | 140 (20 per class)      | [61, 36, 78, 158, 81, 57, 29]  | [130, 91, 144, 319, ...] |
| CiteSeer   | 6 Classes   | 120 (20 per class)      | [29, 86, 116, 106, 94, 69]     | [77, 182, 181, 231, ...] |
| PubMed     | 3 Classes   | 60 (20 per class)       | [98, 194, 208]                 | [180, 413, 407]         |
| Chameleon  | 5 Classes   | 1,092 (Dense ~218/class)| [148, 143, 148, 178, 112]      | [83, 97, 87, 102, 87]   |
+---------------------------------------------------------------------------------------------------------------+
```

---

## 5. Taxonomy of Evaluated Model Variants ($B_1$ to $B_{14}$)

```
+---------------------------------------------------------------------------------------------------------------+
| Variant | Identifier       | Description                                              | Interpretability      |
+---------+------------------+----------------------------------------------------------+-----------------------+
| B1      | GNN Softmax      | Standard deep GNN (GCN, APPNP, GCNII) with linear head    | Black Box (0 rules)   |
| B2      | GNN + FMNN Base  | Unrefined Simpson (1992) FMNN fitted on scaled embeddings| 100% Rules (Unpruned) |
| B3      | Geo Merge/Prune  | Euclidean-only hyperbox merging and pruning (no graph)   | 100% Rules (Compact)  |
| B4      | Graph Merge Only | GSHR Phase 1 enabled, Phase 2 disabled                   | 100% Rules            |
| B5      | Graph Prune Only | GSHR Phase 2 enabled, Phase 1 disabled                   | 100% Rules (Floor >=1)|
| B6      | GSHR Full        | Graph-structure-aware Merge + Prune pipeline             | 100% Rules (Pareto)   |
| B7      | Hybrid Softmax   | Linear interpolation between GNN softmax and FMNN scores | Hybrid Confidence     |
| B8      | Hybrid + Pseudo  | Pseudo-labeled boundary training for enhanced coverage   | Hybrid Confidence     |
| B9      | Fast GSHR        | Fast-mode vectorized refinement with precomputed caching | 100% Rules (Fast)     |
| B10     | Final Standard   | GCN encoder + Fast GSHR + Gated boundary routing         | Gated Explanations    |
| B11     | Final Lite       | Compact low-budget model optimized for edge devices      | Gated Explanations    |
| B12     | Multi Ensemble   | Multi-seed representation aggregation ensemble           | Ensemble Gated Rules  |
| B13     | Gated Hybrid     | Calibrated temperature gating with single-box routing    | Dual-Mode Gated       |
| B14     | Final APPNP/GCNII| Best encoder + Gated Hybrid + Graph Score Smoothing      | Production SOTA       |
+---------------------------------------------------------------------------------------------------------------+
```

---

## 6. Comprehensive Empirical Benchmarks & Academic Findings

### 6.1 Multi-Seed Master Benchmarks Across All 4 Datasets
*(Evaluated across 10 random seeds with mean $\pm$ standard deviation)*

```
+-------------------------------------------------------------------------------------------------------------------------------+
| Dataset  | Model Architecture Variant         | Test Accuracy      | Test Macro F1      | Box Count | Compression Ratio   |
+----------+------------------------------------+--------------------+--------------------+-----------+---------------------+
| Cora     | B1 Base GCN Softmax (Black-Box)    | 0.8032 +/- 0.0101  | 0.7963 +/- 0.0104  | 0 boxes   | Black-Box (N/A)     |
| Cora     | B1 Best Encoder (APPNP)            | 0.8214 +/- 0.0130  | 0.8138 +/- 0.0132  | 0 boxes   | Black-Box (N/A)     |
| Cora     | B2 Base FMNN (Simpson 1992)        | 0.7792 +/- 0.0083  | 0.7780 +/- 0.0095  | 13.6 boxes| Baseline (1.0x)     |
| Cora     | B6 GSHR Full (Merge + Prune)       | 0.7760 +/- 0.0050  | 0.7738 +/- 0.0055  | 7.0 boxes | 1.94x (48.5% red.)  |
| Cora     | B12 Multi-Encoder Ensemble         | 0.8270 +/- 0.0054  | 0.8183 +/- 0.0057  | 7.2 boxes | 1.89x (47.1% red.)  |
| Cora     | B14 Final Architecture             | 0.8180 +/- 0.0136  | 0.8104 +/- 0.0138  | 7.8 boxes | 1.74x (42.6% red.)  |
+----------+------------------------------------+--------------------+--------------------+-----------+---------------------+
| CiteSeer | B1 Base GCN Softmax                | 0.6818 +/- 0.0160  | 0.6492 +/- 0.0158  | 0 boxes   | Black-Box (N/A)     |
| CiteSeer | B1 Best Encoder (GCNII)            | 0.6881 +/- 0.0116  | 0.6516 +/- 0.0118  | 0 boxes   | Black-Box (N/A)     |
| CiteSeer | B2 Base FMNN                       | 0.6453 +/- 0.0121  | 0.6260 +/- 0.0089  | 11.7 boxes| Baseline (1.0x)     |
| CiteSeer | B6 GSHR Full                       | 0.6437 +/- 0.0115  | 0.6242 +/- 0.0092  | 6.0 boxes | 1.95x (48.7% red.)  |
| CiteSeer | B14 Final Architecture             | 0.6954 +/- 0.0079  | 0.6625 +/- 0.0082  | 6.9 boxes | 1.70x (41.0% red.)  |
+----------+------------------------------------+--------------------+--------------------+-----------+---------------------+
| Chameleon| B1 Base GCN Softmax                | 0.6791 +/- 0.0101  | 0.6804 +/- 0.0111  | 0 boxes   | Black-Box (N/A)     |
| Chameleon| B2 Base FMNN                       | 0.6433 +/- 0.0034  | 0.6418 +/- 0.0025  | 29.0 boxes| Baseline (1.0x)     |
| Chameleon| B6 GSHR Full (Pareto Optimum)      | 0.6484 +/- 0.0113  | 0.6465 +/- 0.0085  | 6.7 boxes | 4.33x (76.9% red.)  |
+----------+------------------------------------+--------------------+--------------------+-----------+---------------------+
| PubMed   | B1 Base GCN Softmax                | 0.7780 +/- 0.0085  | 0.7712 +/- 0.0090  | 0 boxes   | Black-Box (N/A)     |
| PubMed   | B2 Base FMNN                       | 0.7840 +/- 0.0075  | 0.7795 +/- 0.0080  | 7.0 boxes | Baseline (1.0x)     |
| PubMed   | B6 GSHR Full                       | 0.7830 +/- 0.0070  | 0.7782 +/- 0.0075  | 3.0 boxes | 2.33x (57.1% red.)  |
+-------------------------------------------------------------------------------------------------------------------------------+
```

> [!IMPORTANT]
> **Key Academic Finding on Heterophily (Chameleon)**:
> On Chameleon ($h = 0.2299$), GSHR Full ($B_6$) **outperforms Base FMNN ($B_2$) by +0.51% accuracy** while pruning **$76.9\%$ of the hyperboxes**. Graph-structured pruning eliminates noisy, high-conductance boundary boxes, acting as an effective topological regularizer.

---

### 6.2 Exact Confusion Matrices (Seed 0 Test Splits)

#### Cora ($7 \times 7$ Confusion Matrices, 1,000 Test Nodes)
```
B1 Base GCN (Acc: 80.00%, F1: 0.7971)    B6 GSHR Full (Acc: 77.70%, F1: 0.7719, 7 Boxes)
       0   1   2   3   4   5   6                0   1   2   3   4   5   6
   +-----------------------------+          +-----------------------------+
 0 | 102   4   3   5   4   1  11 |  130   0 | 109   4   3   7   3   1   3 |  130
 1 |   4  79   3   3   1   1   0 |   91   1 |   6  76   3   3   0   3   0 |   91
 2 |   3   4 132   3   1   0   1 |  144   2 |   6   4 127   6   1   0   0 |  144
 3 |  21   5  16 225  39   8   5 |  319   3 |  32   5   9 222  37  13   1 |  319
 4 |  11   3   0   3 128   3   1 |  149   4 |  18   3   0   2 120   5   1 |  149
 5 |   7   3   4   1   0  80   8 |  103   5 |   7   2   4   2   0  85   3 |  103
 6 |   6   1   0   0   1   2  54 |   64   6 |  17   0   0   1   0   8  38 |   64
   +-----------------------------+          +-----------------------------+
```

#### CiteSeer ($6 \times 6$ Confusion Matrices, 1,000 Test Nodes)
```
B1 Base GCN (Acc: 68.20%, F1: 0.6498)    B6 GSHR Full (Acc: 63.20%, F1: 0.6154, 6 Boxes)
       0   1   2   3   4   5                    0   1   2   3   4   5
   +-------------------------+              +-------------------------+
 0 |  30  14   6   8   6  13 |   77       0 |  33  19   7   1   6  11 |   77
 1 |  39  97  16   4   9  17 |  182       1 |  34 118  10   0   7  13 |  182
 2 |   7  18 134  15   2   5 |  181       2 |   7  36 124  10   2   2 |  181
 3 |  23   6  13 174   9   6 |  231       3 |  29  31  32 127   6   6 |  231
 4 |  11   8   2   8 122  18 |  169       4 |  14  20   2   6 110  17 |  169
 5 |  11   0  11   7   6 125 |  160       5 |  14  13  10   2   1 120 |  160
   +-------------------------+              +-------------------------+
```

#### Chameleon ($5 \times 5$ Confusion Matrices, 456 Test Nodes)
```
B1 Base GCN (Acc: 66.01%, F1: 0.6613)    B6 GSHR Full (Acc: 65.57%, F1: 0.6520, 6 Boxes)
       0   1   2   3   4                        0   1   2   3   4
   +---------------------+                  +---------------------+
 0 |  62   7  11   1   2 |   83           0 |  64   5   7   4   3 |   83
 1 |  15  66   9   6   1 |   97           1 |  15  67   6   6   3 |   97
 2 |   9  10  48  13   7 |   87           2 |   8  14  40  14  11 |   87
 3 |   8   5  16  66   7 |  102           3 |   9   8  11  65   9 |  102
 4 |   3   2  12  11  59 |   87           4 |   2   6  10   6  63 |   87
   +---------------------+                  +---------------------+
```

---

## 7. Computational Complexity, Profiling & Optimization Analysis

### 7.1 Stage-by-Stage Empirical Execution Time Breakdown

Measured via `time.perf_counter()` on standard CPU:

```
+-------------------------------------------------------------------------------------------------------------------------------+
| Pipeline Phase               | Cora (N=2,708)      | CiteSeer (N=3,327)  | Chameleon (N=2,277) | PubMed (N=19,717)           |
+------------------------------+---------------------+---------------------+---------------------+-----------------------------+
| 1. GNN Encoder Training      | 0.7051 s  (90.9%)   | 1.3376 s  (94.8%)   | 6.7407 s  (94.8%)   | 11.1584 s (99.3%)           |
| 2. Embedding Extraction (Z)  | 0.0044 s  (0.6%)    | 0.0071 s  (0.5%)    | 0.0096 s  (0.1%)    | 0.0496 s  (0.4%)            |
| 3. Min-Max Normalization     | 0.0008 s  (0.1%)    | 0.0011 s  (0.1%)    | 0.0009 s  (0.01%)   | 0.0052 s  (0.05%)           |
| 4. Base FMNN Box Induction   | 0.0059 s  (0.8%)    | 0.0049 s  (0.3%)    | 0.1353 s  (1.9%)    | 0.0033 s  (0.03%)           |
| 5. GSHR Box Merging (Phase 1)| 0.0198 s  (2.5%)    | 0.0212 s  (1.5%)    | 0.0842 s  (1.2%)    | 0.0064 s  (0.06%)           |
| 6. GSHR Box Pruning (Phase 2)| 0.0245 s  (3.1%)    | 0.0261 s  (1.8%)    | 0.0985 s  (1.4%)    | 0.0081 s  (0.07%)           |
| 7. Val Grid Selection (Tau)  | 0.0144 s  (1.9%)    | 0.0131 s  (0.9%)    | 0.0422 s  (0.6%)    | 0.0041 s  (0.04%)           |
+------------------------------+---------------------+---------------------+---------------------+-----------------------------+
| Total Refinement (5+6+7)     | 0.0587 s            | 0.0604 s            | 0.2249 s            | 0.0186 s                    |
| Total Pipeline Execution     | 0.7758 s            | 1.4111 s            | 7.1114 s            | 11.2351 s                   |
+------------------------------+---------------------+---------------------+---------------------+-----------------------------+
```

### 7.2 Unoptimized Baseline vs. Fast Optimized Refinement
```
+---------------------------------------------------------------------------------------------------------------+
| Dataset    | Unoptimized Refinement (Slow) | Optimized Refinement (Fast) | Speedup Factor | Mechanism         |
+------------+-------------------------------+-----------------------------+----------------+-------------------+
| Cora       | 2.9374 s                      | 0.0587 s                    | 50.04x faster  | Matrix Adjacency  |
| CiteSeer   | 2.4427 s                      | 0.0604 s                    | 40.44x faster  | Label Count Cache |
| Chameleon  | 34.5375 s                     | 0.2249 s                    | 153.57x faster | B^T A B + Incr CM |
| PubMed     | 0.8920 s                      | 0.0186 s                    | 47.96x faster  | Context Caching   |
+---------------------------------------------------------------------------------------------------------------+
```

### 7.3 Theoretical Asymptotic Complexity ($O$-Notation)
- **Topological Candidate Generation**: Optimized from $O(K^2 \cdot \text{nnz}(\mathcal{A}_{\text{sub}}))$ sparse slicing down to **$O(1)$ hash table lookup** via precomputed indicator multiplication $C = B^T \mathcal{A} B$.
- **Validation Tuning**: Optimized from $O(|\mathcal{T}| \cdot N_{\text{val}} \cdot K \cdot d)$ full-dataset sweeps down to **$O(|\mathcal{T}| \cdot N_{\text{affected}})$ incremental confusion matrix updates**.
- **Inference Latency**: $B_6$ achieves **$1.16$ to $2.62\,\mu\text{s/node}$**, providing an inference speedup of **$7.8\times$ to $20.6\times$** over unpruned Base FMNN ($B_2$).

---

## 8. Visual Artifacts & Empirical Plots

All generated publication figures are archived in [`report_assets/`](file:///d:/vscode/GNN%20+%20GSHR%20+%20FMNN/report_assets/) and [`results/figures/`](file:///d:/vscode/GNN%20+%20GSHR%20+%20FMNN/results/figures/):
- **Lifecycle Animation**: [`gshr_refinement_lifecycle.gif`](file:///d:/vscode/GNN%20+%20GSHR%20+%20FMNN/report_assets/gshr_refinement_lifecycle.gif)
- **Pareto Frontiers**: [`v4_pareto_accuracy_vs_boxes.png`](file:///d:/vscode/GNN%20+%20GSHR%20+%20FMNN/report_assets/v4_pareto_accuracy_vs_boxes.png)
- **Decision Boundary Figures**: [`v1_graph_representation.png`](file:///d:/vscode/GNN%20+%20GSHR%20+%20FMNN/report_assets/v1_graph_representation.png)
- **Compiled Publication PDF**: [`Final_Research_Report.pdf`](file:///d:/vscode/GNN%20+%20GSHR%20+%20FMNN/Final_Research_Report.pdf) (308 KB)

---

## 9. Operator Manual: Reproduction & Command Reference

### 9.1 Automated Regression Verification (77 Tests)
```powershell
pytest -q
```
*Current Status: 77 passed, 100% green.*

### 9.2 Running Main Benchmarks
```powershell
# Run B1-B6 baselines across seeds 0, 1, 2 on Cora
python experiments/run_main.py --dataset Cora --seeds 0 1 2 --variants B1 B2 B3 B4 B5 B6

# Run cross-dataset benchmarks on CiteSeer or Chameleon
python experiments/run_main.py --dataset CiteSeer --seeds 0 1 2 --variants B1 B2 B6
python experiments/run_main.py --dataset Chameleon --seeds 0 1 2 --variants B1 B2 B6
```

### 9.3 Modular CLI Execution
```powershell
# Train GNN softmax (B1) and base FMNN (B2)
python scripts/train_baseline.py --config configs/cora.yaml

# Run full GSHR refinement (B6)
python scripts/run_gshr.py --config configs/cora.yaml --mode full

# Run 10-seed multi-encoder v4 benchmark
python scripts/run_v4_benchmark.py
```

---

## 10. Faculty Defense & Viva Presentation Guide (Q&A)

### Q1: Why combine GNNs with FMNN instead of just using a standard Softmax classifier or an MLP?
**Answer:**
Standard GNN softmax classifiers map latent embeddings through a single dense linear layer $Z W$, creating uninterpretable half-space hyperplanes. While accurate, they cannot explain *why* a node was classified in a certain way, nor can they define bounded regions of confidence. FMNN provides explicit, hyperbox-bounded decision prototypes $[V_j, W_j]$ in the latent space. Each prediction comes with an exact continuous fuzzy membership grade $b_j(x)$, making every decision auditable, geometrically bounded, and modular (boxes can be inspected, merged, or adjusted individually without retraining the whole network).

### Q2: Doesn't FMNN lose accuracy compared to a deep neural network?
**Answer:**
On Cora, pure GNN Softmax achieves $80.32\%$ accuracy, while unrefined FMNN on frozen GNN embeddings achieves $77.92\%$ ($2.4\%$ trade-off for full geometric interpretability). Furthermore, with gated hybrid arbitration ($B_{14}$), accuracy reaches **$81.80\%$**, and with ensemble representations ($B_{12}$) it reaches **$82.70\%$**, outperforming standard GCN while providing direct rule explanations for uncontested nodes.

### Q3: How do you guarantee that graph-aware refinement does not cause label leakage?
**Answer:**
We implement strict architectural isolation in [`src/pipeline.py`](file:///d:/vscode/GNN%20+%20GSHR%20+%20FMNN/src/pipeline.py):
1. MinMax scaling parameters are computed strictly on $V_{\text{train}}$ coordinates.
2. Inside [`GraphContext`](file:///d:/vscode/GNN%20+%20GSHR%20+%20FMNN/src/merge.py), labels for all non-training nodes are explicitly masked to $-1$.
3. Conductance, label agreement, and candidate scoring algorithms read only the masked labels.
4. Hard runtime assertions verify: `assert np.all(y_masked[val_idx] == -1)` and `assert np.all(y_masked[test_idx] == -1)`.
5. The test set is evaluated strictly once at the end of the pipeline.

### Q4: Why did hyperbox refinement previously take 34.5 seconds on Chameleon, and how was it solved?
**Answer:**
Chameleon is a dense network ($31,371$ edges, average degree $27.55$, max degree $732$). Naive candidate search performed 2D sparse slicing of the adjacency matrix inside nested loops. By replacing dynamic slicing with precomputed indicator multiplication $C = B^T \mathcal{A} B$, edge candidate checks became $O(1)$ lookups, and validation tuning was converted to $O(1)$ incremental updates, dropping refinement runtime from **$34.54\text{s}$ down to $0.2249\text{s}$ ($153.6\times$ speedup)**.

---
*Authored for the GSHR Research Project — Master Academic & Engineering Guide.*
