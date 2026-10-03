# GCN Mathematical Formulation

This document explains the complete mathematical derivation of the Vanilla
Graph Convolutional Network (GCN) as described in:

> William L. Hamilton — *Graph Representation Learning* (2020), Chapters 5 & 7.
> Kipf & Welling — *Semi-Supervised Classification with GCNs* (ICLR 2017).

---

## 1. Graph Definition

A graph is a pair **G = (V, E)** where:

- **V** is the set of nodes (vertices), |V| = N
- **E ⊆ V × V** is the set of edges

---

## 2. Notation Summary

| Symbol | Shape | Description |
|--------|-------|-------------|
| **A** | N × N | Raw binary adjacency matrix |
| **Â** | N × N | Adjacency with self-loops: Â = A + I |
| **D̂** | N × N | Degree matrix of Â (diagonal) |
| **D̂^(-1/2)** | N × N | Inverse square-root degree matrix |
| **Ã** | N × N | Symmetrically normalized adjacency |
| **X** | N × F | Node feature matrix (raw input) |
| **H^(0)** | N × F | Initial embeddings = X |
| **H^(k)** | N × d_k | Node embeddings at layer k |
| **W^(k)** | d_{k-1} × d_k | Learnable weight matrix at layer k |
| **σ** | — | Non-linear activation function (e.g. ReLU) |

---

## 3. Adjacency Matrix A

The adjacency matrix **A ∈ {0,1}^(N×N)** encodes graph structure:

```
A_ij = 1   if edge (i,j) ∈ E
A_ij = 0   otherwise
```

For undirected graphs, A is symmetric: **A = A^T**.

---

## 4. Adding Self-Loops: Â

Pure message passing without self-loops would discard a node's own
features when aggregating. We fix this by adding a self-loop to every node:

```
Â = A + I_N
```

where **I_N** is the N×N identity matrix. Now each node aggregates from
its neighbors **and** itself.

---

## 5. Degree Matrix: D̂

The degree matrix **D̂** is a diagonal matrix where each diagonal entry
is the row-sum of **Â**:

```
D̂_ii = Σ_j Â_ij    (degree of node i in the augmented graph)
```

---

## 6. Inverse Square-Root Degree: D̂^(-1/2)

Since D̂ is diagonal:

```
[D̂^(-1/2)]_ii = 1 / sqrt(D̂_ii)
```

This is used to normalize the adjacency so that all nodes contribute
equally regardless of degree, preventing high-degree nodes from
dominating aggregation.

---

## 7. Symmetrically Normalized Adjacency: Ã

```
Ã = D̂^(-1/2) Â D̂^(-1/2)
```

Entry-wise this equals:

```
Ã_ij = Â_ij / sqrt(D̂_ii) / sqrt(D̂_jj)
```

**Properties of Ã:**
- Symmetric: Ã = Ã^T
- All entries are in [0, 1] for binary adjacency
- Row sums ≤ 1 (sub-stochastic)
- Eigenvalues lie in [-1, 1]
- Prevents gradient exploding/vanishing across layers

**Why symmetric (not row) normalization?**  
Row normalization `D^(-1)A` is a random-walk matrix — asymmetric and
does not preserve spectral properties. Symmetric normalization preserves
the undirected graph structure and has better theoretical properties for
spectral graph convolution (Hamilton Ch. 7).

---

## 8. Node Features: X

```
X ∈ R^(N × F)
```

Row i of X is the **F-dimensional feature vector** for node i.
This is the raw input to the GCN.

---

## 9. Initial Embedding: H^(0)

```
H^(0) = X    ∈ R^(N × F)
```

The initial "embedding" is simply the raw feature matrix.

---

## 10. GCN Layer: H^(k)

**One GCN layer** applies:

```
H^(k) = σ( Ã H^(k-1) W^(k) )
```

where:
- **Ã H^(k-1)** aggregates neighbour embeddings (including self)
- **W^(k) ∈ R^(d_{k-1} × d_k)** linearly projects the aggregated embeddings
- **σ** is a non-linear activation (ReLU for hidden layers, none for output)

Step-by-step for node u at layer k:

```
h_u^(k) = σ(  W^(k)^T  Σ_{v ∈ N(u) ∪ {u}}  α_uv  h_v^(k-1)  )
```

where the normalisation coefficient is:

```
α_uv = Ã_uv = 1 / ( sqrt(D̂_uu) × sqrt(D̂_vv) )
```

---

## 11. Full K-Layer GCN

```
H^(0) = X

H^(1) = σ( Ã H^(0) W^(1) )

H^(2) = σ( Ã H^(1) W^(2) )

  ...

H^(K) = σ( Ã H^(K-1) W^(K) )    ← last layer has no activation → logits
```

For node classification with C classes:

```
logits   = H^(K)  ∈ R^(N × C)

ŷ_u  = argmax_{c=0..C-1}  logits_{u,c}
```

---

## 12. Loss Function

```
L = (1/|V_train|) Σ_{u ∈ V_train}  CrossEntropy( logits_u, y_u )
```

CrossEntropyLoss internally applies log-softmax, so the model outputs
raw logits (no softmax layer).

---

## 13. Connection to Graph Signal Processing

GCN can be understood as a first-order approximation to spectral graph
convolution (Chebyshev polynomial truncated at order 1):

```
Convolution on graph  ≈  Ã H W
```

The normalized Laplacian L̃ = I - Ã has eigenvalues in [0, 2].
The GCN filter f_W(L̃) ≈ I - L̃ = Ã is a **low-pass filter** that
smooths node features across the graph. This is exactly why stacking
too many layers causes over-smoothing (Section 14 below).

(Hamilton Ch. 7 derives this connection rigorously.)

---

## 14. Over-Smoothing

With K GCN layers, each node aggregates information from its K-hop
neighborhood. As K → ∞, all node representations converge to the same
vector — a phenomenon called **over-smoothing**.

Formally, as K → ∞:
```
H^(K) → stationary distribution (same for all nodes)
```

This is illustrated experimentally in `notebooks/03_gcn_over_smoothing.ipynb`
and `scripts/run_oversmoothing.py`.

---

## 15. Receptive Field

A K-layer GCN has a **K-hop receptive field**: each node's representation
after K layers depends on all nodes within K hops.

```
K=1  →  direct neighbors
K=2  →  neighbors of neighbors
K=3  →  3-hop neighborhood
```

This is analogous to the receptive field in CNNs.

---

## 16. Parameter Count

For a 2-layer GCN with:
- Input features: F
- Hidden dimension: H
- Number of classes: C

```
Layer 1 parameters: F × H
Layer 2 parameters: H × C
─────────────────────────
Total:              F×H + H×C
```

Example (F=4, H=16, C=3): 4×16 + 16×3 = 64 + 48 = **112 parameters**.
