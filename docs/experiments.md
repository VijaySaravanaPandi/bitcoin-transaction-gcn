# Elliptic++ Experiments

The project supports transaction and wallet-address node classification with the same vanilla GCN architecture.

## Data modes

`elliptic` loads `txs_features.csv`, `txs_classes.csv`, and `txs_edgelist.csv`.
`elliptic_actors` loads `wallets_features.csv`, `wallets_classes.csv`, and `AddrAddr_edgelist.csv`.
The AddrTx and TxAddr files represent a different relation and are not silently combined with a homogeneous graph.

Unknown class-3 nodes remain in message passing but are excluded from train, validation, and test masks.

## Feature engineering

The Elliptic++ loader can append three finite graph features:

- log-scaled in-degree
- log-scaled out-degree
- log-scaled total degree

Enable or disable this through `data.add_graph_features` in YAML. Missing numeric feature values are replaced with zero.

## Splits

The default random split divides labelled nodes according to `train_ratio`, `validation_ratio`, and `test_ratio`.

The temporal split uses `node_times`:

```yaml
data:
    split_strategy: "temporal"
    temporal_train_end: 30
    temporal_validation_end: 39
```

This produces train steps 1-30, validation steps 31-39, and test steps 40-49.

## Imbalance-aware training

When `training.class_weighted_loss` is enabled, class weights are computed from the training mask and passed to cross-entropy loss. This prevents the dominant licit class from hiding poor illicit recall.

## Metrics

Evaluation reports accuracy plus:

- illicit precision, recall, and F1
- macro-F1
- balanced accuracy
- PR-AUC
- ROC-AUC

Use PR-AUC and illicit recall as the primary fraud-detection metrics. Overall accuracy is not sufficient for this imbalanced dataset.

## Risk and clustering analysis

```bash
make analyze-risk
```

The analysis script writes:

- `outputs/analysis/risk_ranking.csv`
- `outputs/analysis/risk_ranking.png`
- `outputs/analysis/calibration_curve.png`
- `outputs/analysis/test_confusion_matrix.png`
- `outputs/analysis/embedding_clusters.csv`
- `outputs/analysis/embedding_clusters.png`
- `outputs/analysis/cluster_assignments.npy`

Risk ranking combines illicit probability and predictive entropy. Cluster summaries report node counts and the fraction of labelled nodes classified as illicit.

## Link prediction

```bash
make evaluate-links
```

This uses the final hidden embeddings and compares observed edges with sampled non-edges using ROC-AUC and average precision. It is an analysis workflow; it does not change the node-classification objective.

The link workflow also saves `outputs/figures/link_prediction.png`.

## Large-graph constraints

The Elliptic++ graph is kept as a sparse COO tensor. Do not convert it to a dense `N x N` matrix. Full pairwise cosine similarity and full-graph NetworkX visualization are also unsuitable at this scale; use sampled nodes or sampled pairs for those analyses.
