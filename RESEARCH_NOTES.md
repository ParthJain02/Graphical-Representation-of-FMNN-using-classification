# Research notes

## Scope

This project implements the research scaffold for a GCN + FMNN baseline and a graph-aware hyperbox refinement layer. The implementation is intentionally modular so that the baseline is evaluated separately from the refinement logic and so that ablations can be compared using a consistent dataset split.

## Literature themes

- Graph neural networks for node classification
- Graph embeddings extracted from GCN encoders
- Fuzzy min-max neural networks and hyperbox classifiers
- Hyperbox compression via merging and pruning
- Graph-aware prototype refinement and decision-region regularization
- Fuzzy representations on graphs

## Implementation status

- Data loading and split validation: complete
- GCN baseline scaffold: complete
- FMNN adapter and hyperbox tracking: complete
- Graph statistics, merge, and prune logic: complete
- Graph-aware refinement loop: complete with cohesion-aware merge checks and validation-safe pruning
- Experiment configuration and smoke scripts: complete
- Full PyTorch / PyG execution: complete in a dedicated Python 3.11 environment
- Real Cora Planetoid dataset loading: complete
- Held-out evaluation reporting: complete; validation is used for refinement decisions and test is reserved for final variant metrics
- Label leakage guard: complete; graph-aware refinement receives training labels only
- Objective-driven candidate acceptance: complete; merges and prunes are accepted only when the validation objective is preserved or improved
- Validation-only hyperparameter selection: complete; selected FMNN theta is 0.5 with gamma 1.0 and merge side limit 1.0
- Frozen held-out test evaluation: complete; final results are stored in `results/tables/final_evaluation.json`
- Multi-seed robustness evaluation: configured for five seeds with standard deviations and 95% confidence-interval half-widths
- Conservative validation-improving pruning: complete; pruning now accepts only strict validation-objective improvements and recomputes class coverage after each removal

## Verification commands

```bash
python -m pytest -q
python scripts/train_baseline.py --config configs/cora.yaml
python scripts/run_ablation.py --config configs/experiments.yaml
python scripts/evaluate.py --results-dir results
python scripts/final_evaluation.py
```

## Current result interpretation

The ablation artifact reports separate train, validation, and test metrics. The current Cora runs expose multiple FMNN boxes and apply validation-objective acceptance to merge/prune candidates. Test metrics are reported only after refinement decisions are complete. Current results should be treated as an experimental result, not a guaranteed improvement: the best variant can vary by seed.

The frozen validation-selected configuration produced the following five-seed test means after conservative pruning:

- Baseline: accuracy 0.7800, macro-F1 0.7666
- Merge: accuracy 0.7792, macro-F1 0.7675, delta -0.0008
- Prune: accuracy 0.7676, macro-F1 0.7494, delta -0.0124
- Full: accuracy 0.7778, macro-F1 0.7657, delta -0.0022

Pruning now retains approximately 98.5% of the baseline boxes on average instead of collapsing to a small, unsafe representation. These are held-out test results after validation-only selection; they should be reported as the current experiment outcome, not as a universal claim of improvement.

The robustness evaluation now uses five seeds and reports uncertainty in `results/tables/final_evaluation.json`. Confidence intervals describe variation across these experiment seeds; they are not a substitute for additional datasets or repeated graph splits.
