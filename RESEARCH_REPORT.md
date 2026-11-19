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
- CiteSeer generalization evaluation: configured with the frozen Cora-selected configuration; no CiteSeer retuning is performed

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

The frozen Cora-selected configuration was also evaluated on CiteSeer without retuning. CiteSeer results are stored in `results/tables/final_evaluation_citeseer.json`:

- Baseline: accuracy 0.6398, macro-F1 0.6088
- Merge: accuracy 0.6394, macro-F1 0.6086, delta -0.0004
- Prune: accuracy 0.6426, macro-F1 0.6081, delta +0.0028
- Full: accuracy 0.6544, macro-F1 0.6144, delta +0.0146

The frozen refinement behavior generalizes more favorably to CiteSeer than to Cora: full refinement improves CiteSeer accuracy by 1.46 percentage points while reducing the mean box count from 90.8 to 53.6. This is cross-dataset evidence, not a claim of universal improvement.

## Latest verification run

On 2026-11-01, the config-driven Cora baseline completed with seed 42 using the public Planetoid split. The baseline achieved test accuracy 0.7500 and test Macro-F1 0.7325 with 90 hyperboxes. The full GSHR run used the same seed and validation-only refinement decisions, reduced the representation to 47 hyperboxes, and achieved test accuracy 0.7750 and test Macro-F1 0.7553. The artifacts are stored under `results/processed/baseline_seed_42_20261101T060632Z` and `results/processed/gshr_full_seed_42_20261101T061112Z`.

This is a single-seed verification result. It does not establish a general improvement; multi-seed and cross-dataset evaluation remain necessary.

## Current five-seed rerun

The frozen five-seed Cora evaluation was rerun on 2026-11-01 with the current implementation and `results/tables/validation_tuning.json`. The current aggregate is materially different from the earlier stored evaluation: baseline mean test accuracy is 0.7620 with Macro-F1 0.7544 and 12.2 boxes; geometry merge and full GSHR both average 9.0 boxes, accuracy 0.7524, and Macro-F1 0.7471. Full GSHR therefore has accuracy delta -0.0096 and Macro-F1 delta -0.0073 versus baseline in this rerun.

The earlier `final_evaluation.json` values should be treated as historical output from a previous implementation state. The current rerun is stored in `results/tables/final_evaluation_current.json`. These results do not support claiming a predictive improvement from GSHR, although they do show a compactness reduction under the selected configuration.

## E1-E8 ablation run

The five-seed E1-E8 matrix completed on 2026-11-01 and is stored in `results/tables/ablation_summary.json`. Mean results were: E1 softmax accuracy 0.8042 and Macro-F1 0.7946; E2 FMNN accuracy 0.7528 and Macro-F1 0.7391 with 87.4 boxes; E3 geometry-only merge accuracy 0.7436 and Macro-F1 0.7367 with 45.4 boxes; E4 graph-aware merge accuracy 0.7542 and Macro-F1 0.7395 with 45.4 boxes; E5 pruning-only accuracy 0.7528 and Macro-F1 0.7391 with 87.4 boxes; and E8 unsafe full refinement accuracy 0.7492 and Macro-F1 0.7288 with 43.8 boxes.

The first matrix run selected the baseline entry when recording E6 and E7 because full mode returns a multi-variant summary. That reporting bug is fixed in `scripts/run_ablation.py`; corrected E6/E7 results are stored separately in `results/tables/ablation_e6_e7_summary.json`: E6 full graph-aware refinement averages accuracy 0.7492, Macro-F1 0.7288, and 43.8 boxes; E7 graph-disabled full refinement averages accuracy 0.7630, Macro-F1 0.7446, and 43.2 boxes. The original E6/E7 rows in `ablation_summary.json` should not be used as final evidence until the complete matrix is rerun with the corrected selector.

The authoritative rerun is stored in `results/tables/ablation_summary_corrected.json`. Its corrected means are: E1 softmax accuracy 0.8042 and Macro-F1 0.7946; E2 baseline accuracy 0.7528 and Macro-F1 0.7391 with 87.4 boxes; E3 geometry-only merge accuracy 0.7436 and Macro-F1 0.7367 with 45.4 boxes; E4 graph-aware merge accuracy 0.7542 and Macro-F1 0.7395 with 45.4 boxes; E5 pruning-only accuracy 0.7528 and Macro-F1 0.7391 with 87.4 boxes; E6 full GSHR accuracy 0.7492 and Macro-F1 0.7288 with 43.8 boxes; E7 graph-disabled full accuracy 0.7630 and Macro-F1 0.7446 with 43.2 boxes; and E8 unsafe full accuracy 0.7492 and Macro-F1 0.7288 with 43.8 boxes.

## CiteSeer frozen evaluation

The Cora-selected configuration was evaluated on CiteSeer without retuning on 2026-11-01. Results are stored in `results/tables/final_evaluation_citeseer_current.json`. Baseline means were accuracy 0.6288, Macro-F1 0.6130, and 9.8 boxes. Full GSHR means were accuracy 0.6282, Macro-F1 0.6142, and 7.8 boxes. Accuracy changed by -0.0006 while Macro-F1 increased by 0.0012. Pruning-only reached accuracy 0.6350 but Macro-F1 decreased to 0.6069. These results suggest compactness is reproducible across datasets, while predictive gains remain inconsistent and modest.
