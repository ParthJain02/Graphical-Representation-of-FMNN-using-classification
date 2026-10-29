# Graph-Structure-Aware Hyperbox Refinement for Graph Neural Network-Based Node Classification

This repository contains a reproducible research scaffold for a graph neural network + fuzzy min-max neural network pipeline. It follows the requested research phases and keeps the baseline model separate from the GSHR refinement logic.

## Project structure

- `src/`: reusable baseline and refinement modules.
- `scripts/`: config-driven command line entry points.
- `configs/`: YAML experiment templates.
- `tests/`: validation tests for data, graph statistics, merging, and pruning safety.
- `results/`: experiment outputs, tables, and figures.

## Baseline status

The project supports a synthetic Cora-like fallback and the real Cora, CiteSeer, and PubMed benchmarks through PyG's Planetoid loader. The default configuration points at the real Cora dataset in a dedicated Python 3.11 + PyTorch + PyTorch Geometric environment.

## Quick start

```bash
python -m pytest -q
python scripts/train_baseline.py --config configs/cora.yaml
python scripts/run_gshr.py --config configs/cora.yaml --mode full
python scripts/run_ablation.py --config configs/experiments.yaml
python scripts/evaluate.py --results-dir results/
python scripts/tune_validation.py --config configs/experiments.yaml
python scripts/final_evaluation.py
python scripts/final_evaluation.py --experiments configs/citeseer_experiments.yaml --output results/tables/final_evaluation_citeseer.json
```

## Notes

- Dataset validation enforces disjoint train/val/test masks.
- Embeddings and FMNN hyperboxes are modeled with explicit safety checks.
- GSHR refinement components are separated from the baseline for ablation studies.
- Refinement decisions use the validation mask, while final ablation metrics use the held-out test mask.
- Ablation output includes split-level metrics in `results/tables/ablation_summary.json`.
- Candidate merges and prunes are accepted through a validation objective combining validation accuracy and macro-F1.
- Validation tuning writes `results/tables/validation_tuning.json`; final held-out evaluation writes `results/tables/final_evaluation.json`.
- Final evaluation reports five-seed standard deviations and 95% confidence-interval half-widths.
- Pruning is conservative: a candidate is removed only when the validation objective strictly improves and class coverage remains safe.
- CiteSeer can be evaluated with the Cora-selected configuration using `configs/citeseer_experiments.yaml`; results are written to `results/tables/final_evaluation_citeseer.json`.
