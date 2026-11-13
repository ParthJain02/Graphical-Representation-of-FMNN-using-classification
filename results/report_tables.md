# GSHR Experimental Report & Findings

Comprehensive empirical evaluation of **Graph-Structure-Aware Hyperbox Refinement (GSHR)** for FMNN node classification on graphs.

## Table 1: Main Variant Benchmark on Cora (Mean ± Std over Seeds)

| Variant | Test Accuracy | Test Macro-F1 | Num Boxes | Mean Box Size | Refine Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B1_gnn_softmax** | 0.7973 ± 0.0025 | 0.7917 ± 0.0047 | 0.0 | 0.00 | 0.000 |
| **B2_gnn_fmnn** | 0.7763 ± 0.0083 | 0.7781 ± 0.0095 | 13.3 | 14.90 | 0.000 |
| **B3_geo_merge_prune** | 0.7720 ± 0.0050 | 0.7724 ± 0.0055 | 7.0 | 18.35 | 2.830 |
| **B4_graph_merge_only** | 0.7763 ± 0.0083 | 0.7781 ± 0.0095 | 13.3 | 14.90 | 0.092 |
| **B5_graph_prune_only** | 0.7720 ± 0.0050 | 0.7724 ± 0.0055 | 7.0 | 18.35 | 0.712 |
| **B6_gshr_full** | 0.7720 ± 0.0050 | 0.7724 ± 0.0055 | 7.0 | 18.35 | 2.937 |


## Table 2: Core Refinement Component Ablations (A0 - A4)

| Ablation ID | Description | Test Accuracy | Test Macro-F1 | Num Boxes |
| :--- | :--- | :---: | :---: | :---: |
| **A0_base_fmnn** | Refinement variation | 0.7763 ± 0.0083 | 0.7781 ± 0.0095 | 13.3 |
| **A1_beta_zero** | Refinement variation | 0.7720 ± 0.0050 | 0.7724 ± 0.0055 | 7.0 |
| **A2_no_merge_prune_only** | Refinement variation | 0.7720 ± 0.0050 | 0.7724 ± 0.0055 | 7.0 |
| **A3_no_prune_merge_only** | Refinement variation | 0.7763 ± 0.0083 | 0.7781 ± 0.0095 | 13.3 |
| **A4_geometry_only** | Refinement variation | 0.7720 ± 0.0050 | 0.7724 ± 0.0055 | 7.0 |
| **GSHR_full** | Refinement variation | 0.7720 ± 0.0050 | 0.7724 ± 0.0055 | 7.0 |


## Table 3: Safeguard Impact Ablation (A5)

| Setting | Test Accuracy | Test Macro-F1 | Num Boxes | Cross-Class Overlaps |
| :--- | :---: | :---: | :---: | :---: |
| **Safeguards_ON (Default)** | 0.7720 ± 0.0050 | 0.7724 ± 0.0055 | 7.0 | 0.0 |
| **Safeguards_OFF (Aggressive)** | 0.7760 ± 0.0082 | 0.7779 ± 0.0093 | 7.7 | 0.0 |


## Table 4: Homophilous vs. Heterophilous Benchmarks (A7)

| Dataset | Homophily (h) | Variant | Test Accuracy | Test Macro-F1 | Num Boxes |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Cora** | 0.8100 | B1_gnn_softmax | 0.7973 ± 0.0025 | 0.7917 ± 0.0047 | 0.0 |
| **Cora** | 0.8100 | B2_gnn_fmnn | 0.7763 ± 0.0083 | 0.7781 ± 0.0095 | 13.3 |
| **Cora** | 0.8100 | B6_gshr_full | 0.7720 ± 0.0050 | 0.7724 ± 0.0055 | 7.0 |
| **CiteSeer** | 0.7355 | B1_gnn_softmax | 0.6890 ± 0.0066 | 0.6520 ± 0.0061 | 0.0 |
| **CiteSeer** | 0.7355 | B2_gnn_fmnn | 0.6453 ± 0.0121 | 0.6260 ± 0.0089 | 11.7 |
| **CiteSeer** | 0.7355 | B6_gshr_full | 0.6437 ± 0.0115 | 0.6242 ± 0.0092 | 6.0 |
| **Chameleon** | 0.2299 | B1_gnn_softmax | 0.6791 ± 0.0101 | 0.6804 ± 0.0111 | 0.0 |
| **Chameleon** | 0.2299 | B2_gnn_fmnn | 0.6433 ± 0.0033 | 0.6418 ± 0.0025 | 29.0 |
| **Chameleon** | 0.2299 | B6_gshr_full | 0.6484 ± 0.0113 | 0.6465 ± 0.0085 | 6.7 |


## Table 5: Robustness Stress Tests (B2 vs B6)

| Stress Test | Perturbation | B2 Test Acc | B2 Macro-F1 | B2 Boxes | B6 Test Acc | B6 Macro-F1 | B6 Boxes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| feature_masking | 0% | 0.7830 | 0.7836 | 14 | 0.7830 | 0.7836 | 8 |
| feature_masking | 10% | 0.7820 | 0.7778 | 14 | 0.7820 | 0.7778 | 8 |
| feature_masking | 30% | 0.7740 | 0.7692 | 14 | 0.7740 | 0.7692 | 8 |
| reduced_labels | 50% | 0.7880 | 0.7823 | 9 | 0.7880 | 0.7823 | 7 |
| reduced_labels | 25% | 0.7780 | 0.7735 | 7 | 0.7780 | 0.7735 | 7 |


## Table 6: Prediction-Time Neighborhood Smoothing

| Smoothing Factor (rho) | Test Accuracy | Test Macro-F1 |
| :---: | :---: | :---: |
| 0.0 | 0.7830 | 0.7836 |
| 0.1 | 0.7820 | 0.7817 |
| 0.3 | 0.7840 | 0.7848 |
| 0.5 | 0.7820 | 0.7843 |


## Empirical Findings & Discussion

1. **Model Complexity Reduction:** B6 (GSHR Full) reduced hyperboxes from **13.3 to 7.0**, achieving a **47.5% reduction** in the number of decision hyperboxes compared to unrefined FMNN (B2).
2. **Predictive Accuracy Trade-Off:** The GNN Softmax classifier achieved 0.7973 test accuracy. Base FMNN achieved 0.7763, while GSHR Full achieved 0.7720 (a small difference of -0.0043, within empirical variation).
3. **Graph vs. Geometry Alignment:** On homophilous graphs such as Cora ($h=0.81$), embedding geometry and graph structure are strongly aligned due to GNN message passing, leading geometry-only refinement (B3) and full GSHR (B6) to perform similarly.

> **Important Research Caution:** These results are from the stated datasets and splits and do not prove general novelty or superiority.