# GSHR v2 Experimental Report & Multi-Dataset Comparison

Evaluated on datasets `Cora, CiteSeer, Chameleon` across seeds `[0, 1, 2, 3, 4]` (all selection on validation only; test evaluated once).

## Cora (Nodes=2708, Classes=7, Homophily h=0.8100)

**B10_final validation-selected components:** `best_encoder=True`, `l2_norm=True`, `fast_gshr=True`, `hybrid=True`, `pseudo=True`

| Variant | Test Acc | Test Macro-F1 | Boxes | Refine Time (s) | Latency (us/node) | Peak Mem (MB) | Box Storage (B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_gnn_softmax** | 0.8034 ± 0.0086 | 0.7969 ± 0.0078 | 0.0 ± 0.0 | 0.000 ± 0.000 | 10.94 ± 1.78 | 0.86 ± 0.00 | 0 ± 0 |
| **B2_gnn_fmnn** | 0.7792 ± 0.0130 | 0.7780 ± 0.0102 | 13.6 ± 1.7 | 0.000 ± 0.000 | 10.49 ± 1.29 | 2.68 ± 0.28 | 6963 ± 857 |
| **B6_gshr_full** | 0.7760 ± 0.0113 | 0.7738 ± 0.0077 | 7.0 ± 0.0 | 1.047 ± 0.123 | 4.28 ± 0.26 | 9.69 ± 1.07 | 3584 ± 0 |
| **B7_hybrid** | 0.8040 ± 0.0066 | 0.7968 ± 0.0051 | 7.0 ± 0.0 | 0.949 ± 0.082 | 0.27 ± 0.13 | 9.19 ± 0.29 | 3584 ± 0 |
| **B8_hybrid_pseudo** | 0.7992 ± 0.0068 | 0.7923 ± 0.0058 | 7.0 ± 0.0 | 22.121 ± 8.413 | 0.13 ± 0.05 | 51.06 ± 14.92 | 3584 ± 0 |
| **B9_gshr_fast** | 0.7966 ± 0.0051 | 0.7900 ± 0.0039 | 7.2 ± 0.4 | 0.835 ± 0.074 | 5.00 ± 0.91 | 9.00 ± 0.28 | 3686 ± 229 |
| **B10_final** | 0.8026 ± 0.0071 | 0.7947 ± 0.0068 | 10.4 ± 1.5 | 16.736 ± 7.233 | 0.13 ± 0.05 | 50.00 ± 14.79 | 5325 ± 776 |

- **B10_final vs B1:** Accuracy gap `-0.0008` (paired t-test: `t=-0.146`, `p=0.8912`, `n=5`), Macro-F1 gap `-0.0022` (`t=-0.390`, `p=0.7167`)
- **Box reduction vs B2:** `23.5%` (`13.6` -> `10.4`)

## CiteSeer (Nodes=3327, Classes=6, Homophily h=0.7355)

**B10_final validation-selected components:** `best_encoder=True`, `l2_norm=True`, `fast_gshr=True`, `hybrid=True`, `pseudo=True`

| Variant | Test Acc | Test Macro-F1 | Boxes | Refine Time (s) | Latency (us/node) | Peak Mem (MB) | Box Storage (B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_gnn_softmax** | 0.6754 ± 0.0192 | 0.6434 ± 0.0142 | 0.0 ± 0.0 | 0.000 ± 0.000 | 21.89 ± 6.17 | 0.85 ± 0.00 | 0 ± 0 |
| **B2_gnn_fmnn** | 0.6510 ± 0.0154 | 0.6280 ± 0.0148 | 12.8 ± 4.2 | 0.000 ± 0.000 | 11.18 ± 3.76 | 2.78 ± 0.45 | 6554 ± 2154 |
| **B6_gshr_full** | 0.6500 ± 0.0162 | 0.6266 ± 0.0161 | 6.0 ± 0.0 | 1.091 ± 0.353 | 4.43 ± 0.78 | 9.16 ± 2.69 | 3072 ± 0 |
| **B7_hybrid** | 0.6844 ± 0.0105 | 0.6504 ± 0.0107 | 6.4 ± 0.5 | 1.142 ± 0.152 | 0.27 ± 0.12 | 9.56 ± 0.87 | 3277 ± 280 |
| **B8_hybrid_pseudo** | 0.6914 ± 0.0040 | 0.6521 ± 0.0055 | 6.6 ± 0.5 | 7.869 ± 2.117 | 0.11 ± 0.02 | 21.67 ± 4.38 | 3379 ± 280 |
| **B9_gshr_fast** | 0.6814 ± 0.0098 | 0.6470 ± 0.0077 | 6.2 ± 0.4 | 0.996 ± 0.096 | 5.12 ± 0.90 | 9.37 ± 0.85 | 3174 ± 229 |
| **B10_final** | 0.6906 ± 0.0029 | 0.6520 ± 0.0053 | 8.8 ± 2.2 | 6.805 ± 1.473 | 0.12 ± 0.03 | 21.04 ± 4.32 | 4506 ± 1110 |

- **B10_final vs B1:** Accuracy gap `+0.0152` (paired t-test: `t=+1.693`, `p=0.1658`, `n=5`), Macro-F1 gap `+0.0086` (`t=+1.445`, `p=0.2219`)
- **Box reduction vs B2:** `31.2%` (`12.8` -> `8.8`)

## Chameleon (Nodes=2277, Classes=5, Homophily h=0.2299)

**B10_final validation-selected components:** `best_encoder=False`, `l2_norm=False`, `fast_gshr=True`, `hybrid=True`, `pseudo=True`

| Variant | Test Acc | Test Macro-F1 | Boxes | Refine Time (s) | Latency (us/node) | Peak Mem (MB) | Box Storage (B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_gnn_softmax** | 0.6697 ± 0.0150 | 0.6704 ± 0.0158 | 0.0 ± 0.0 | 0.000 ± 0.000 | 50.73 ± 25.48 | 1.19 ± 0.00 | 0 ± 0 |
| **B2_gnn_fmnn** | 0.6447 ± 0.0056 | 0.6438 ± 0.0052 | 29.0 ± 2.3 | 0.000 ± 0.000 | 38.47 ± 25.61 | 7.90 ± 0.56 | 14848 ± 1201 |
| **B6_gshr_full** | 0.6535 ± 0.0182 | 0.6528 ± 0.0172 | 6.2 ± 0.8 | 6.075 ± 1.747 | 4.44 ± 1.75 | 20.82 ± 1.52 | 3174 ± 428 |
| **B7_hybrid** | 0.6443 ± 0.0108 | 0.6471 ± 0.0096 | 25.0 ± 9.2 | 25.667 ± 8.079 | 0.35 ± 0.26 | 75.19 ± 3.50 | 12800 ± 4707 |
| **B8_hybrid_pseudo** | 0.6447 ± 0.0169 | 0.6470 ± 0.0162 | 26.2 ± 8.0 | 59.439 ± 19.702 | 0.17 ± 0.09 | 82.63 ± 2.43 | 13414 ± 4118 |
| **B9_gshr_fast** | 0.6636 ± 0.0188 | 0.6657 ± 0.0195 | 49.0 ± 6.9 | 12.856 ± 2.756 | 63.62 ± 25.73 | 73.95 ± 3.47 | 25088 ± 3510 |
| **B10_final** | 0.6697 ± 0.0150 | 0.6704 ± 0.0158 | 10.6 ± 3.2 | 23.027 ± 5.766 | 0.17 ± 0.06 | 29.86 ± 1.48 | 5427 ± 1643 |

- **B10_final vs B1:** Accuracy gap `+0.0000` (paired t-test: `t=+0.000`, `p=1.0000`, `n=5`), Macro-F1 gap `+0.0000` (`t=+1.000`, `p=0.3739`)
- **Box reduction vs B2:** `63.4%` (`29.0` -> `10.6`)

## Validation-Only Theta Sensitivity (`theta_sensitivity_val.csv`)

| Dataset | Optimal theta* (Val) | Val Accuracy | Val Macro-F1 | Base Boxes |
| :--- | :---: | :---: | :---: | :---: |
| **Cora** | 0.4 | 0.7848 | 0.7759 | 7.8 |
| **CiteSeer** | 0.2 | 0.6752 | 0.6431 | 23.6 |
| **Chameleon** | 0.1 | 0.6903 | 0.6917 | 298.2 |
