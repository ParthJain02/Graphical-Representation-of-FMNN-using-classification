# GSHR v4 Experimental Report

Datasets: `Cora, CiteSeer` | Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` | Selection strictly on validation macro-F1.

## 1. Per-Dataset Encoder Comparison (Validation-Selected Configs, 10-Seed B1)

| Dataset | Encoder | Best Val Acc | Best Val F1 | Test Acc | Test Macro-F1 | Selected |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Cora | gcn | 0.7960 ± 0.0098 | 0.7850 ± 0.0093 | 0.8080 ± 0.0060 | 0.8014 ± 0.0054 |  |
| Cora | sage | 0.7820 ± 0.0133 | 0.7775 ± 0.0119 | 0.7930 ± 0.0152 | 0.7861 ± 0.0134 |  |
| Cora | appnp | 0.8166 ± 0.0149 | 0.8075 ± 0.0115 | 0.8214 ± 0.0130 | 0.8138 ± 0.0096 | YES |
| Cora | gcnii | 0.8008 ± 0.0117 | 0.7915 ± 0.0105 | 0.8051 ± 0.0075 | 0.7997 ± 0.0063 |  |
| CiteSeer | gcn | 0.7074 ± 0.0069 | 0.6686 ± 0.0063 | 0.6877 ± 0.0087 | 0.6531 ± 0.0091 |  |
| CiteSeer | sage | 0.6966 ± 0.0111 | 0.6584 ± 0.0110 | 0.6760 ± 0.0167 | 0.6427 ± 0.0142 |  |
| CiteSeer | appnp | 0.7034 ± 0.0127 | 0.6692 ± 0.0076 | 0.6952 ± 0.0141 | 0.6610 ± 0.0118 |  |
| CiteSeer | gcnii | 0.7052 ± 0.0119 | 0.6718 ± 0.0145 | 0.6881 ± 0.0116 | 0.6516 ± 0.0090 | YES |

## Cora (Best Encoder: `appnp`)

- **B14_final validation-selected components:** `{'use_l2_norm': True, 'use_ensemble': False, 'ensemble_embed_mode': 'concat', 'use_fast_gshr': True, 'use_capped_pseudo': True, 'hybrid_mode': 'global_lam', 'use_smoothing': False}`

| Variant | Test Acc | Test Macro-F1 | Boxes | Refine (s) | Lat (us/node) | Peak Mem (MB) | Storage (B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_base_gcn** | 0.8032 ± 0.0101 | 0.7963 ± 0.0099 | 0.0 ± 0.0 | 0.000 ± 0.000 | 3.42 ± 0.13 | 0.86 ± 0.00 | 0 |
| **B1_best_encoder** | 0.8214 ± 0.0130 | 0.8138 ± 0.0096 | 0.0 ± 0.0 | 0.000 ± 0.000 | 7.45 ± 0.25 | 0.86 ± 0.00 | 0 |
| **B11_final_lite** | 0.8020 ± 0.0082 | 0.7950 ± 0.0070 | 8.4 ± 1.3 | 3.152 ± 0.116 | 8.57 ± 0.49 | 18.81 ± 1.35 | 4301 |
| **B7_hybrid** | 0.8202 ± 0.0132 | 0.8130 ± 0.0100 | 7.3 ± 0.5 | 0.696 ± 0.119 | 10.54 ± 0.28 | 9.91 ± 1.59 | 3738 |
| **B13_gated_hybrid** | 0.8214 ± 0.0130 | 0.8138 ± 0.0096 | 7.4 ± 0.7 | 0.679 ± 0.076 | 11.10 ± 0.38 | 9.85 ± 1.57 | 3789 |
| **B12_ensemble** | 0.8270 ± 0.0054 | 0.8183 ± 0.0036 | 7.2 ± 0.4 | 2.576 ± 0.139 | 48.82 ± 10.25 | 53.62 ± 3.72 | 9830 |
| **B14_final** | 0.8180 ± 0.0136 | 0.8104 ± 0.0116 | 7.8 ± 1.0 | 3.356 ± 0.259 | 10.06 ± 0.30 | 15.17 ± 1.14 | 3994 |

- **B13_gated_hybrid diagnostics:** box-decided fraction `0.163`, accuracy on box-decided `0.9820` vs rest `0.7901`, predictions flipped vs GNN `0.0000`, vs B7 global lam Test Acc `0.8204`.

## CiteSeer (Best Encoder: `gcnii`)

- **B14_final validation-selected components:** `{'use_l2_norm': True, 'use_ensemble': True, 'ensemble_embed_mode': 'avg', 'use_fast_gshr': True, 'use_capped_pseudo': False, 'hybrid_mode': 'global_lam', 'use_smoothing': False}`

| Variant | Test Acc | Test Macro-F1 | Boxes | Refine (s) | Lat (us/node) | Peak Mem (MB) | Storage (B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_base_gcn** | 0.6818 ± 0.0160 | 0.6492 ± 0.0136 | 0.0 ± 0.0 | 0.000 ± 0.000 | 8.14 ± 0.47 | 0.85 ± 0.00 | 0 |
| **B1_best_encoder** | 0.6881 ± 0.0116 | 0.6516 ± 0.0090 | 0.0 ± 0.0 | 0.000 ± 0.000 | 13.07 ± 0.72 | 0.85 ± 0.00 | 0 |
| **B11_final_lite** | 0.6912 ± 0.0079 | 0.6542 ± 0.0084 | 7.2 ± 1.1 | 3.063 ± 0.199 | 13.57 ± 0.66 | 15.04 ± 0.76 | 3686 |
| **B7_hybrid** | 0.6866 ± 0.0110 | 0.6506 ± 0.0077 | 7.7 ± 0.8 | 0.740 ± 0.155 | 16.61 ± 0.34 | 10.49 ± 2.18 | 3942 |
| **B13_gated_hybrid** | 0.6880 ± 0.0115 | 0.6515 ± 0.0089 | 8.1 ± 1.8 | 0.802 ± 0.127 | 17.35 ± 0.76 | 10.42 ± 2.16 | 4147 |
| **B12_ensemble** | 0.6982 ± 0.0100 | 0.6648 ± 0.0077 | 7.1 ± 1.3 | 2.941 ± 0.200 | 86.45 ± 17.93 | 57.07 ± 5.02 | 12237 |
| **B14_final** | 0.6954 ± 0.0079 | 0.6625 ± 0.0068 | 6.9 ± 1.0 | 0.865 ± 0.056 | 74.17 ± 1.67 | 19.76 ± 0.00 | 3533 |

- **B13_gated_hybrid diagnostics:** box-decided fraction `0.146`, accuracy on box-decided `0.9229` vs rest `0.6472`, predictions flipped vs GNN `0.0001`, vs B7 global lam Test Acc `0.6883`.

## Paired t-Tests (10 Seeds vs B1 and B11)

| Dataset | Variant vs Baseline | dAcc (p-val) | dMacro-F1 (p-val) | dBoxes (p-val) | dRefineTime_s (p-val) | dPeakMem_MB (p-val) | dLatency_us (p-val) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Cora | B12_ensemble vs B1_base_gcn | +0.0238 (p=0.0002) | +0.0220 (p=0.0001) | +7.20 (p=0.0000) | +2.576 (p=0.0000) | +52.76 (p=0.0000) | +45.40 (p=0.0000) |
| Cora | B12_ensemble vs B1_best_encoder | +0.0056 (p=0.1874) | +0.0046 (p=0.1712) | +7.20 (p=0.0000) | +2.576 (p=0.0000) | +52.76 (p=0.0000) | +41.38 (p=0.0000) |
| Cora | B12_ensemble vs B11_final_lite | +0.0250 (p=0.0000) | +0.0234 (p=0.0000) | -1.20 (p=0.0368) | -0.575 (p=0.0000) | +34.81 (p=0.0000) | +40.26 (p=0.0000) |
| Cora | B13_gated_hybrid vs B1_base_gcn | +0.0182 (p=0.0057) | +0.0175 (p=0.0032) | +7.40 (p=0.0000) | +0.679 (p=0.0000) | +8.99 (p=0.0000) | +7.68 (p=0.0000) |
| Cora | B13_gated_hybrid vs B1_best_encoder | +0.0000 (p=1.0000) | +0.0000 (p=1.0000) | +7.40 (p=0.0000) | +0.679 (p=0.0000) | +8.99 (p=0.0000) | +3.65 (p=0.0000) |
| Cora | B13_gated_hybrid vs B11_final_lite | +0.0194 (p=0.0089) | +0.0188 (p=0.0025) | -1.00 (p=0.0957) | -2.473 (p=0.0000) | -8.96 (p=0.0000) | +2.53 (p=0.0000) |
| Cora | B14_final vs B1_base_gcn | +0.0148 (p=0.0230) | +0.0141 (p=0.0281) | +7.80 (p=0.0000) | +3.356 (p=0.0000) | +14.32 (p=0.0000) | +6.64 (p=0.0000) |
| Cora | B14_final vs B1_best_encoder | -0.0034 (p=0.1370) | -0.0034 (p=0.1259) | +7.80 (p=0.0000) | +3.356 (p=0.0000) | +14.31 (p=0.0000) | +2.62 (p=0.0000) |
| Cora | B14_final vs B11_final_lite | +0.0160 (p=0.0228) | +0.0154 (p=0.0097) | -0.60 (p=0.4048) | +0.204 (p=0.0321) | -3.63 (p=0.0003) | +1.50 (p=0.0001) |
| CiteSeer | B12_ensemble vs B1_base_gcn | +0.0164 (p=0.0165) | +0.0156 (p=0.0127) | +7.10 (p=0.0000) | +2.941 (p=0.0000) | +56.22 (p=0.0000) | +78.31 (p=0.0000) |
| CiteSeer | B12_ensemble vs B1_best_encoder | +0.0101 (p=0.0643) | +0.0132 (p=0.0107) | +7.10 (p=0.0000) | +2.941 (p=0.0000) | +56.22 (p=0.0000) | +73.39 (p=0.0000) |
| CiteSeer | B12_ensemble vs B11_final_lite | +0.0070 (p=0.0696) | +0.0106 (p=0.0136) | -0.10 (p=0.8402) | -0.121 (p=0.2350) | +42.03 (p=0.0000) | +72.88 (p=0.0000) |
| CiteSeer | B13_gated_hybrid vs B1_base_gcn | +0.0062 (p=0.3461) | +0.0023 (p=0.6819) | +8.10 (p=0.0000) | +0.802 (p=0.0000) | +9.57 (p=0.0000) | +9.20 (p=0.0000) |
| CiteSeer | B13_gated_hybrid vs B1_best_encoder | -0.0001 (p=0.3434) | -0.0001 (p=0.3434) | +8.10 (p=0.0000) | +0.802 (p=0.0000) | +9.57 (p=0.0000) | +4.28 (p=0.0000) |
| CiteSeer | B13_gated_hybrid vs B11_final_lite | -0.0032 (p=0.5097) | -0.0027 (p=0.4815) | +0.90 (p=0.1589) | -2.261 (p=0.0000) | -4.62 (p=0.0001) | +3.78 (p=0.0000) |
| CiteSeer | B14_final vs B1_base_gcn | +0.0136 (p=0.0169) | +0.0133 (p=0.0064) | +6.90 (p=0.0000) | +0.865 (p=0.0000) | +18.91 (p=0.0000) | +66.03 (p=0.0000) |
| CiteSeer | B14_final vs B1_best_encoder | +0.0073 (p=0.2086) | +0.0109 (p=0.0429) | +6.90 (p=0.0000) | +0.865 (p=0.0000) | +18.91 (p=0.0000) | +61.10 (p=0.0000) |
| CiteSeer | B14_final vs B11_final_lite | +0.0042 (p=0.2622) | +0.0083 (p=0.0212) | -0.30 (p=0.5203) | -2.198 (p=0.0000) | +4.72 (p=0.0000) | +60.60 (p=0.0000) |
