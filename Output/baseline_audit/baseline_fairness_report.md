# Baseline Fairness Report

- data_dir: /home/runner/work/FMFD-python/FMFD-python/Output/sim_spectrum
- split_indices: /home/runner/work/FMFD-python/FMFD-python/Output/compare_methods/split_indices.json
- seed: 2025

| method | sys_acc | macro_f1 | mod_top1 | n_rules | n_params | n_features | missing_features |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ours | 0.9500 | 0.9492 | 0.0000 | 48 | 68 | 22 | 0 |
| hcf | 0.6125 | 0.5360 | 0.0000 | 90 | 130 | 76 | 0 |
| brb_p | 0.3875 | 0.3881 | 0.0000 | 81 | 571 | 15 | 0 |
| brb_mu | 0.7750 | 0.7786 | 0.0000 | 72 | 110 | 76 | 0 |
| dbrb | 0.7000 | 0.6952 | 0.0000 | 60 | 90 | 76 | 0 |
| a_ibrb | 0.4375 | 0.3975 | 0.0000 | 45 | 65 | 5 | 0 |

## Confusion Matrices
- confusion_matrix_ours.png
- confusion_matrix_hcf.png
- confusion_matrix_brb_p.png
- confusion_matrix_brb_mu.png
- confusion_matrix_dbrb.png
- confusion_matrix_a_ibrb.png
