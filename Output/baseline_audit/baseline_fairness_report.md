# Baseline Fairness Report

- data_dir: /home/runner/work/FMFD-python/FMFD-python/Output/sim_spectrum
- split_indices: /home/runner/work/FMFD-python/FMFD-python/Output/compare_methods_test/split_indices.json
- seed: 2025

| method | sys_acc | macro_f1 | mod_top1 | mod_top3 | n_rules | n_params | n_features | missing_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours | 0.3000 | 0.1920 | 0.0667 | 0.0833 | 48 | 68 | 22 | 0 |
| hcf | 0.5250 | 0.4439 | 0.0000 | 0.0000 | 90 | 130 | 40 | 0 |

## Confusion Matrices
- confusion_matrix_ours.png
- confusion_matrix_hcf.png
