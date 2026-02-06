# Baseline Fairness Report

- data_dir: /home/runner/work/FMFD-python/FMFD-python/Output/sim_spectrum
- split_indices: /home/runner/work/FMFD-python/FMFD-python/Output/compare_methods/split_indices.json
- seed: 2025

| method | sys_acc | macro_f1 | mod_top1 | mod_top3 | n_rules | n_params | n_features | missing_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours | 0.1833 | 0.1805 | 0.0167 | 0.1333 | 48 | 68 | 22 | 0 |
| hcf | 0.0333 | 0.0244 | 0.0000 | 0.0000 | 90 | 130 | 76 | 0 |

## Confusion Matrices
- confusion_matrix_ours.png
- confusion_matrix_hcf.png
