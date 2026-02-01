# Baseline Fairness Report

- data_dir: /home/runner/work/FMFD-python/FMFD-python/Output/sim_spectrum
- split_indices: /home/runner/work/FMFD-python/FMFD-python/Output/compare_methods/split_indices.json
- seed: 2025

| method | sys_acc | macro_f1 | mod_top1 | n_rules | n_params | n_features | missing_features |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ours | 0.9375 | 0.9378 | 0.0000 | 48 | 68 | 22 | 0 |
| hcf | 0.6750 | 0.6494 | 0.0000 | 90 | 130 | 77 | 0 |
| brb_p | 0.4250 | 0.3087 | 0.0000 | 81 | 571 | 15 | 0 |
| brb_mu | 0.6875 | 0.6887 | 0.0000 | 72 | 110 | 77 | 0 |
| dbrb | 0.7250 | 0.7007 | 0.0000 | 60 | 90 | 77 | 0 |
| a_ibrb | 0.3625 | 0.2987 | 0.0000 | 30 | 65 | 5 | 0 |

## Confusion Matrices
- confusion_matrix_ours.png
- confusion_matrix_hcf.png
- confusion_matrix_brb_p.png
- confusion_matrix_brb_mu.png
- confusion_matrix_dbrb.png
- confusion_matrix_a_ibrb.png
