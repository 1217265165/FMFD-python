# Three-Pipeline Consistency Report

Generated: 2026-02-05T11:15:37.615873

---

## 1. Sample Set Comparison

### Compare vs CLI Batch

| Metric | Compare | CLI Batch |
|--------|---------|-----------|
| N_samples | 400 | 300 |
| Intersection | 300 | - |
| Sample sets identical? | False | - |

**Only in compare (first 20)**: ['sim_00300', 'sim_00301', 'sim_00302', 'sim_00303', 'sim_00304', 'sim_00305', 'sim_00306', 'sim_00307', 'sim_00308', 'sim_00309', 'sim_00310', 'sim_00311', 'sim_00312', 'sim_00313', 'sim_00314', 'sim_00315', 'sim_00316', 'sim_00317', 'sim_00318', 'sim_00319']

**Only in CLI (first 20)**: []

### CLI Batch vs Eval Module

| Metric | CLI Batch | Eval Module |
|--------|-----------|-------------|
| N_samples | 300 | 300 |
| Intersection | 300 | - |
| Sample sets identical? | True | - |

---

## 2. System-Level Accuracy Comparison

| Path | sys_acc | N_eval | Include Normal? |
|------|---------|--------|-----------------|
| Compare | N/A | 400 | True |
| CLI Batch | 0.4767 | 300 | False |
| Eval Module | N/A | 300 | False |

**sys_acc Comparable?**: 
- Compare vs CLI: No (different normal inclusion)
- CLI vs Eval: Yes

---

## 3. Module-Level Accuracy Comparison

| Path | mod_top1 | mod_top3 |
|------|----------|----------|
| Compare | N/A | N/A |
| CLI Batch | 0.33 | 0.6033 |
| Eval Module | 0.05333333333333334 | 0.15 |

### Consistency Check (CLI vs Eval)

| Metric | Diff (samples) | Consistent (≤1 sample)? |
|--------|----------------|-------------------------|
| mod_top1 | 83.0 | ❌ No |
| mod_top3 | 136.0 | ❌ No |

---

## 4. Architecture Summary

From `architecture_snapshot.md`:

- **System-level entry**: `methods/ours_adapter.py::infer_system_and_modules()`
- **System-level backend**: RF+BRB Gating Fusion (alpha*BRB + beta*RF)
- **Module-level backend**: hierarchical_module_infer_soft_gating() with delta=0.1
- **Feature pools**: NOT implemented (X1-X22 used uniformly)

---

## 5. Truth Fields

From `config/eval_truth.json`:

- **System truth field**: `system_fault_class`
- **Module truth field**: `module_v2`
- **Module eval policy**: Exclude normal samples

---

## 6. Conclusion

### Sample Set Consistency
- Compare vs CLI: ⚠️ Different
- CLI vs Eval: ✅ Identical

### Metric Consistency (CLI vs Eval)
- mod_top1: ❌ Inconsistent
- mod_top3: ❌ Inconsistent

### Overall Status
⚠️ Some inconsistencies detected - see details above

---
