# Legacy Code Archive

This directory contains deprecated code that has been archived during the V-D.1 architecture refactoring.

## Archived Files

### utils/feature_pool.py
- **Reason**: Confirmed as unused dead code
- **Replacement**: Feature pools are now defined in `config/feature_definitions.json`

### BRB/system_brb_amp.py, BRB/system_brb_freq.py, BRB/system_brb_ref.py
- **Reason**: Replaced by the new unified layered engine
- **Replacement**: `BRB/engines/layered_engine.py` + `config/feature_definitions.json`

## Archive Date
2026-02-06

## Notes
- Do NOT delete these files until the new architecture is fully validated
- They may be needed for reference during debugging
