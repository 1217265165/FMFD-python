#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate simulation labels against module hierarchy constraints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from tools.label_mapping import (
    HIERARCHY_MAP,
    expected_system_class_for_module,
    get_system_class_en,
    canonical_module_key,
)


def _load_labels(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"labels.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_labels(labels: Dict[str, dict]) -> List[str]:
    issues: List[str] = []
    for sample_id, info in labels.items():
        sys_class_raw = info.get("system_fault_class")
        sys_class = get_system_class_en(sys_class_raw)
        module_name = info.get("module_v2") or info.get("module") or ""
        module_key = canonical_module_key(module_name)
        expected = expected_system_class_for_module(module_name)

        if module_key == "Power" and sys_class in ("amp_error", "freq_error", "ref_error"):
            issues.append(f"{sample_id}: Label=Amp/Freq/Ref but Module=Power ({module_name})")
        if module_key == "RF_Match" and sys_class == "freq_error":
            issues.append(f"{sample_id}: Label=Freq but Module=RF_Match ({module_name})")

        if expected and expected != "other" and sys_class and sys_class != expected:
            issues.append(
                f"{sample_id}: system_fault_class={sys_class_raw} does not match module {module_name} "
                f"(expected {expected})"
            )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate simulation labels")
    parser.add_argument(
        "--labels",
        default="Output/sim_spectrum/labels.json",
        help="Path to labels.json (default: Output/sim_spectrum/labels.json)",
    )
    args = parser.parse_args()
    labels_path = Path(args.labels)

    labels = _load_labels(labels_path)
    issues = validate_labels(labels)

    if issues:
        print("[ERROR] Label validation failed:", file=sys.stderr)
        for issue in issues[:200]:
            print(f"  - {issue}", file=sys.stderr)
        print(f"[SUMMARY] {len(issues)} issues detected.", file=sys.stderr)
        return 1

    print("[OK] Label validation passed.")
    print(f"[SUMMARY] {len(labels)} samples checked. Keys: {len(HIERARCHY_MAP)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
