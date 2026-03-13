from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModuleSpec:
    module_label: str
    system_label: str
    primary_templates: List[str]
    secondary_templates: List[str]


MODULE_LIBRARY: List[ModuleSpec] = [
    ModuleSpec("衰减器", "amp_error", ["T1"], ["T5"]),
    ModuleSpec("低频段前置低通滤波器", "amp_error", ["T2"], ["T3"]),
    ModuleSpec("低频段第一混频器", "amp_error", ["T2"], ["T5"]),
    ModuleSpec("本振混频组件", "amp_error", ["T2", "T5"], ["T8"]),
    ModuleSpec("校准源", "ref_error", ["T1"], []),
    ModuleSpec("存储器", "ref_error", ["T1", "T3"], ["T4"]),
    ModuleSpec("校准信号开关", "ref_error", ["T4"], ["T5"]),
    ModuleSpec("中频放大器", "amp_error", ["T1"], ["T2"]),
    ModuleSpec("ADC", "ref_error", ["T7"], ["T1"]),
    ModuleSpec("数字RBW", "amp_error", ["T3"], ["T1"]),
    ModuleSpec("数字放大器", "amp_error", ["T1"], ["T4"]),
    ModuleSpec("数字检波器", "ref_error", ["T1"], ["T6"]),
    ModuleSpec("VBW滤波器", "amp_error", ["T1"], ["T6"]),
    ModuleSpec("电源模块", "amp_error", ["T6"], ["T5"]),
    ModuleSpec("时钟振荡器", "freq_error", ["T8"], []),
    ModuleSpec("时钟合成与同步网络", "freq_error", ["T8"], []),
    ModuleSpec("本振源（谐波发生器）", "freq_error", ["T8"], []),
]


def module_templates(spec: ModuleSpec) -> List[str]:
    return list(dict.fromkeys(spec.primary_templates + spec.secondary_templates))


def module_specs_by_system(system_label: str, disabled_modules: List[str]) -> List[ModuleSpec]:
    return [
        spec
        for spec in MODULE_LIBRARY
        if spec.system_label == system_label and spec.module_label not in disabled_modules
    ]


def module_spec_by_label(module_label: str) -> ModuleSpec | None:
    for spec in MODULE_LIBRARY:
        if spec.module_label == module_label:
            return spec
    return None
