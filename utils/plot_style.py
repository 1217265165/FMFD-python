#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5: Unified plot styling with Chinese font support.

This module provides a single entry point for configuring matplotlib
to display Chinese characters correctly across different platforms.

Usage:
    from utils.plot_style import apply
    apply()  # Call at the start of any plotting script
"""

import platform
import warnings
from pathlib import Path


def get_available_fonts():
    """Get list of available font families."""
    try:
        import matplotlib.font_manager as fm
        return [f.name for f in fm.fontManager.ttflist]
    except Exception:
        return []


def _try_register_noto_cjk():
    """Try to register Noto CJK SC font directly from .ttc file."""
    try:
        import matplotlib.font_manager as fm
        noto_paths = [
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"),
        ]
        for p in noto_paths:
            if p.exists():
                fm.fontManager.addfont(str(p))
                return True
    except Exception:
        pass
    return False


def find_chinese_font() -> str:
    """
    Find an available Chinese font based on the operating system.

    Returns
    -------
    str
        Name of an available Chinese font, or fallback font
    """
    system = platform.system()
    available = get_available_fonts()

    # Platform-specific font preferences
    if system == "Windows":
        preferred = [
            "SimHei",
            "Microsoft YaHei",
            "Microsoft YaHei UI",
            "SimSun",
            "NSimSun",
            "FangSong",
            "KaiTi",
        ]
    elif system == "Darwin":  # macOS
        preferred = [
            "Heiti SC",
            "STHeiti",
            "PingFang SC",
            "Hiragino Sans GB",
            "STSong",
        ]
    else:  # Linux
        preferred = [
            "Noto Sans CJK SC",
            "Noto Sans CJK",
            "Noto Sans CJK JP",  # JP variant also supports Chinese characters
            "Noto Sans CJK TC",
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "Droid Sans Fallback",
            "Source Han Sans CN",
            "Source Han Sans SC",
        ]

    # Find first available preferred font
    for font in preferred:
        if font in available:
            return font

    # Try to register Noto CJK from .ttc file (some systems need explicit registration)
    if _try_register_noto_cjk():
        available = get_available_fonts()
        for font in preferred:
            if font in available:
                return font

    # Fallback - try to find any CJK font
    cjk_keywords = ["CJK", "Chinese", "SC", "SimHei", "YaHei", "WenQuanYi",
                    "Source Han"]
    for font in available:
        for kw in cjk_keywords:
            if kw.lower() in font.lower():
                return font

    # Ultimate fallback
    return "DejaVu Sans"


def apply():
    """
    Apply unified plot styling with Chinese font support.

    This function configures matplotlib rcParams for:
    - Chinese character display
    - Minus sign display
    - Clean plot aesthetics

    Call this at the start of any script that generates plots.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
    except ImportError:
        warnings.warn("matplotlib not installed, skipping plot style configuration")
        return

    font = find_chinese_font()

    # Configure matplotlib
    plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

    # Additional aesthetic settings
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['figure.facecolor'] = 'white'

    # Suppress font warnings
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def set_font():
    """Alias for apply() for backward compatibility."""
    apply()