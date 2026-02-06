"""BRB Routing Package - 软路由模块"""
from .soft_router import (
    SoftModuleRouter,
    get_soft_router,
    soft_route_modules,
    load_coupling_matrix,
    load_module_taxonomy,
)

__all__ = [
    "SoftModuleRouter",
    "get_soft_router",
    "soft_route_modules",
    "load_coupling_matrix",
    "load_module_taxonomy",
]
