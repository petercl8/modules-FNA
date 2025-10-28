from . import test_funcs

from .test_funcs import *

__all__ = []
for module in [test_funcs]:
    __all__.extend(getattr(module, "__all__", []))