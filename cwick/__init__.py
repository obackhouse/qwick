"""Wick solver for the derivation of equations in quantum chemistry.
"""

__version__ = "0.9.0"

import sys


# Import the shared objects:

from .build import _index as index
from .build import _operator as operator
from .build import _expression as expression
from .build import _wick as wick


# Register the operators - in the future these should just be
# subclasses on the C++ side:

class Projector(operator.Operator):
    def __init__(self):
        super().__init__()
operator.Projector = Projector

class FOperator(operator.Operator):
    def __init__(self, idx, ca):
        super().__init__(idx, ca, True)
operator.FOperator = FOperator

class BOperator(operator.Operator):
    def __init__(self, idx, ca):
        super().__init__(idx, ca, False)
operator.BOperator = BOperator


# Register the modules:

sys.modules["cwick.index"] = index
sys.modules["cwick.operator"] = operator
sys.modules["cwick.expression"] = expression
sys.modules["cwick.wick"] = wick
