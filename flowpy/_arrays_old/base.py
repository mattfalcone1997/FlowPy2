from abc import ABC
from numpy.lib.mixins import NDArrayOperatorsMixin
import numbers


class ArrayBase(NDArrayOperatorsMixin, ABC):
    _array_class = None
    _array_creator = None

    _HANDLED_TYPES = (numbers.Number,)
    _ALLOWED_METHODS = ('__call__')
    _NOT_ALLOWED_UFUNCS = ()
    _NOT_ALLOWED_KWARGS = ('axis', 'out', 'axes')
