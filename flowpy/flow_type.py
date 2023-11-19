from abc import ABC, abstractproperty
from typing import Tuple
from matplotlib.projections import get_projection_class
_symbols = {'CHANNEL': 0,
            'BLAYER': 1}


# coordinate transformations
# Store information on matplotlib axes projections
# Quantities for computing gradients ->
# laplacians, divergences etc

class FlowType:
    def __init__(self, keys: Tuple[str], projection: str = None, symbols: dict = None):
        self._projection = projection
        if not all(isinstance(key, str) for key in keys):
            raise TypeError("All keys must be of type str")

        if len(keys) != len(set(keys)):
            raise ValueError("Keys cannot be repeated")

        self._base_keys = tuple(keys)

        self._symbols = {}
        if symbols is not None:
            self._symbols.update(symbols)

    @property
    def symbols(self):
        return self._symbols

    @abstractproperty
    def Transform(self):
        pass

    def set_mpl_projection(self, projection):
        get_projection_class(projection)

        self._projection = projection

    @property
    def projection(self):
        return self._projection

    def validate_keys(self, keys):
        if not all(key in self._base_keys for key in keys):
            raise ValueError("Invalid key for flow "
                             f"type {self.__class__.__name__}")


CartesianFlow = FlowType(('x', 'y', 'z'))
PolarFlow = FlowType(('z', 'r', 'theta'), projection='polar')
