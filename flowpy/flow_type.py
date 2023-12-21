from .io import hdf5
import numpy as np
import logging
from abc import ABC, abstractproperty
from typing import Tuple
from matplotlib.projections import get_projection_class
_symbols = {'CHANNEL': 0,
            'BLAYER': 1}

logger = logging.getLogger(__name__)


# coordinate transformations
# Store information on matplotlib axes projections
# Quantities for computing gradients ->
# laplacians, divergences etc

class FlowType:
    def __init__(self, name: str, keys: Tuple[str], projection: str = None, symbols: dict = None):
        self._projection = projection
        if not all(isinstance(key, str) for key in keys):
            raise TypeError("All keys must be of type str")

        if len(keys) != len(set(keys)):
            raise ValueError("Keys cannot be repeated")

        self._name = name
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

    def to_hdf(self,  fn_or_obj, mode=None, key=None):
        g = hdf5.make_group(fn_or_obj, mode, key)
        hdf5.set_type_tag(type(self), g)

        g.attrs['name'] = self._name
        if self._projection is not None:
            g.attrs['projection'] = self._projection

        for k, v in self._symbols.items():
            g.attrs[k] = v

        g.create_dataset("base_keys", data=np.array(
            self._base_keys, dtype=np.string_))

    def to_netcdf(self, g):
        if self._projection is not None:
            g.projection = np.string_(self._projection)

        g.flowtype_name = np.string_(self._name)
        g.base_keys = self._base_keys

        for k, v in self._symbols.items():
            attr_name = "flowtype_attr_%s" % k
            setattr(g, attr_name, v)

    @classmethod
    def from_netcdf(cls, g):
        if hasattr(g, 'projection'):
            projection = str(g.projection)
        else:
            projection = None

        name = str(g.flowtype_name)
        base_keys = tuple(g.base_keys)

        attrs = [key for key in g.__dict__.keys()
                 if 'flowtype_attr_' in key]
        symbols = {}

        for attr in attrs:
            key = attr.split('flowtype_attr_')
            symbols[key] = getattr(g, attr)

        return cls(name,
                   tuple(base_keys),
                   projection=projection,
                   symbols=attrs)

    def __eq__(self, value: object) -> bool:
        if type(self) != type(value):
            logger.debug("Types do not match")
            return False

        if self._name != value._name:
            logger.debug("Names do not match: "
                         f"{self._name} vs {value._name}")
            return False

        if self._projection != value._projection:
            logger.debug("Projections do not "
                         f"match: {self._projection} vs {value._projection}")
            return False

        if self._base_keys != value._base_keys:
            logger.debug("Base keys do not match "
                         f"{self._base_keys} vs {value._base_keys}")
            return False

        return True

    @classmethod
    def from_hdf(cls,  fn_or_obj, key):
        g = hdf5.access_group(fn_or_obj, key)
        hdf5.validate_tag(cls, g, 'strict')

        attrs = dict(g.attrs)
        name = attrs.pop('name')
        projection = attrs.pop('projection', None)

        return cls(name,
                   tuple(key.decode('utf-8') for key in g["base_keys"][:]),
                   projection=projection,
                   symbols=attrs)


CartesianFlow = FlowType("Cartesian", ('x', 'y', 'z'))
PolarFlow = FlowType("Polar", ('z', 'r', 'theta'), projection='polar')
