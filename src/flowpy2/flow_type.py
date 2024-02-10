from .io import hdf5
import numpy as np
import logging
from abc import ABC, abstractproperty
from typing import Tuple, Callable, Mapping
from matplotlib.projections import get_projection_class
from .plotting import subplots

_symbols = {'CHANNEL': 0,
            'BLAYER': 1}

logger = logging.getLogger(__name__)


# coordinate transformations
# Store information on matplotlib axes projections
# Quantities for computing gradients ->
# laplacians, divergences etc

class FlowType:
    def __init__(self, name: str, keys: Tuple[str],
                 projection: str = None,
                 symbols: dict = None,
                 transform_cartesian: Callable = None):

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

        self._transform_cartesian = self._validate_transform(
            transform_cartesian)

    def _validate_transform(self, transform_cartesian: Callable):

        if transform_cartesian is not None:
            if not callable(transform_cartesian):
                raise TypeError("transform_cartesian must be callable or None")

        else:
            return None

        input_data = {key: np.zeros(100) for key in self._base_keys}

        try:
            out_data = transform_cartesian(input_data)
        except Exception as e:

            raise ValueError("transform_cartesian produced "
                             f"error: {e.args[0]}") from None

        if not isinstance(out_data, Mapping):
            raise TypeError("transform_cartesian must return a Mapping")

        if not all(k in ['x', 'y', 'z'] for k in out_data.keys()):
            raise ValueError("Keys in ouput not in base keys")

        return transform_cartesian

    def transform(self, input_grid):
        if self._transform_cartesian is None:
            return input_grid
        else:
            return self._transform_cartesian(input_grid)

    @property
    def symbols(self):
        return self._symbols

    @property
    def name(self):
        return self._name

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

    def subplots(self, *args, **kwargs):
        subplots_kw = kwargs.get('subplot_kw', {})
        subplots_kw['projection'] = self.projection

        kwargs['subplot_kw'] = subplots_kw

        return subplots(*args, **kwargs)


_flow_types = {}


def register_flow_type(flow_type: FlowType):
    if flow_type.name in _flow_types:
        raise KeyError(f"Name {flow_type.name} is already a flow_type")

    _flow_types[flow_type.name] = flow_type


def get_flow_type(name) -> FlowType:
    if name not in _flow_types:
        raise ValueError("Invalid flow type name. Available "
                         f"flow types: {list(_flow_types.keys())}")

    return _flow_types[name]


register_flow_type(FlowType("Cartesian",
                            ('x', 'y', 'z'),
                            projection='FlowAxes'))


def _polar_to_cartesian(polar_data: Mapping):
    x = polar_data['z']
    y = polar_data['r']*np.sin(polar_data['theta'])
    z = polar_data['r']*np.cos(polar_data['theta'])

    return {'x': x, 'y': y, 'z': z}


register_flow_type(FlowType("Polar",
                            ('z', 'r', 'theta'),
                            projection='polar',
                            transform_cartesian=_polar_to_cartesian))
