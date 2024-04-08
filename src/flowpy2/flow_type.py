from __future__ import annotations
from .io import hdf5
import numpy as np
import logging
import copy
from typing import Tuple, Callable, Mapping, Union
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
    def __init__(self,
                 name: str,
                 keys: Tuple[str],
                 projection: str = None,
                 symbols: dict = None,
                 transform_cartesian: Callable = None,
                 plot_line_processor: Callable=None,
                 contour_processor: Callable=None):

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

        self._plot_line_process = self._validate_line_data(plot_line_processor)
        self._contour_process = self._validate_line_data(contour_processor)

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

    def _validate_line_data(self,func: Union[Callable,None]):
        if func is None: return None

        try:
            func('y',np.arange(50),np.arange(50),{})
        except Exception as e:
            raise Exception("Line plot processor validator "
                            f"failed with exception:\n\n {e}") from None
        
        return func
    
    def _validate_contour_data(self,func: Union[Callable,None]):
        if func is None: return None

        try:
            x = np.arange(50)
            y = np.arange(100)
            c = np.arange(5000).reshape(50,100)
            func('y',x ,y , c,{})

        except Exception as e:
            raise Exception("Contour processor validator "
                            f"failed with exception:\n\n {e}") from None
        
        return func


    def transform(self, input_grid):
        if self._transform_cartesian is None:
            return input_grid
        else:
            return self._transform_cartesian(input_grid)

    @property
    def symbols(self):
        return self._symbols

    @property
    def name(self)->str:
        return self._name

    @property
    def is_time_type(self)->bool:
        return 't' in self._base_keys

    def get_time_type(self)->FlowType:
        if self.is_time_type:
            return self
        else:
            return get_flow_type(self._name+' (time)')
            
    def Transform(self):
        return self._transform_cartesian

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

    # def to_hdf(self,  fn_or_obj, mode=None, key=None):
    #     g = hdf5.make_group(fn_or_obj, mode, key)
    #     hdf5.set_type_tag(type(self), g)

    #     g.attrs['name'] = self._name
    #     if self._projection is not None:
    #         g.attrs['projection'] = self._projection

    #     for k, v in self._symbols.items():
    #         g.attrs[k] = v

    #     g.create_dataset("base_keys", data=np.array(
    #         self._base_keys, dtype=np.string_))

    # def to_netcdf(self, g):
    #     if self._projection is not None:
    #         g.projection = np.string_(self._projection)

    #     g.flowtype_name = np.string_(self._name)
    #     g.base_keys = self._base_keys

    #     for k, v in self._symbols.items():
    #         attr_name = "flowtype_attr_%s" % k
    #         setattr(g, attr_name, v)

    # @classmethod
    # def from_netcdf(cls, g):
    #     if hasattr(g, 'projection'):
    #         projection = str(g.projection)
    #     else:
    #         projection = None

    #     name = str(g.flowtype_name)
    #     base_keys = tuple(g.base_keys)

    #     attrs = [key for key in g.__dict__.keys()
    #              if 'flowtype_attr_' in key]
    #     symbols = {}

    #     for attr in attrs:
    #         key = attr.split('flowtype_attr_')
    #         symbols[key] = getattr(g, attr)

    #     return cls(name,
    #                tuple(base_keys),
    #                projection=projection,
    #                symbols=attrs)

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


    def subplots(self, *args, **kwargs):
        subplots_kw = kwargs.get('subplot_kw', {})
        subplots_kw['projection'] = self.projection

        kwargs['subplot_kw'] = subplots_kw

        return subplots(*args, **kwargs)

    def __deepcopy__(self,memo):
        return self.__class__(self._name,
                              self._base_keys,
                              self._projection,
                              self._symbols,
                              self._transform_cartesian,
                              self._plot_line_process,
                              self._contour_process)

    def process_data_line(self,
                          line: str,
                          x: np.ndarray,
                          y: np.ndarray,
                          kwargs):
        
        if self._plot_line_process is None:
            return x, y
        
        return self._plot_line_process(line, x, y, kwargs)
    
    def process_data_contour(self,
                             plane: str,
                             x: np.ndarray,
                             y: np.ndarray,
                             c: np.ndarray,
                             kwargs):
        if self._contour_process is None:
            return x, y, c
        
        return self._contour_process(plane, x, y, c, kwargs)

    

_flow_types = {}


def register_flow_type(flow_type: FlowType):
    if flow_type.name in _flow_types:
        raise KeyError(f"Name {flow_type.name} is already a flow_type")

    _flow_types[flow_type.name] = flow_type

    # add type for time as dimension

    flow_type_time = copy.deepcopy(flow_type)
    flow_type_time._base_keys += ('t',)
    flow_type_time._name += " (time)"

    _flow_types[flow_type_time.name] = flow_type_time


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