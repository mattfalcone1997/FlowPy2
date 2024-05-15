from __future__ import annotations
from .io import hdf5
import numpy as np
import logging
import copy
from typing import Tuple, Callable, Mapping, Union
from matplotlib.projections import get_projection_class

logger = logging.getLogger(__name__)


# coordinate transformations
# Store information on matplotlib axes projections
# Quantities for computing gradients ->
# laplacians, divergences etc

class FlowType:
    def __init__(self,
                 name: str,
                 keys: Tuple[str] = None,
                 projection: Callable = None,
                 symbols: dict = None,
                 transform_cartesian: Callable = None,
                 plot_line_processor: Callable = None,
                 contour_processor: Callable = None):

        self._name = name

        self._base_keys = self._verify_keys(keys)
            

        self._projection = self._validate_projection(projection)

        self._symbols = {}
        if symbols is not None:
            self._symbols.update(symbols)

        self._transform_cartesian = self._validate_transform(
            transform_cartesian)

        self._plot_line_process = self._validate_line_data(plot_line_processor)
        self._contour_process = self._validate_line_data(contour_processor)

    def _verify_keys(self,keys):
        if keys is not None:
            if not all(isinstance(key, str) for key in keys):
                raise TypeError("All keys must be of type str")

            if len(keys) != len(set(keys)):
                raise ValueError("Keys cannot be repeated")

            return tuple(keys)
        else:
            return None

    def projection(self, loc):
        if isinstance(self._projection, str) or self._projection is None:
            return self._projection
        else:
            return self._projection(loc)

    @property
    def symbols(self):
        return self._symbols

    @property
    def name(self) -> str:
        return self._name

    @property
    def has_base_keys(self):
        return self._base_keys is not None

    def _validate_projection(self, func: Union[Callable, str]) -> Callable:

        if isinstance(func, str) or func is None:
            return func

        if not self.has_base_keys:
            raise ValueError("projection must be None or "
                             "str is not base keys")

        try:
            for key in self._base_keys:
                func(key)

            func(self._base_keys)
        except Exception:
            raise ValueError("Projection validation error") from None

        else:
            return func

    def _validate_transform(self, transform_cartesian: Callable):

        if transform_cartesian is not None:
            if not self.has_base_keys:
                raise ValueError("Transform cannot be provided to "
                                 f"{type(self).__name__} without base keys")

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

    def _validate_line_data(self, func: Union[Callable, None]):
        if func is None:
            return None

        try:
            func('y', np.arange(50), np.arange(50), {})
        except Exception as e:
            raise Exception("Line plot processor validator "
                            f"failed with exception:\n\n {e}") from None

        return func

    def _validate_contour_data(self, func: Union[Callable, None]):
        if func is None:
            return None

        try:
            x = np.arange(50)
            y = np.arange(100)
            c = np.arange(5000).reshape(50, 100)
            func('y', x, y, c, {})

        except Exception as e:
            raise Exception("Contour processor validator "
                            f"failed with exception:\n\n {e}") from None

        return func

    def transform(self, input_grid):
        if self._transform_cartesian is None:
            return input_grid
        else:
            return self._transform_cartesian(input_grid)

    def set_mpl_projection(self, projection: str):
        get_projection_class(projection)

        self._projection = projection

    @property
    def is_time_type(self) -> bool:
        return 't' in self._base_keys

    def get_time_type(self) -> FlowType:
        if self.is_time_type:
            return self
        else:
            return get_flow_type(self._name+' (time)')

    def Transform(self):
        return self._transform_cartesian

    def validate_keys(self, keys):
        if not self.has_base_keys:
            return

        if not all(key in self._base_keys for key in keys):
            raise ValueError("Invalid key for flow "
                             f"type {self.__class__.__name__}")

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

    def __deepcopy__(self, memo):
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
    if flow_type.has_base_keys:
        flow_type_time = copy.deepcopy(flow_type)
        flow_type_time._base_keys += ('t',)
        flow_type_time._name += " (time)"

        _flow_types[flow_type_time.name] = flow_type_time


def get_flow_type(name) -> FlowType:
    if name not in _flow_types:
        raise ValueError("Invalid flow type name. Available "
                         f"flow types: {list(_flow_types.keys())}")

    return _flow_types[name]


register_flow_type(FlowType("Base",
                            projection='FlowAxes'))

register_flow_type(FlowType("Cartesian",
                            ('x', 'y', 'z'),
                            projection='FlowAxes'))


def _polar_to_cartesian(polar_data: Mapping):
    x = polar_data['z']
    y = polar_data['r']*np.sin(polar_data['theta'])
    z = polar_data['r']*np.cos(polar_data['theta'])

    return {'x': x, 'y': y, 'z': z}


def _polar_projection(proj):
    if hasattr(proj, '__iter__'):
        if 'theta' in proj and 'r' in proj:
            return 'polar'

    return 'FlowAxes'


register_flow_type(FlowType("Polar",
                            ('z', 'r', 'theta'),
                            projection=_polar_projection,
                            transform_cartesian=_polar_to_cartesian))
