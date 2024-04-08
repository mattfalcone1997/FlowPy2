from .io import hdf5, netcdf
import numpy as np
import flowpy2 as fp2
from .plotting import subplots
import logging
from typing import Sequence, Union, Callable, Mapping
from numbers import Number

from .datastruct import DataStruct
from .flow_type import (get_flow_type,
                        FlowType)
from .gradient import (first_derivative,
                       second_derivative)
from .indexers import CompIndexer

from pyvista import StructuredGrid
logger = logging.getLogger(__name__)


class CoordStruct(DataStruct):
    def __init__(self, flow_type: str, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._flow_type = get_flow_type(flow_type)
        self._flow_type.validate_keys(self.index)
        self._validate_coords()

    def _validate_coords(self):
        for d in self._data:
            diff = np.diff(d)
            if any(diff < 0):
                raise ValueError("Coordinates must be in ascending order")

    def _init_args_from_kwargs(self, **kwargs):
        kwargs = super()._init_args_from_kwargs(**kwargs)
        self._init_update_kwargs(kwargs,
                                 'flow_type',
                                 self._flow_type.name)
        return kwargs

    def _hdf5_write_hook(self, g: hdf5.H5_Group_File):
        g.attrs['flow_type'] = self._flow_type.name
    
    @classmethod
    def _hdf5_read_hook(cls,h5_group: hdf5.H5_Group_File):
        return {'flow_type' : h5_group.attrs['flow_type']}

    def Translate(self, **kwargs):
        for k in kwargs:
            if k not in self.index:
                raise ValueError(f"{k} not in {self.__class__.__name__}")
            if not isinstance(kwargs[k], Number):
                raise TypeError("Invalid type")

            self[k] += kwargs[k]

    def to_netcdf(self, group):
        if not netcdf.HAVE_NETCDF4:
            raise ModuleNotFoundError("netCDF cannot be used")

        netcdf.set_type_tag(type(self), group, "coords_tag")

        group.flow_type = self._flow_type.name
        for key in self.index:
            data = self.get(key)
            group.createDimension(key, data.size)
            dtype = data.dtype.kind + str(data.dtype.itemsize)
            var = group.createVariable(key, dtype, (key,))
            var[:] = data

    @classmethod
    def from_netcdf(cls, group, key=None, tag_check=None):
        if not netcdf.HAVE_NETCDF4:
            raise ModuleNotFoundError("netCDF cannot be used")

        g = netcdf.access_dataset(group, key)
        if tag_check is None:
            tag_check = fp2.rcParams['io.tag_check']

        real_cls = netcdf.validate_tag(cls, g, tag_check, "coords_tag")

        flow_type = get_flow_type(g.flow_type)
        data = {k: g[k][:] for k in flow_type._base_keys}

        return real_cls._create_struct(flow_type=flow_type.name,
                                       data=data)

    def _validate_inputs(self, inputs):
        super()._validate_inputs(inputs)
        for x in inputs:
            if isinstance(x, self.__class__):
                if self.flow_type is not x.flow_type:
                    raise ValueError("Flow type should be the same")

    @property
    def flow_type(self):
        return self._flow_type

    @flow_type.setter
    def flow_type(self,value: Union[str,FlowType]):
        if isinstance(value, str):
            value = get_flow_type(value)
        
        old_base_keys = self._flow_type._base_keys
        if not all(val in value._base_keys for val in old_base_keys):
            raise ValueError("FlowType old base_keys must all "
                             "be in the new base keys")
        self._flow_type = value

    def rescale(self, key: str, val: Number):
        self._data[key] /= val

    def plot_line(self, comp: str, data: Sequence, ax=None,
                  transform_xdata: Callable = None,
                  fig_kw=None, **kwargs):

        if ax is None:
            if fig_kw is None:
                fig_kw = {}

            fig, ax = self._flow_type.subplots(**fig_kw)

        coords = self.get(comp)

        coords, data = self.flow_type.process_data_line(comp,
                                                  coords,
                                                  data,
                                                  kwargs)

        if transform_xdata is not None:
            if not callable(transform_xdata):
                raise TypeError("transform_xdata must be callable")
            coords = transform_xdata(coords)

        if len(data) != len(coords):
            raise ValueError("Coordinate and data have different "
                             f"shapes ({len(data)}) vs ({len(coords)}).")

        ax.plot(coords, data, **kwargs)

        return ax

    def _get_coords_contour(self, plane, transform_xdata: Callable = None, transform_ydata: Callable = None):
        if len(plane) != 2:
            raise ValueError("Length of plane must be 2")

        xcoords = self.get(plane[0])
        ycoords = self.get(plane[1])

        if transform_xdata is not None:
            if not callable(transform_xdata):
                raise TypeError("transform_xdata must be callable")
            xcoords = transform_xdata(xcoords)

        if transform_ydata is not None:
            if not callable(transform_ydata):
                raise TypeError("transform_ydata must be callable")
            ycoords = transform_ydata(ycoords)

        return xcoords, ycoords

    def contour(self, plane: Sequence[str], data: np.ndarray, ax=None,
                transform_xdata=None,
                transform_ydata=None,
                fig_kw=None, **kwargs):

        if ax is None:
            if fig_kw is None:
                fig_kw = {}
            fig, ax = self._flow_type.subplots(**fig_kw)

        xcoords, ycoords = self._get_coords_contour(plane,
                                                    transform_xdata,
                                                    transform_ydata)

        x, y, c = self.flow_type.process_data_contour(plane,
                                                      xcoords,
                                                      ycoords,
                                                      data,
                                                      kwargs)
                                                      
        return ax.contour(x, y, c.T, **kwargs)

    def contourf(self, plane: Sequence[str], data: np.ndarray, ax=None,
                 transform_xdata=None,
                 transform_ydata=None,
                 fig_kw=None, **kwargs):

        if ax is None:
            if fig_kw is None:
                fig_kw = {}
            fig, ax = self._flow_type.subplots(**fig_kw)

        xcoords, ycoords = self._get_coords_contour(plane, transform_xdata,
                                                    transform_ydata)

        x, y, c = self.flow_type.process_data_contour(plane,
                                                      xcoords,
                                                      ycoords,
                                                      data,
                                                      kwargs)
        return ax.contourf(x, y, c.T, **kwargs)

    def pcolormesh(self, plane: Sequence[str], data: np.ndarray, ax=None,
                   transform_xdata=None,
                   transform_ydata=None,
                   fig_kw: Mapping = None, **kwargs):

        if ax is None:
            if fig_kw is None:
                fig_kw = {}
            fig, ax = self._flow_type.subplots(**fig_kw)

        xcoords, ycoords = self._get_coords_contour(plane, transform_xdata,
                                                    transform_ydata)
        
        x, y, c = self.flow_type.process_data_contour(plane,
                                                      xcoords,
                                                      ycoords,
                                                      data,
                                                      kwargs)
        return ax.pcolormesh(x, y, c.T, **kwargs)

    def coord_index(self, comp: str, loc: Union[Number, list, slice]):
        if isinstance(loc, Number):
            logger.debug("Input loc is a number")
            coords = self.get(comp)

            max_coord = np.amax(coords)
            if loc > max_coord:
                if loc < max_coord + 0.5*(coords[-1] - coords[-2]):
                    return coords.size - 1
                raise ValueError("loc outside of bounds")

            min_coord = np.amin(coords)
            if loc < min_coord:
                if loc > min_coord - 0.5*(coords[1] - coords[0]):
                    return 0
                raise ValueError("loc outside of bounds")

            return np.argmin(np.abs(coords - loc))

        elif isinstance(loc, (list, tuple, np.ndarray)):
            logger.debug("Input loc is a is list, tuple or array")
            return [self.coord_index(comp, l) for l in loc]

        elif isinstance(loc, slice):
            logger.debug("Input loc is a slice")

            if loc.step is not None:
                raise NotImplementedError("slice step not allowed for now")

            if loc.stop is None:
                stop = self.get(comp).size
            else:
                stop = self.coord_index(comp, loc.stop) + 1

            if loc.start is None:
                start = 0
            else:
                start = self.coord_index(comp, loc.start)

            if start >= stop:
                raise ValueError("slice stop must be greater than start")

            return slice(start, stop)

        else:
            raise TypeError("CoordStruct cannot be indexed with "
                            f"type {type(loc).__name__}")

    def first_derivative(self, comp, data, axis=0, method=None):
        coords = self.get(comp)
        return first_derivative(data, coords, axis=axis, method=method)

    def second_derivative(self, comp, data, axis=0, method=None):
        coords = self.get(comp)

        return second_derivative(data, coords, axis=axis, method=method)

    def integrate(self, comp, data, axis=0, method=None):
        pass

    def cumulative_integrate(self, comp, data, axis=0, method=None):
        pass

    def interpolate(self, other_cstruct, data):
        pass

    def copy(self):
        return self.__class__(self._flow_type.name, self.to_dict(), copy=True)

    def to_vtk(self):
        cart_grid = self._flow_type.transform(self)

        x = cart_grid['x']
        y = cart_grid['y']
        z = cart_grid['z']

        shape = (x.size, y.size, z.size)
        X = np.ones(shape)*x[:, None, None]
        Y = np.ones(shape)*y[None, :, None]
        Z = np.ones(shape)*z[None, None, :]

        return StructuredGrid(X, Y, Z)


@CoordStruct.implements(np.allclose)
def allclose(dstruct1: CoordStruct, dstruct2: CoordStruct, *args, **kwargs):
    if dstruct1.index != dstruct2.index:
        logger.debug("index doesn't not match")
        return False

    if dstruct1.flow_type != dstruct2.flow_type:
        logger.debug("flow_type doesn't not match")
        return False

    for d1, d2 in zip(dstruct1._data, dstruct2._data):
        if not np.allclose(d1, d2, *args, **kwargs):
            logger.debug("data do not match")
            return False

    return True


@CoordStruct.implements(np.array_equal)
def array_equal(dstruct1: CoordStruct, dstruct2: CoordStruct, *args, **kwargs):
    if dstruct1.index != dstruct2.index:
        return False

    if dstruct1.flow_type != dstruct2.flow_type:
        return False

    for d1, d2 in zip(dstruct1._data, dstruct2._data):
        if not np.array_equal(d1, d2, *args, **kwargs):
            return False

    return True
