from .io import hdf5, netcdf
import numpy as np
import flowpy2 as fp2
from .plotting import subplots, promote_axes
import logging
from typing import Sequence, Union, Callable, Mapping, Tuple
from numbers import Number
from matplotlib.axes import Axes
from matplotlib.quiver import Quiver
from .datastruct import DataStruct
from .flow_type import (get_flow_type,
                        FlowType)

from .gradient import (first_derivative,
                       second_derivative)

from .integrate import (integrate,
                        cumulative_integrate)

from pyvista import StructuredGrid
logger = logging.getLogger(__name__)


class CoordStruct(DataStruct):
    def __init__(self, flow_type: str, *args, location=None, **kwargs):

        super().__init__(*args, **kwargs)

        if location is None:
            location = {}
        self._location = dict(location)

        self._flow_type = get_flow_type(flow_type)
        self._flow_type.validate_keys(self.index)
        self._flow_type.validate_keys(self._location.keys())

    @property
    def is_consecutive(self):
        for d in self._data:
            diff = np.diff(d)
            if not all(diff < 0) and not all(diff > 0):
                return False

        return True

    @property
    def location(self):
        return self._location

    def _init_args_from_kwargs(self, **kwargs):
        kwargs = super()._init_args_from_kwargs(**kwargs)
        self._init_update_kwargs(kwargs,
                                 'flow_type',
                                 self._flow_type.name)
        self._init_update_kwargs(kwargs,
                                 'location',
                                 self._location)

        return kwargs

    def _hdf5_write_hook(self, g: hdf5.hdfHandler):
        g.attrs['flow_type'] = self._flow_type.name

        for k, v in self._location.items():
            g.attrs[f'location_{k}'] = v

    @classmethod
    def _hdf5_read_hook(cls, h5_group: hdf5.hdfHandler):
        location = {}
        for attr in h5_group.attrs.keys():
            if attr.startswith('location_'):
                key = attr.removeprefix('location_')
                location[key] = h5_group.attrs[attr]

        return {'flow_type': h5_group.attrs['flow_type'],
                'location': location}

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
        for k, v in self._location.items():
            setattr(group, f'_location_{k}', v)

        group._index = np.array([np.string_(ind) for ind in self.index])
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
        data = {k: g[k][:] for k in g._index}

        location = {}
        for k in g.__dict__.keys():
            if '_location_' in k:
                attr = k.removeprefix('_location_')
                location[attr] = getattr(g, k)

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
    def flow_type(self, value: Union[str, FlowType]):
        if isinstance(value, str):
            value = get_flow_type(value)

        if self._flow_type.has_base_keys:
            old_base_keys = self._flow_type._base_keys

            if not all(val in value._base_keys for val in old_base_keys):
                raise ValueError(f"{type(self).__name__} old base_keys "
                                 "must all be in the new base keys")

        self._flow_type = value

    def create_subdomain(self,
                         drop_coords=True,
                         return_indexer=False, **coord_kw):

        coord_dict = {}
        location = self._location.copy()
        indexer = {}

        for k in self.index:
            if k in coord_kw:
                index = self.coord_index(k, coord_kw[k])
            else:
                index = slice(None)

            val = self.get(k)[index]
            if isinstance(val, Number):
                if drop_coords:
                    location[k] = val
                else:
                    coord_dict[k] = np.array([val])

            else:
                coord_dict[k] = val

            indexer[k] = index

        kwargs = self._init_args_from_kwargs(data=coord_dict,
                                             location=location)
        coords = self._create_struct(**kwargs)

        if return_indexer:
            return coords, indexer
        else:
            return coords

    def rescale(self, key: str, val: Number):
        index = self.index.get(key)
        self._data[index] /= val

    def plot_line(self, comp: str, data: Sequence, ax=None,
                  transform_xdata: Callable = None,
                  fig_kw=None, **kwargs):

        ax = self._update_axes(ax, fig_kw, comp)

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

    def _get_coords_contour(self,
                            plane,
                            transform_xdata: Callable = None,
                            transform_ydata: Callable = None):

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

        ax = self._update_axes(ax, fig_kw, tuple(plane))

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

        ax = self._update_axes(ax, fig_kw, tuple(plane))

        xcoords, ycoords = self._get_coords_contour(plane, transform_xdata,
                                                    transform_ydata)

        x, y, c = self.flow_type.process_data_contour(plane,
                                                      xcoords,
                                                      ycoords,
                                                      data,
                                                      kwargs)
        return ax.contourf(x, y, c.T, **kwargs)

    def _update_axes(self,
                     ax,
                     fig_kw: Mapping,
                     loc: Union[str, Sequence]) -> Axes:

        projection = self.flow_type.projection(loc)
        if ax is None:
            if fig_kw is None:
                fig_kw = {}

            subplots_kw = fig_kw.get('subplot_kw', {})
            subplots_kw['projection'] = projection

            fig_kw['subplot_kw'] = subplots_kw

            _, ax = subplots(**fig_kw)
        else:
            ax = promote_axes(ax, projection=projection)

        return ax

    def pcolormesh(self, plane: Sequence[str], data: np.ndarray, ax=None,
                   transform_xdata=None,
                   transform_ydata=None,
                   fig_kw: Mapping = None, **kwargs):

        ax = self._update_axes(ax, fig_kw, tuple(plane))

        xcoords, ycoords = self._get_coords_contour(plane, transform_xdata,
                                                    transform_ydata)

        x, y, c = self.flow_type.process_data_contour(plane,
                                                      xcoords,
                                                      ycoords,
                                                      data,
                                                      kwargs)
        return ax.pcolormesh(x, y, c.T, **kwargs)

    def quiver(self, plane: Sequence[str],
               U: np.ndarray,
               V: np.ndarray,
               spacing: Tuple[int] = (1, 1),
               ax: Axes = None,
               transform_xdata: Callable = None,
               transform_ydata: Callable = None,
               fig_kw: Mapping = None,
               angles='uv',
               scale_units='spacing',
               scale=1,
               **kwargs) -> Quiver:

        ax = self._update_axes(ax, fig_kw, tuple(plane))

        xcoords, ycoords = self._get_coords_contour(plane,
                                                    transform_xdata,
                                                    transform_ydata)

        xcoords = xcoords[::spacing[0]]
        ycoords = ycoords[::spacing[1]]

        U = U[::spacing[0], ::spacing[1]]
        V = V[::spacing[0], ::spacing[1]]

        if callable(transform_xdata):
            U = transform_xdata(U)

        if callable(transform_ydata):
            V = transform_ydata(V)

        return ax.quiver(xcoords, ycoords, U.T, V.T,
                         scale=scale,
                         angles=angles,
                         scale_units=scale_units,
                         **kwargs)

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
            return [self.coord_index(comp, lc) for lc in loc]

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
        coords = self.get(comp)

        return integrate(data, x=coords, axis=axis, method=method)

    def cumulative_integrate(self, comp, data, initial=0, axis=0, method=None):
        coords = self.get(comp)

        return cumulative_integrate(data,
                                    x=coords,
                                    axis=axis,
                                    method=method,
                                    initial=initial)

    def interpolate(self, other_cstruct, data):
        pass

    def copy(self):
        kwargs = self._init_args_from_kwargs(copy=True)
        return self._create_struct(**kwargs)

    def to_vtk(self, layout=None):
        if layout is None:
            layout = self.index

        if any(lo in layout for lo in self._location.keys()):
            raise ValueError(f"Locations and {type(self)}"
                             " cannot overlap")

        if self.flow_type.has_base_keys:
            base_keys = self.flow_type._base_keys
        else:
            base_keys = ('x', 'y', 'z')

        args = [self.get(lo) if lo in layout
                else np.array([self._location.get(lo, 0)])
                for lo in base_keys]

        grid = dict(zip(['x', 'y', 'z'],
                        np.meshgrid(*args,
                                    indexing='ij')))

        cart_grid = self._flow_type.transform(grid)

        X = cart_grid['x'].astype('f4')
        Y = cart_grid['y'].astype('f4')
        Z = cart_grid['z'].astype('f4')

        return StructuredGrid(X, Y, Z)

    def equals(self, other_cstruct):

        if type(self) is not type(other_cstruct):
            logger.debug("Types do not match: "
                         f"({type(self)}) vs ({type(other_cstruct)})")
            return False

        if self.flow_type != other_cstruct.flow_type:
            logger.debug("flow_type doesn't match: "
                         f"({self.flow_type}) vs ({other_cstruct.flow_type})")
            return False

        if self._location != other_cstruct._location:
            logger.debug("location doesn't match "
                         f"({self._location}) vs ({other_cstruct._location})")
            return False

        return super().equals(other_cstruct)

    def __str__(self) -> str:
        return "%s(%s, index=%s)" % (type(self).__name__,
                                     self._flow_type.name,
                                     list(self.index))


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
