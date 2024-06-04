import numpy as np
import logging
import matplotlib.pyplot as plt

from matplotlib.testing.decorators import check_figures_equal

from flowpy2.coords import (CoordStruct,
                            logger)
from flowpy2.flow_type import register_flow_type

import pytest
from flowpy2.io import netcdf
from test_hdf5 import test_filename


@pytest.fixture()
def test_data():
    data = [np.arange(100, dtype='f8'),
            np.arange(50, dtype='f8'),
            np.arange(200, dtype='f8')]
    return data


@pytest.fixture()
def test_index():
    return ['x', 'y', 'z']


@pytest.fixture()
def test_dstruct():
    data = [np.arange(100, dtype='f8'),
            np.arange(50, dtype='f8'),
            np.arange(200, dtype='f8')]

    index = ['x', 'y', 'z']
    return CoordStruct("Cartesian",
                       data,
                       index=index)


def test_ascending(test_data, test_index):
    data = test_data.copy()

    cs = CoordStruct("Cartesian",
                     test_data,
                     index=test_index)

    assert cs.is_consecutive

    data[0] = test_data[0][::-1]
    cs = CoordStruct("Cartesian",
                     data,
                     index=test_index)

    assert cs.is_consecutive

    n = test_data[0].size // 2
    d1 = np.concatenate([test_data[0][:n],
                         test_data[0][n:][::-1]],
                        axis=0)
    data[0][::2] = 2
    data[0][1::2] = -2

    cs = CoordStruct("Cartesian",
                     data,
                     index=test_index)

    assert not cs.is_consecutive


def test_integer_index(test_dstruct):
    ind = test_dstruct.coord_index('x', 20)
    assert ind == 20

    ind = test_dstruct.coord_index('x', 20.499)
    assert ind == 20

    ind = test_dstruct.coord_index('x', 19.501)
    assert ind == 20

    ind = test_dstruct.coord_index('x', -0.499)
    assert ind == 0

    ind = test_dstruct.coord_index('x', 99.499)
    assert ind == 99

    with pytest.raises(ValueError):
        test_dstruct.coord_index('x', 99.501)

    with pytest.raises(ValueError):
        test_dstruct.coord_index('x', -0.501)

    with pytest.raises(KeyError):
        test_dstruct.coord_index('t', 50)

    with pytest.raises(TypeError):
        test_dstruct.coord_index('t', 'f')


def test_list(test_dstruct: CoordStruct):
    comp_data = [0, 50, 70]
    data = [-0.4, 50, 70]
    ind = test_dstruct.coord_index('x', data)
    assert ind == comp_data

    ind = test_dstruct.coord_index('x', tuple(data))
    assert ind == comp_data

    ind = test_dstruct.coord_index('x', np.array(data))
    assert ind == comp_data

    data[-1] = -1
    with pytest.raises(ValueError):
        test_dstruct.coord_index('x', data)


def test_getitem(test_dstruct: CoordStruct):
    array = list(test_dstruct._data)
    index = list(test_dstruct._index)
    a = test_dstruct['x']
    flow_type = test_dstruct.flow_type.name
    assert np.array_equal(a, array[0]), "check integer index"

    sub_ds = CoordStruct(flow_type, array[:2], index=index[:2])
    sub_ds2 = CoordStruct(flow_type, array[1:], index=index[1:])
    sub_ds3 = CoordStruct(flow_type, [array[0]], index=[index[0]])

    assert np.array_equal(
        sub_ds, test_dstruct['x':'y']), "check slice index with start and stop"
    assert np.array_equal(
        sub_ds, test_dstruct[:'y']), "check slice index with stop"
    assert np.array_equal(
        sub_ds2, test_dstruct['y':]), "check slice index with start"
    assert np.array_equal(
        sub_ds3, test_dstruct['x':'x']), "check slice index with start"

    with pytest.raises(ValueError):
        test_dstruct['x':'y': 1]

    with pytest.raises(KeyError):
        test_dstruct['a']
    with pytest.raises(KeyError):
        test_dstruct['x':'a']
    with pytest.raises(KeyError):
        test_dstruct['a':'x']

    assert np.array_equal(
        sub_ds, test_dstruct[['x', 'y']]), "check slice index"
    assert np.array_equal(
        sub_ds3, test_dstruct[['x']]), "check slice index with start"

    with pytest.raises(KeyError):
        test_dstruct[['a']]

    with pytest.raises(KeyError):
        test_dstruct[['x', 'a']]

    with pytest.raises(KeyError):
        test_dstruct[('x', 'a')]


def test_slice(test_dstruct):
    slicer1 = slice(None, None)
    ind = test_dstruct.coord_index('x', slicer1)
    assert ind == slice(0, 100)

    slicer2 = slice(None, 50.501)
    ind = test_dstruct.coord_index('x', slicer2)
    assert ind == slice(0, 52)

    slicer3 = slice(49.501, None)
    ind = test_dstruct.coord_index('x', slicer3)
    assert ind == slice(50, 100)

    slicer4 = slice(50.1, 70)
    ind = test_dstruct.coord_index('x', slicer4)
    assert ind == slice(50, 71)

    slicer5 = slice(50.1, 70, 1)
    with pytest.raises(NotImplementedError):
        test_dstruct.coord_index('x', slicer5)


def test_copy(test_dstruct):
    copy_test = test_dstruct.copy()

    assert copy_test.flow_type is copy_test.flow_type, "Check flow type"
    assert copy_test._data is not test_dstruct._data, "Check it has actually copied"
    for d1, d2 in zip(test_dstruct._data, copy_test._data):
        assert np.array_equal(d1, d2)


def test_hdf(test_dstruct, test_filename):

    test_dstruct.to_hdf(test_filename, 'w')

    dstruct2 = test_dstruct.__class__.from_hdf(test_filename)

    assert test_dstruct == dstruct2


def test_netcdf(test_dstruct, test_filename):
    f = netcdf.make_dataset(test_filename, 'w')
    test_dstruct.to_netcdf(f)
    f.close()

    dstruct2 = test_dstruct.__class__.from_netcdf(test_filename)

    assert test_dstruct == dstruct2


def test_translate(test_dstruct: CoordStruct):
    test_dstruct.Translate(x=100)
    assert np.array_equal(test_dstruct['x'], np.arange(100, 200, dtype='f8'),)


@check_figures_equal()
def test_line_plots(fig_test, fig_ref, test_dstruct):

    coords = np.arange(100, dtype='f8')

    ax1 = fig_ref.subplots()
    ax1.plot(coords, coords)

    ax2 = fig_test.subplots()
    test_dstruct.plot_line('x', coords, ax=ax2)


@check_figures_equal()
def test_pcolormesh(fig_test, fig_ref, test_dstruct):

    coords1 = np.arange(100, dtype='f8')
    coords2 = np.arange(50, dtype='f8')

    data = np.random.randn(100, 50)

    ax1 = fig_ref.subplots()
    ax1.pcolormesh(coords1, coords2, data.T)

    ax2 = fig_test.subplots()
    test_dstruct.pcolormesh('xy', data, ax=ax2)


@check_figures_equal()
def test_contourf(fig_test, fig_ref, test_dstruct):

    coords1 = np.arange(100, dtype='f8')
    coords2 = np.arange(50, dtype='f8')

    data = np.random.randn(100, 50)

    ax1 = fig_ref.subplots()
    ax1.contourf(coords1, coords2, data.T)

    ax2 = fig_test.subplots()
    test_dstruct.contourf('xy', data, ax=ax2)


@check_figures_equal()
def test_contour(fig_test, fig_ref, test_dstruct):

    coords1 = np.arange(100, dtype='f8')
    coords2 = np.arange(50, dtype='f8')

    data = np.random.randn(100, 50)

    ax1 = fig_ref.subplots()
    ax1.contour(coords1, coords2, data.T)

    ax2 = fig_test.subplots()
    test_dstruct.contour('xy', data, ax=ax2)


@check_figures_equal()
def test_quiver(fig_test, fig_ref, test_dstruct: CoordStruct):

    coords1 = test_dstruct['x']
    coords2 = test_dstruct['y']

    data1 = np.random.randn(coords1.size, coords2.size)
    data2 = np.random.randn(coords1.size, coords2.size)

    ax1 = fig_ref.subplots()
    ax1.quiver(coords1, coords2, data1.T, data2.T)

    ax2 = fig_test.subplots()
    test_dstruct.quiver('xy', data1, data2, ax=ax2,
                        scale=None,
                        scale_units=None)


def test_first_derivative(test_dstruct: CoordStruct):
    array = np.linspace(0, 100, 100)

    test_dstruct.first_derivative('x', array)

    array1 = np.random.randn(10, 100, 20)

    test_dstruct.first_derivative('x', array1, axis=1)


def test_second_derivative(test_dstruct: CoordStruct):
    array = np.linspace(0, 100, 100)

    test_dstruct.second_derivative('x', array)

    array1 = np.random.randn(10, 100, 20)

    test_dstruct.second_derivative('x', array1, axis=1)

def test_integrate(test_dstruct: CoordStruct):
    array = np.linspace(0, 100, 100)

    test_dstruct.integrate('x', array)

    array1 = np.random.randn(10, 100, 20)

    test_dstruct.integrate('x', array1, axis=1)

def test_cumulative_integrate(test_dstruct: CoordStruct):
    array = np.linspace(0, 100, 100)

    test_dstruct.cumulative_integrate('x', array)

    array1 = np.random.randn(10, 100, 20)

    test_dstruct.cumulative_integrate('x', array1, axis=1)


def test_to_vtk(test_dstruct: CoordStruct):
    test_dstruct.to_vtk()
