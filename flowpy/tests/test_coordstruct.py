import numpy as np
import logging
import matplotlib.pyplot as plt

from matplotlib.testing.decorators import check_figures_equal

from flowpy.coords import (CoordStruct,
                           logger)
from flowpy.flow_type import (CartesianFlow,
                              PolarFlow)

import pytest
from flowpy.io import netcdf
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
    return CoordStruct(CartesianFlow,
                       data,
                       index=index)


def test_ascending(test_data, test_index):
    data = test_data.copy()
    data[0] = test_data[0][::-1]

    with pytest.raises(ValueError):
        CoordStruct(CartesianFlow,
                    data,
                    index=test_index)


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
    ax1.pcolormesh(coords2, coords1, data)

    ax2 = fig_test.subplots()
    test_dstruct.pcolormesh('xy', data, ax=ax2)


@check_figures_equal()
def test_contourf(fig_test, fig_ref, test_dstruct):

    coords1 = np.arange(100, dtype='f8')
    coords2 = np.arange(50, dtype='f8')

    data = np.random.randn(100, 50)

    ax1 = fig_ref.subplots()
    ax1.contourf(coords2, coords1, data)

    ax2 = fig_test.subplots()
    test_dstruct.contourf('xy', data, ax=ax2)


@check_figures_equal()
def test_contour(fig_test, fig_ref, test_dstruct):

    coords1 = np.arange(100, dtype='f8')
    coords2 = np.arange(50, dtype='f8')

    data = np.random.randn(100, 50)

    ax1 = fig_ref.subplots()
    ax1.contour(coords2, coords1, data)

    ax2 = fig_test.subplots()
    test_dstruct.contour('xy', data, ax=ax2)


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
