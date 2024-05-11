from flowpy2.flowstruct import FlowStructND
from flowpy2.coords import CoordStruct
from flowpy2.flow_type import get_flow_type
from matplotlib.testing.decorators import check_figures_equal

import pytest
from math import prod
import numpy as np

from test_hdf5 import test_filename


@pytest.fixture
def reference_fstruct():
    data = {'x': np.linspace(0, 100, 200),
            'y': np.linspace(0, 2, 50),
            'z': np.linspace(0, 6, 100)}
    coords = CoordStruct("Cartesian",
                         data)
    times = [100, 200, 300]
    comps = ['u', 'v', 'w']
    size = prod([len(times), len(comps), *[d.size for d in data.values()]])
    array_flat = np.arange(1, size+1, dtype='f8')
    array = array_flat.reshape(
        (len(times), len(comps), *[d.size for d in data.values()]))
    return FlowStructND(coords, array, comps=comps, times=times, data_layout='xyz')


@pytest.fixture
def reference_coords():
    data = {'x': np.linspace(0, 100, 200),
            'y': np.linspace(0, 2, 50),
            'z': np.linspace(0, 6, 100)}
    return CoordStruct("Cartesian",
                       data)


def test_valid_init(reference_fstruct):

    times = reference_fstruct.times
    comps = reference_fstruct.comps
    coords = reference_fstruct.coords
    array = reference_fstruct._array.squeeze()

    FlowStructND(coords, array[0, ...],
                 comps=comps)
    
    FlowStructND(coords, array[0, ...],
                 comps=comps,
                 data_layout='xyz')

    FlowStructND(coords, array,
                 comps=comps,
                 times=times,
                 data_layout='xyz')

    test1 = FlowStructND(coords, array,
                         comps=comps,
                         times=times,
                         data_layout='xyz',
                         attrs={'test': 'attr'})

    test2 = FlowStructND(coords, array,
                         comps=comps,
                         times=times,
                         data_layout='xyz',
                         dtype='f4',
                         attrs={'test': 'attr'})

    test3 = FlowStructND(coords, array,
                         comps=comps,
                         times=times,
                         data_layout='xyz',
                         attrs={'test': 'attr'},
                         copy=True)

    assert np.shares_memory(test1._array, array), "Ensure this one is a view"
    assert not np.shares_memory(
        test3._array, array), "Ensure this one is a copy"
    assert np.array_equal(test1._array, test3._array)
    assert test1.dtype == array.dtype.type, "Check type"
    assert test2.dtype == np.float32, "Check type"
    assert test1._coords == coords, "check CoordStruct"
    assert np.array_equal(test1._array, array), "Check main data"
    assert list(test1.times) == list(times), "Check times"
    assert test1.comps == comps, "Check comps"
    assert test1._data_layout == tuple('xyz'), "Check data layout"


def test_invalid_init(reference_fstruct):

    array = reference_fstruct._array.squeeze()
    comps = reference_fstruct.comps
    times = reference_fstruct.times

    coords = reference_fstruct.coords.copy()
    coords.remove('z')
    with pytest.raises(ValueError):

        FlowStructND(coords, array,
                     comps=comps,
                     times=times,
                     data_layout='xyz',
                     dtype='f4',
                     attrs={'test': 'attr'})

    with pytest.raises(ValueError):
        FlowStructND(coords, array,
                     comps=comps,
                     times=times,
                     data_layout='xy',
                     dtype='f4',
                     attrs={'test': 'attr'})

    with pytest.raises(ValueError):
        FlowStructND(reference_fstruct.coords, array,
                     comps=comps,
                     times=times,
                     data_layout='xzy',
                     attrs={'test': 'attr'})

    with pytest.raises(ValueError):
        FlowStructND(reference_fstruct.coords, array,
                     comps=comps,
                     times=times[:len(times)//2],
                     data_layout='xyz',
                     attrs={'test': 'attr'})

    with pytest.raises(ValueError):
        FlowStructND(reference_fstruct.coords, array,
                     comps=comps[:2],
                     times=times,
                     data_layout='xyz',
                     attrs={'test': 'attr'})


def test_properties(reference_fstruct):
    assert reference_fstruct.ndim == 3
    assert reference_fstruct.shape == (200, 50, 100)
    assert reference_fstruct.comps == ['u', 'v', 'w']
    assert np.array_equal(reference_fstruct.times,
                          np.array([100., 200., 300.]))
    assert reference_fstruct.dtype == np.float64
    assert reference_fstruct.flow_type is get_flow_type("Cartesian")

    reference_fstruct.times = [200., 300., 400.]
    assert np.array_equal(reference_fstruct.times,
                          np.array([200., 300., 400.]))
    reference_fstruct.times += 100
    assert np.array_equal(reference_fstruct.times,
                          np.array([300., 400., 500.]))


def test_get_comps_times(reference_fstruct):
    f = reference_fstruct.get(comp='u')
    assert isinstance(f, type(reference_fstruct))
    assert f.comps == ['u'], "Incorrect comps"
    assert f.coords == reference_fstruct.coords
    assert np.array_equal(f.get(comp='u', output_fs=False),
                          reference_fstruct.get(comp='u', output_fs=False))
    assert np.array_equal(f._array,
                          reference_fstruct._array[:, [0]])

    f = reference_fstruct.get(time=100)
    assert isinstance(f, type(reference_fstruct))
    assert f.comps == ['u', 'v', 'w'], "Incorrect comps"
    assert np.array_equal(f.times, np.array([100.])), "Incorrect comps"
    assert f.coords == reference_fstruct.coords
    assert np.array_equal(f._array, reference_fstruct._array[[0]])
    assert np.array_equal(f.get(time=100, output_fs=False),
                          reference_fstruct.get(time=100, output_fs=False))

    f = reference_fstruct.get(comp=slice('u', 'w'))
    assert isinstance(f, type(reference_fstruct))
    assert f.comps == reference_fstruct.comps, "Incorrect comps"
    assert f.coords == reference_fstruct.coords
    assert np.array_equal(f._array,
                          reference_fstruct._array)

    f = reference_fstruct.get(comp=slice('v', None))
    assert isinstance(f, type(reference_fstruct))
    assert f.comps == ['v', 'w'], "Incorrect comps"
    assert f.coords == reference_fstruct.coords
    assert np.array_equal(f._array,
                          reference_fstruct._array[:, 1:, ...])

    f = reference_fstruct.get(comp=['u', 'w'])
    assert isinstance(f, type(reference_fstruct))
    assert f.comps == ['u', 'w'], "Incorrect comps"
    assert f.coords == reference_fstruct.coords
    assert np.array_equal(f.get(comp='u', output_fs=False),
                          reference_fstruct.get(comp='u',
                                                output_fs=False))
    assert np.array_equal(f.get(comp='w', output_fs=False),
                          reference_fstruct.get(comp='w',
                          output_fs=False))

    f = reference_fstruct.get(time=100, comp='u')
    assert isinstance(f, np.ndarray)
    assert np.array_equal(f, reference_fstruct._array[0, 0])

    f = reference_fstruct.get(time=[100, 200], comp='u')
    assert isinstance(f, type(reference_fstruct))
    assert f.comps == ['u'], "Incorrect comps"
    assert np.array_equal(f.times, np.array([100, 200]))
    assert np.array_equal(f._array, reference_fstruct._array[:2, [0]])

    # tests get when times is None: error used to occur here
    f1 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[0],
                      reference_fstruct.comps,
                      reference_fstruct._data_layout,
                      times=None)
    
    
    f1.get(comp=['u', 'w'])

    with pytest.raises(KeyError):
        reference_fstruct.get(comp='z')

    with pytest.raises(KeyError):
        reference_fstruct.get(comp=slice('u', 'z'))

    with pytest.raises(KeyError):
        reference_fstruct.get(comp=['u', 'z'])

    with pytest.raises(ValueError):
        reference_fstruct.get(comp=slice('v', 'u'))

    with pytest.raises(ValueError):
        reference_fstruct.get(comp=slice('u', 'v', 1))

    with pytest.raises(KeyError):
        reference_fstruct.get(time=150)

    with pytest.raises(KeyError):
        reference_fstruct.get(time=[100, 400])

    with pytest.raises(ValueError):
        reference_fstruct.get(time=slice(200, 100))

    with pytest.raises(ValueError):
        reference_fstruct.get(time=slice(100, 200, 1))


def test_get_coords(reference_fstruct):
    f = reference_fstruct.get(x=slice(0, 50))
    assert all(f.coords['x'] < 50+0.5*np.diff(f.coords['x'])[0])
    f = reference_fstruct.get(y=slice(0, 1))
    assert all(f.coords['y'] < 1+0.5*np.diff(f.coords['y'])[0])

    f = reference_fstruct.get(x=slice(0, 50), y=slice(0, 1))
    assert all(f.coords['x'] < 50+0.5*np.diff(f.coords['x'])[0])
    assert all(f.coords['y'] < 1+0.5*np.diff(f.coords['y'])[0])

    f = reference_fstruct.get(x=50)
    assert 'x' not in f.coords.index
    assert np.array_equal(f._array, reference_fstruct._array[:, :, 100])

    f = reference_fstruct.get(x=50, y=1)
    assert 'x' not in f.coords.index
    assert 'y' not in f.coords.index
    assert np.array_equal(f._array, reference_fstruct._array[:, :, 100, 24])

    f = reference_fstruct.get(x=50, y=slice(0, 1))
    assert 'x' not in f.coords.index

    f = reference_fstruct.get(comp='u', x=50, y=slice(0, 1))

    with pytest.raises(ValueError):
        f = reference_fstruct.get(x=slice(None, 101))
    with pytest.raises(ValueError):
        f = reference_fstruct.get(x=slice(-0.5, None))

def test_slice(reference_fstruct):
    f = reference_fstruct.slice[0:50]
    assert all(f.coords['x'] < 50+0.5*np.diff(f.coords['x'])[0])
    f = reference_fstruct.slice[:,0:1]
    assert all(f.coords['y'] < 1+0.5*np.diff(f.coords['y'])[0])

    f = reference_fstruct.slice[0:50,0:1]
    assert all(f.coords['x'] < 50+0.5*np.diff(f.coords['x'])[0])
    assert all(f.coords['y'] < 1+0.5*np.diff(f.coords['y'])[0])

    f = reference_fstruct.slice[50]
    assert 'x' not in f.coords.index
    assert np.array_equal(f._array, reference_fstruct._array[:, :, 100])

    f = reference_fstruct.slice[50,1]
    assert 'x' not in f.coords.index
    assert 'y' not in f.coords.index
    assert np.array_equal(f._array, reference_fstruct._array[:, :, 100, 24])

    f = reference_fstruct.slice[50, 0:1]
    assert 'x' not in f.coords.index

    with pytest.raises(ValueError):
        f = reference_fstruct.slice[:101]
    with pytest.raises(ValueError):
        f = reference_fstruct.slice[-0.5:]

def test_reduce(reference_fstruct):
    sum_op = reference_fstruct.reduce(np.sum, axis='z')
    sum_array = reference_fstruct._array.sum(axis=-1)
    assert sum_op.comps == reference_fstruct.comps
    assert np.array_equal(sum_op.times, reference_fstruct.times)
    assert np.array_equal(sum_array, sum_op._array)


def test_ufuncs(reference_fstruct):
    sqrt_op = np.sqrt(reference_fstruct)
    assert np.array_equal(np.sqrt(reference_fstruct._array),
                          sqrt_op._array)


def test_functions(reference_fstruct):

    with pytest.raises(TypeError):
        np.tensordot(reference_fstruct, reference_fstruct)

    np.allclose(reference_fstruct, reference_fstruct)

    np.array_equal(reference_fstruct, reference_fstruct)


def test_getitem(reference_fstruct):
    f1 = reference_fstruct.get(time=100, comp='u')
    f2 = reference_fstruct[100, 'u']

    assert np.array_equal(f1, f2)


def test_setitem(reference_fstruct):
    array = np.random.randn(*reference_fstruct.shape)
    reference_fstruct[100, 'u'] = array

    assert np.array_equal(reference_fstruct[100, 'u'], array)

    with pytest.raises(ValueError):
        reference_fstruct[100, 'u'] = array[:, :, :-1]

    array = np.random.randn(3,1,*reference_fstruct.shape)
    reference_fstruct[[100,200,300],'p' ] = array

    ref2 = FlowStructND(reference_fstruct._coords,
                        array=reference_fstruct._array,
                        comps=['u','v','w','p'],
                        times=[100,200,300],
                        data_layout=reference_fstruct._data_layout)
    
    assert reference_fstruct == ref2

    f1 = reference_fstruct.get(time=100,
                               comp=slice('u','w'))
    f1[100,'p'] = reference_fstruct[100,'p']

def test_concat_comps(reference_fstruct):
    f1 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[:, :1],
                      ['u'],
                      'xyz',
                      reference_fstruct.times)

    f2 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[:, 1:],
                      ['v', 'w'],
                      'xyz',
                      reference_fstruct.times)

    f3 = f1.concat_comps(f2)
    assert f3 == reference_fstruct

    f1.concat_comps(f2, inplace=True)
    assert f1 == reference_fstruct

    f1 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[:, :1],
                      ['u'],
                      'xyz',
                      reference_fstruct.times)

    f2 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[:, 1:],
                      ['u', 'w'],
                      'xyz',
                      reference_fstruct.times)

    with pytest.raises(ValueError):
        f1.concat_comps(f2)


def test_concat_times(reference_fstruct):
    f1 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[:1],
                      ['u', 'v', 'w'],
                      'xyz',
                      reference_fstruct.times[:1])

    f2 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[1:],
                      ['u', 'v', 'w'],
                      'xyz',
                      reference_fstruct.times[1:])

    f3 = f1.concat_times(f2)
    assert f3 == reference_fstruct

    f1.concat_times(f2, inplace=True)
    assert f1 == reference_fstruct


    f1 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[:1],
                      ['u', 'v', 'w'],
                      'xyz',
                      [100])

    f2 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[1:],
                      ['u', 'v', 'w'],
                      'xyz',
                      [100, 300])

    with pytest.raises(ValueError):
        f1.concat_times(f2)


def test_concat(reference_fstruct):
    f1 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[:, :1],
                      ['u'],
                      'xyz',
                      reference_fstruct.times)

    f2 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[:, 1:],
                      ['v', 'w'],
                      'xyz',
                      reference_fstruct.times)
    
    f3 = f1.concat(f2)
    assert f3 == reference_fstruct

    f1.concat(f2, inplace=True)
    assert f1 == reference_fstruct

    f1 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[:1],
                      ['u', 'v', 'w'],
                      'xyz',
                      [100])

    f2 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[1:],
                      ['u', 'v', 'w'],
                      'xyz',
                      [100, 300])

    with pytest.raises(ValueError):
        f1.concat(f2)

    f1 = FlowStructND(reference_fstruct.coords,
                    reference_fstruct._array[:, :1],
                    ['u'],
                    'xyz',
                    reference_fstruct.times)

    f2 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[:, 1:],
                      ['v', 'w'],
                      'xyz',
                      reference_fstruct.times)

    f3 = f1.concat(f2)
    assert f3 == reference_fstruct

    f1 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[:, :1],
                      ['u'],
                      'xyz',
                      reference_fstruct.times)

    f2 = FlowStructND(reference_fstruct.coords,
                      reference_fstruct._array[:, 1:],
                      ['u', 'w'],
                      'xyz',
                      reference_fstruct.times)

    with pytest.raises(ValueError):
        f1.concat(f2)

def test_copy(reference_fstruct):
    f = reference_fstruct.copy()

    assert not np.shares_memory(f._array,
                                reference_fstruct._array), \
        "Ensure this one is a view"


def test_translate(reference_fstruct):
    f = reference_fstruct
    f.Translate(x=-50)

    assert all(f.coords['x'] < 50+0.5*np.diff(f.coords['x'])[0])
    assert all(f.coords['x'] > -50-0.5*np.diff(f.coords['x'])[0])


def test_to_hdf(reference_fstruct, test_filename):

    reference_fstruct.to_hdf(test_filename, 'w')

    fstruct2 = reference_fstruct.__class__.from_hdf(test_filename)

    assert reference_fstruct == fstruct2


def test_to_netcdf(reference_fstruct, test_filename):

    reference_fstruct.to_netcdf(test_filename, 'w')

    fstruct2 = reference_fstruct.__class__.from_netcdf(test_filename)

    assert reference_fstruct == fstruct2

def test_plot_exceptions(reference_fstruct):

    struct1d = reference_fstruct.get(x=50,z=3)
    struct2d = reference_fstruct.get(y=1)

    with pytest.raises(ValueError):
        struct1d.plot_line('u','y')

    with pytest.raises(ValueError):
        struct2d.pcolormesh('u','xz')

    with pytest.warns(UserWarning):
        struct1d.plot_line('u','x',time=100)

    with pytest.warns(UserWarning):
        struct1d.plot_line('u',loc=2,time=100)

    with pytest.warns(UserWarning):
        struct2d.pcolormesh('u','xz',1,time=100)

    with pytest.raises(ValueError):
        struct1d.pcolormesh('u','xz',time=100)


@check_figures_equal()
def test_plot_line(fig_test, fig_ref, reference_fstruct):
    ax = fig_test.subplots()
    reference_fstruct.plot_line('u','y', {'x': 50, 'z': 3},
                                time=100,
                                ax=ax)

    # y = reference_fstruct._array[0, 0, 100, :, 50]
    y = reference_fstruct.get(x=50, z=3, time=100, comp='u')
    x = reference_fstruct.coords['y']

    ax1 = fig_ref.subplots()
    ax1.plot(x, y)

@check_figures_equal()
def test_plot_line1D(fig_test, fig_ref, reference_fstruct):
    ax = fig_test.subplots()
    struct1d = reference_fstruct.get(x=50,z=3)

    struct1d.plot_line('u',
                        time=100,
                        ax=ax)

    # y = reference_fstruct._array[0, 0, 100, :, 50]
    y = reference_fstruct.get(x=50, z=3, time=100, comp='u')
    x = reference_fstruct.coords['y']

    ax1 = fig_ref.subplots()
    ax1.plot(x, y)


@check_figures_equal()
def test_pcolormesh(fig_test, fig_ref, reference_fstruct):
    ax = fig_test.subplots()
    reference_fstruct.pcolormesh('u','xz', 1, time=100, ax=ax)

    ax1 = fig_ref.subplots()
    y = reference_fstruct.coords['z']
    x = reference_fstruct.coords['x']
    z = reference_fstruct.get(y=1, comp='u', time=100)

    ax1.pcolormesh(x, y, z.T)

@check_figures_equal()
def test_pcolormesh2D(fig_test, fig_ref, reference_fstruct):
    ax = fig_test.subplots()

    struct2d = reference_fstruct.get(y=1)
    struct2d.pcolormesh('u','xz', time=100, ax=ax)

    ax1 = fig_ref.subplots()
    y = reference_fstruct.coords['z']
    x = reference_fstruct.coords['x']
    z = reference_fstruct.get(y=1, comp='u', time=100)

    ax1.pcolormesh(x, y, z.T)


@check_figures_equal()
def test_contour(fig_test, fig_ref, reference_fstruct):
    ax = fig_test.subplots()
    reference_fstruct.contour('u', 'xz', 1, time=100, ax=ax)

    ax1 = fig_ref.subplots()
    y = reference_fstruct.coords['z']
    x = reference_fstruct.coords['x']
    z = reference_fstruct.get(y=1, comp='u', time=100)

    ax1.contour(x, y, z.T)

@check_figures_equal()
def test_contour2D(fig_test, fig_ref, reference_fstruct):
    ax = fig_test.subplots()
    
    struct2d = reference_fstruct.get(y=1)
    struct2d.contour('u','xz', time=100, ax=ax)

    ax1 = fig_ref.subplots()
    y = reference_fstruct.coords['z']
    x = reference_fstruct.coords['x']
    z = reference_fstruct.get(y=1, comp='u', time=100)

    ax1.contour(x, y, z.T)


@check_figures_equal()
def test_contourf(fig_test, fig_ref, reference_fstruct):
    ax = fig_test.subplots()
    reference_fstruct.contourf('u', 'xz', 1, time=100, ax=ax)

    ax1 = fig_ref.subplots()
    y = reference_fstruct.coords['z']
    x = reference_fstruct.coords['x']
    z = reference_fstruct.get(y=1, comp='u', time=100)

    ax1.contourf(x, y, z.T)

@check_figures_equal()
def test_contourf2D(fig_test, fig_ref, reference_fstruct):
    ax = fig_test.subplots()
    
    struct2d = reference_fstruct.get(y=1)
    struct2d.contourf('u','xz', time=100, ax=ax)

    ax1 = fig_ref.subplots()
    y = reference_fstruct.coords['z']
    x = reference_fstruct.coords['x']
    z = reference_fstruct.get(y=1, comp='u', time=100)

    ax1.contourf(x, y, z.T)


def test_first_derivative(reference_fstruct):

    data = reference_fstruct.first_derivative('u', 'x')
    assert data.shape == (3, 200, 50, 100)

    data = reference_fstruct.first_derivative('u', 'x', time=100)
    assert data.shape == (200, 50, 100)


def test_second_derivative(reference_fstruct):

    data = reference_fstruct.second_derivative('u', 'x')
    assert data.shape == (3, 200, 50, 100)

    data = reference_fstruct.second_derivative('u', 'x', time=100)
    assert data.shape == (200, 50, 100)


def test_to_vtk(reference_fstruct):
    reference_fstruct.to_vtk(time=100)


def test_window_uniform(reference_fstruct):

    window_ref = reference_fstruct.window('uniform',101)

    ref_array = reference_fstruct._array[:2].mean(axis=0)
    assert np.array_equal(ref_array,window_ref._array[0])

    ref_array = reference_fstruct._array.mean(axis=0)
    assert np.array_equal(ref_array,window_ref._array[1])

    ref_array = reference_fstruct._array[1:].mean(axis=0)
    assert np.array_equal(ref_array,window_ref._array[2])

    with pytest.warns(UserWarning):
        reference_fstruct.window('uniform',50)

def test_times_to_ND(reference_fstruct):
    timestruct = reference_fstruct.time_to_ND()

    assert timestruct.times is None
    assert np.array_equal(timestruct.coords['t'], reference_fstruct.times)
    for i in range(len(reference_fstruct.times)):
        assert np.allclose(reference_fstruct._array[i],timestruct._array[...,i],atol=0,rtol=1e-10)