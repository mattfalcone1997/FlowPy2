from flowpy2.flowstruct import FlowStructND
from flowpy2.coords import CoordStruct
from flowpy2.flow_type import get_flow_type
from matplotlib.testing.decorators import check_figures_equal

import pytest

from math import prod
import numpy as np

from tempfile import NamedTemporaryFile


@pytest.fixture
def fstruct_with_times():
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
def fstruct_time_None():
    data = {'x': np.linspace(0, 100, 200),
            'y': np.linspace(0, 2, 50),
            'z': np.linspace(0, 6, 100)}
    coords = CoordStruct("Cartesian",
                         data)
    times = None
    comps = ['u', 'v', 'w']
    size = prod([len(comps), *[d.size for d in data.values()]])
    array_flat = np.arange(1, size+1, dtype='f8')
    array = array_flat.reshape(
        (len(comps), *[d.size for d in data.values()]))

    return FlowStructND(coords, array, comps=comps, times=times, data_layout='xyz')


@pytest.fixture
def reference_coords():
    data = {'x': np.linspace(0, 100, 200),
            'y': np.linspace(0, 2, 50),
            'z': np.linspace(0, 6, 100)}
    return CoordStruct("Cartesian",
                       data)


def loop_fstructs(func):

    d = pytest.mark.parametrize("fstruct, time",
                                [('fstruct_with_times', 100),
                                 ('fstruct_time_None', None)])
    return d(func)


def test_valid_init(fstruct_with_times):

    times = fstruct_with_times.times
    comps = fstruct_with_times.comps
    coords = fstruct_with_times.coords

    array = fstruct_with_times._array.squeeze()

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


def test_invalid_init(fstruct_with_times):

    array = fstruct_with_times._array.squeeze()
    comps = fstruct_with_times.comps
    times = fstruct_with_times.times

    coords = fstruct_with_times.coords.copy()
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
        FlowStructND(fstruct_with_times.coords, array,
                     comps=comps,
                     times=times,
                     data_layout='xzy',
                     attrs={'test': 'attr'})

    with pytest.raises(ValueError):
        FlowStructND(fstruct_with_times.coords, array,
                     comps=comps,
                     times=times[:len(times)//2],
                     data_layout='xyz',
                     attrs={'test': 'attr'})

    with pytest.raises(ValueError):
        FlowStructND(fstruct_with_times.coords, array,
                     comps=comps[:2],
                     times=times,
                     data_layout='xyz',
                     attrs={'test': 'attr'})


def test_properties(fstruct_with_times):
    assert fstruct_with_times.ndim == 3
    assert fstruct_with_times.shape == (200, 50, 100)
    assert fstruct_with_times.comps == ['u', 'v', 'w']
    assert np.array_equal(fstruct_with_times.times,
                          np.array([100., 200., 300.]))
    assert fstruct_with_times.dtype == np.float64
    assert fstruct_with_times.flow_type is get_flow_type("Cartesian")

    fstruct_with_times.times = [200., 300., 400.]
    assert np.array_equal(fstruct_with_times.times,
                          np.array([200., 300., 400.]))
    fstruct_with_times.times += 100
    assert np.array_equal(fstruct_with_times.times,
                          np.array([300., 400., 500.]))


def test_get_comps_times(fstruct_with_times):

    f = fstruct_with_times.get(comp=['u'])

    assert isinstance(f, type(fstruct_with_times))
    assert f.comps == ['u'], "Incorrect comps"
    assert f.coords == fstruct_with_times.coords
    assert np.array_equal(f.get(comp=['u'], output_fs=False),
                          fstruct_with_times.get(comp=['u'], output_fs=False))
    assert np.array_equal(f._array,
                          fstruct_with_times._array[:, [0]])

    f = fstruct_with_times.get(time=100)
    assert isinstance(f, type(fstruct_with_times))
    assert f.comps == ['u', 'v', 'w'], "Incorrect comps"
    assert np.array_equal(f.times, np.array([100.])), "Incorrect comps"
    assert f.coords == fstruct_with_times.coords
    assert np.array_equal(f._array, fstruct_with_times._array[[0]])
    assert np.array_equal(f.get(time=100, output_fs=False),
                          fstruct_with_times.get(time=100, output_fs=False))

    f = fstruct_with_times.get(comp=slice('u', 'w'))
    assert isinstance(f, type(fstruct_with_times))
    assert f.comps == fstruct_with_times.comps, "Incorrect comps"
    assert f.coords == fstruct_with_times.coords
    assert np.array_equal(f._array,
                          fstruct_with_times._array)

    f = fstruct_with_times.get(comp=slice('v', None))
    assert isinstance(f, type(fstruct_with_times))
    assert f.comps == ['v', 'w'], "Incorrect comps"
    assert f.coords == fstruct_with_times.coords
    assert np.array_equal(f._array,
                          fstruct_with_times._array[:, 1:, ...])

    f = fstruct_with_times.get(comp=['u', 'w'])
    assert isinstance(f, type(fstruct_with_times))
    assert f.comps == ['u', 'w'], "Incorrect comps"
    assert f.coords == fstruct_with_times.coords
    assert np.array_equal(f.get(comp='u', output_fs=False),
                          fstruct_with_times.get(comp='u',
                                                 output_fs=False))
    assert np.array_equal(f.get(comp='w', output_fs=False),
                          fstruct_with_times.get(comp='w',
                          output_fs=False))

    f = fstruct_with_times.get(time=100, comp='u')
    assert isinstance(f, np.ndarray)
    assert np.array_equal(f, fstruct_with_times._array[0, 0])

    f = fstruct_with_times.get(time=[100, 200], comp='u')
    assert isinstance(f, type(fstruct_with_times))
    assert f.comps == ['u'], "Incorrect comps"
    assert np.array_equal(f.times, np.array([100, 200]))
    assert np.array_equal(f._array, fstruct_with_times._array[:2, [0]])

    # tests get when times is None: error used to occur here
    f1 = FlowStructND(fstruct_with_times.coords,
                      fstruct_with_times._array[0],
                      fstruct_with_times.comps,
                      fstruct_with_times._data_layout,
                      times=None)

    f1.get(comp=['u', 'w'])

    with pytest.raises(KeyError):
        fstruct_with_times.get(comp='z')

    with pytest.raises(KeyError):
        fstruct_with_times.get(comp=slice('u', 'z'))

    with pytest.raises(KeyError):
        fstruct_with_times.get(comp=['u', 'z'])

    with pytest.raises(ValueError):
        fstruct_with_times.get(comp=slice('v', 'u'))

    with pytest.raises(ValueError):
        fstruct_with_times.get(comp=slice('u', 'v', 1))

    with pytest.raises(KeyError):
        fstruct_with_times.get(time=150)

    with pytest.raises(KeyError):
        fstruct_with_times.get(time=[100, 400])

    with pytest.raises(ValueError):
        fstruct_with_times.get(time=slice(200, 100))

    with pytest.raises(ValueError):
        fstruct_with_times.get(time=slice(100, 200, 1))


def test_get_coords(fstruct_with_times):
    f = fstruct_with_times.get(x=slice(0, 50))
    assert all(f.coords['x'] < 50+0.5*np.diff(f.coords['x'])[0])
    f = fstruct_with_times.get(y=slice(0, 1))
    assert all(f.coords['y'] < 1+0.5*np.diff(f.coords['y'])[0])

    f = fstruct_with_times.get(x=slice(0, 50), y=slice(0, 1))
    assert all(f.coords['x'] < 50+0.5*np.diff(f.coords['x'])[0])
    assert all(f.coords['y'] < 1+0.5*np.diff(f.coords['y'])[0])

    f = fstruct_with_times.get(x=50)
    assert 'x' not in f.coords.index
    assert np.array_equal(f._array, fstruct_with_times._array[:, :, 100])

    f = fstruct_with_times.get(x=50, y=1)
    assert 'x' not in f.coords.index
    assert 'y' not in f.coords.index
    assert np.array_equal(f._array, fstruct_with_times._array[:, :, 100, 24])

    f = fstruct_with_times.get(x=50, y=slice(0, 1))
    assert 'x' not in f.coords.index

    f = fstruct_with_times.get(comp='u', x=50, y=slice(0, 1))

    with pytest.raises(ValueError):
        f = fstruct_with_times.get(x=slice(None, 101))
    with pytest.raises(ValueError):
        f = fstruct_with_times.get(x=slice(-0.5, None))


def test_slice(fstruct_with_times):
    f = fstruct_with_times.slice[0:50]
    assert all(f.coords['x'] < 50+0.5*np.diff(f.coords['x'])[0])
    f = fstruct_with_times.slice[:, 0:1]
    assert all(f.coords['y'] < 1+0.5*np.diff(f.coords['y'])[0])

    f = fstruct_with_times.slice[0:50, 0:1]
    assert all(f.coords['x'] < 50+0.5*np.diff(f.coords['x'])[0])
    assert all(f.coords['y'] < 1+0.5*np.diff(f.coords['y'])[0])

    f = fstruct_with_times.slice[50]
    assert 'x' not in f.coords.index
    assert np.array_equal(f._array, fstruct_with_times._array[:, :, 100])

    f = fstruct_with_times.slice[50, 1]
    assert 'x' not in f.coords.index
    assert 'y' not in f.coords.index
    assert np.array_equal(f._array, fstruct_with_times._array[:, :, 100, 24])

    f = fstruct_with_times.slice[50, 0:1]
    assert 'x' not in f.coords.index

    with pytest.raises(ValueError):
        f = fstruct_with_times.slice[:101]
    with pytest.raises(ValueError):
        f = fstruct_with_times.slice[-0.5:]


def test_reduce(fstruct_with_times):
    sum_op = fstruct_with_times.reduce(np.sum, axis='z')
    sum_array = fstruct_with_times._array.sum(axis=-1)
    assert sum_op.comps == fstruct_with_times.comps
    assert np.array_equal(sum_op.times, fstruct_with_times.times)
    assert np.array_equal(sum_array, sum_op._array)


def test_ufuncs(fstruct_with_times):
    sqrt_op = np.sqrt(fstruct_with_times)
    assert np.array_equal(np.sqrt(fstruct_with_times._array),
                          sqrt_op._array)


def test_functions(fstruct_with_times):

    with pytest.raises(TypeError):
        np.tensordot(fstruct_with_times, fstruct_with_times)

    np.allclose(fstruct_with_times, fstruct_with_times)

    np.array_equal(fstruct_with_times, fstruct_with_times)


def test_getitem(fstruct_with_times):
    f1 = fstruct_with_times.get(time=100, comp='u')
    f2 = fstruct_with_times[100, 'u']

    assert np.array_equal(f1, f2)


def test_setitem(fstruct_with_times):
    array = np.random.randn(*fstruct_with_times.shape)
    fstruct_with_times[100, 'u'] = array

    assert np.array_equal(fstruct_with_times[100, 'u'], array)

    with pytest.raises(ValueError):
        fstruct_with_times[100, 'u'] = array[:, :, :-1]

    array = np.random.randn(3, 1, *fstruct_with_times.shape)
    fstruct_with_times[[100, 200, 300], 'p'] = array

    ref2 = FlowStructND(fstruct_with_times._coords,
                        array=fstruct_with_times._array,
                        comps=['u', 'v', 'w', 'p'],
                        times=[100, 200, 300],
                        data_layout=fstruct_with_times._data_layout)

    assert fstruct_with_times == ref2

    f1 = fstruct_with_times.get(time=100,
                                comp=slice('u', 'w'))
    f1[100, 'p'] = fstruct_with_times[100, 'p']


@loop_fstructs
def test_concat_comps(fstruct, time, request):

    fstruct = request.getfixturevalue(fstruct)
    f1 = FlowStructND(fstruct.coords,
                      fstruct._array[:, :1],
                      ['u'],
                      'xyz',
                      fstruct.times)

    f2 = FlowStructND(fstruct.coords,
                      fstruct._array[:, 1:],
                      ['v', 'w'],
                      'xyz',
                      fstruct.times)

    f3 = f1.concat_comps(f2)
    assert f3 == fstruct

    f1.concat_comps(f2, inplace=True)
    assert f1 == fstruct

    f1 = FlowStructND(fstruct.coords,
                      fstruct._array[:, :1],
                      ['u'],
                      'xyz',
                      fstruct.times)

    f2 = FlowStructND(fstruct.coords,
                      fstruct._array[:, 1:],
                      ['u', 'w'],
                      'xyz',
                      fstruct.times)

    with pytest.raises(ValueError):
        f1.concat_comps(f2)


def test_concat_times(fstruct_with_times):
    f1 = FlowStructND(fstruct_with_times.coords,
                      fstruct_with_times._array[:1],
                      ['u', 'v', 'w'],
                      'xyz',
                      fstruct_with_times.times[:1])

    f2 = FlowStructND(fstruct_with_times.coords,
                      fstruct_with_times._array[1:],
                      ['u', 'v', 'w'],
                      'xyz',
                      fstruct_with_times.times[1:])

    f3 = f1.concat_times(f2)
    assert f3 == fstruct_with_times

    f1.concat_times(f2, inplace=True)
    assert f1 == fstruct_with_times

    f1 = FlowStructND(fstruct_with_times.coords,
                      fstruct_with_times._array[:1],
                      ['u', 'v', 'w'],
                      'xyz',
                      [100])

    f2 = FlowStructND(fstruct_with_times.coords,
                      fstruct_with_times._array[1:],
                      ['u', 'v', 'w'],
                      'xyz',
                      [100, 300])

    with pytest.raises(ValueError):
        f1.concat_times(f2)


def test_concat(fstruct_with_times):
    f1 = FlowStructND(fstruct_with_times.coords,
                      fstruct_with_times._array[:, :1],
                      ['u'],
                      'xyz',
                      fstruct_with_times.times)

    f2 = FlowStructND(fstruct_with_times.coords,
                      fstruct_with_times._array[:, 1:],
                      ['v', 'w'],
                      'xyz',
                      fstruct_with_times.times)

    f3 = f1.concat(f2)
    assert f3 == fstruct_with_times

    f1.concat(f2, inplace=True)
    assert f1 == fstruct_with_times

    f1 = FlowStructND(fstruct_with_times.coords,
                      fstruct_with_times._array[:1],
                      ['u', 'v', 'w'],
                      'xyz',
                      [100])

    f2 = FlowStructND(fstruct_with_times.coords,
                      fstruct_with_times._array[1:],
                      ['u', 'v', 'w'],
                      'xyz',
                      [100, 300])

    with pytest.raises(ValueError):
        f1.concat(f2)

    f1 = FlowStructND(fstruct_with_times.coords,
                      fstruct_with_times._array[:, :1],
                      ['u'],
                      'xyz',
                      fstruct_with_times.times)

    f2 = FlowStructND(fstruct_with_times.coords,
                      fstruct_with_times._array[:, 1:],
                      ['v', 'w'],
                      'xyz',
                      fstruct_with_times.times)

    f3 = f1.concat(f2)
    assert f3 == fstruct_with_times

    f1 = FlowStructND(fstruct_with_times.coords,
                      fstruct_with_times._array[:, :1],
                      ['u'],
                      'xyz',
                      fstruct_with_times.times)

    f2 = FlowStructND(fstruct_with_times.coords,
                      fstruct_with_times._array[:, 1:],
                      ['u', 'w'],
                      'xyz',
                      fstruct_with_times.times)

    with pytest.raises(ValueError):
        f1.concat(f2)


def test_copy(fstruct_with_times):
    f = fstruct_with_times.copy()

    assert not np.shares_memory(f._array,
                                fstruct_with_times._array), \
        "Ensure this one is a view"


def test_translate(fstruct_with_times):
    f = fstruct_with_times
    f.Translate(x=-50)

    assert all(f.coords['x'] < 50+0.5*np.diff(f.coords['x'])[0])
    assert all(f.coords['x'] > -50-0.5*np.diff(f.coords['x'])[0])


@loop_fstructs
def test_to_hdf(fstruct, time, request):
    fstruct = request.getfixturevalue(fstruct)

    with NamedTemporaryFile(suffix='.h5') as f:
        fstruct.to_hdf(f.name, 'w')

        fstruct2 = fstruct.__class__.from_hdf(f.name)

    assert fstruct == fstruct2

@loop_fstructs
def test_to_netcdf(fstruct, time, request):
    fstruct = request.getfixturevalue(fstruct)

    with NamedTemporaryFile(suffix='.h5') as f:
        fstruct.to_netcdf(f.name, 'w')

        fstruct2 = fstruct.__class__.from_netcdf(f.name)

    assert fstruct == fstruct2


def test_plot_exceptions(fstruct_with_times):

    struct1d = fstruct_with_times.get(x=50, z=3)
    struct2d = fstruct_with_times.get(y=1)

    with pytest.raises(ValueError):
        struct1d.plot_line('u', 'y')

    with pytest.raises(ValueError):
        struct2d.pcolormesh('u', 'xz')

    with pytest.warns(UserWarning):
        struct1d.plot_line('u', 'x', time=100)

    with pytest.warns(UserWarning):
        struct1d.plot_line('u', loc=2, time=100)

    with pytest.warns(UserWarning):
        struct2d.pcolormesh('u', 'xz', 1, time=100)

    with pytest.raises(ValueError):
        struct1d.pcolormesh('u', 'xz', time=100)


@check_figures_equal()
def test_plot_line(fig_test, fig_ref, fstruct_with_times):
    ax = fig_test.subplots()
    fstruct_with_times.plot_line('u', 'y', {'x': 50, 'z': 3},
                                 time=100,
                                 ax=ax)

    # y = fstruct_with_times._array[0, 0, 100, :, 50]
    y = fstruct_with_times.get(x=50, z=3, time=100, comp='u')
    x = fstruct_with_times.coords['y']

    ax1 = fig_ref.subplots()
    ax1.plot(x, y)


@check_figures_equal()
def test_plot_line1D(fig_test, fig_ref, fstruct_with_times):
    ax = fig_test.subplots()
    struct1d = fstruct_with_times.get(x=50, z=3)

    struct1d.plot_line('u',
                       time=100,
                       ax=ax)

    # y = fstruct_with_times._array[0, 0, 100, :, 50]
    y = fstruct_with_times.get(x=50, z=3, time=100, comp='u')
    x = fstruct_with_times.coords['y']

    ax1 = fig_ref.subplots()
    ax1.plot(x, y)

# Create more detail but standardised tests to the quiver plots


@check_figures_equal()
def test_pcolormesh(fig_test, fig_ref, fstruct_with_times):
    ax = fig_test.subplots()
    fstruct_with_times.pcolormesh('u', 'xz', 1, time=100, ax=ax)

    ax1 = fig_ref.subplots()
    y = fstruct_with_times.coords['z']
    x = fstruct_with_times.coords['x']
    z = fstruct_with_times.get(y=1, comp='u', time=100)

    ax1.pcolormesh(x, y, z.T)


@check_figures_equal()
def test_pcolormesh2D(fig_test, fig_ref, fstruct_with_times):
    ax = fig_test.subplots()

    struct2d = fstruct_with_times.get(y=1)
    struct2d.pcolormesh('u', 'xz', time=100, ax=ax)

    ax1 = fig_ref.subplots()
    y = fstruct_with_times.coords['z']
    x = fstruct_with_times.coords['x']
    z = fstruct_with_times.get(y=1, comp='u', time=100)

    ax1.pcolormesh(x, y, z.T)


@check_figures_equal()
def test_contour(fig_test, fig_ref, fstruct_with_times):
    ax = fig_test.subplots()
    fstruct_with_times.contour('u', 'xz', 1, time=100, ax=ax)

    ax1 = fig_ref.subplots()
    y = fstruct_with_times.coords['z']
    x = fstruct_with_times.coords['x']
    z = fstruct_with_times.get(y=1, comp='u', time=100)

    ax1.contour(x, y, z.T)


@check_figures_equal()
def test_contour2D(fig_test, fig_ref, fstruct_with_times):
    ax = fig_test.subplots()

    struct2d = fstruct_with_times.get(y=1)
    struct2d.contour('u', 'xz', time=100, ax=ax)

    ax1 = fig_ref.subplots()
    y = fstruct_with_times.coords['z']
    x = fstruct_with_times.coords['x']
    z = fstruct_with_times.get(y=1, comp='u', time=100)

    ax1.contour(x, y, z.T)


@check_figures_equal()
def test_quiver(fig_test, fig_ref, fstruct_with_times):
    ax = fig_test.subplots()
    fstruct_with_times.quiver(['u', 'v'], 'xy', 1, time=100, ax=ax,
                              scale=None,
                              scale_units=None)

    ax1 = fig_ref.subplots()
    y = fstruct_with_times.coords['y']
    x = fstruct_with_times.coords['x']

    u = fstruct_with_times.get(z=1, comp='u', time=100)
    v = fstruct_with_times.get(z=1, comp='v', time=100)

    ax1.quiver(x, y, u.T, v.T)


@check_figures_equal()
def test_quiver2D(fig_test, fig_ref, fstruct_with_times):
    ax = fig_test.subplots()

    struct2d = fstruct_with_times.get(z=1)
    struct2d.quiver(['u', 'v'], 'xy', time=100, ax=ax,
                    scale=None,
                    scale_units=None)

    ax1 = fig_ref.subplots()
    y = struct2d.coords['y']
    x = struct2d.coords['x']

    u = struct2d.get(comp='u', time=100)
    v = struct2d.get(comp='v', time=100)

    ax1.quiver(x, y, u.T, v.T)


@check_figures_equal()
def test_contourf(fig_test, fig_ref, fstruct_with_times):
    ax = fig_test.subplots()
    fstruct_with_times.contourf('u', 'xz', 1, time=100, ax=ax)

    ax1 = fig_ref.subplots()
    y = fstruct_with_times.coords['z']
    x = fstruct_with_times.coords['x']
    z = fstruct_with_times.get(y=1, comp='u', time=100)

    ax1.contourf(x, y, z.T)


@check_figures_equal()
def test_contourf2D(fig_test, fig_ref, fstruct_with_times):
    ax = fig_test.subplots()

    struct2d = fstruct_with_times.get(y=1)
    struct2d.contourf('u', 'xz', time=100, ax=ax)

    ax1 = fig_ref.subplots()
    y = fstruct_with_times.coords['z']
    x = fstruct_with_times.coords['x']
    z = fstruct_with_times.get(y=1, comp='u', time=100)

    ax1.contourf(x, y, z.T)


def test_first_derivative(fstruct_with_times):

    data = fstruct_with_times.first_derivative('u', 'x')
    assert data.shape == (3, 200, 50, 100)

    data = fstruct_with_times.first_derivative('u', 'x', time=100)
    assert data.shape == (200, 50, 100)


def test_second_derivative(fstruct_with_times):

    data = fstruct_with_times.second_derivative('u', 'x')
    assert data.shape == (3, 200, 50, 100)

    data = fstruct_with_times.second_derivative('u', 'x', time=100)
    assert data.shape == (200, 50, 100)

def test_integrate(fstruct_with_times):
    data = fstruct_with_times.integrate('u', 'x')
    assert data.shape == (3, 50, 100)

    data = fstruct_with_times.integrate('u', 'x', time=100)
    assert data.shape == (50, 100)

def test_cumulative_integrate(fstruct_with_times):
    data = fstruct_with_times.cumulative_integrate('u', 'x')
    assert data.shape == (3, 200, 50, 100)

    data = fstruct_with_times.cumulative_integrate('u', 'x', time=100)
    assert data.shape == (200, 50, 100)



def test_to_vtk(fstruct_with_times):
    fstruct_with_times.to_vtk(time=100)


def test_window_uniform(fstruct_with_times):

    window_ref = fstruct_with_times.window('uniform', 101)

    ref_array = fstruct_with_times._array[:2].mean(axis=0)
    assert np.array_equal(ref_array, window_ref._array[0])

    ref_array = fstruct_with_times._array.mean(axis=0)
    assert np.array_equal(ref_array, window_ref._array[1])

    ref_array = fstruct_with_times._array[1:].mean(axis=0)
    assert np.array_equal(ref_array, window_ref._array[2])

    with pytest.warns(UserWarning):
        fstruct_with_times.window('uniform', 50)


def test_times_to_ND(fstruct_with_times):
    timestruct = fstruct_with_times.time_to_ND()

    assert timestruct.times is None
    assert np.array_equal(timestruct.coords['t'], fstruct_with_times.times)
    for i in range(len(fstruct_with_times.times)):
        assert np.allclose(
            fstruct_with_times._array[i], timestruct._array[..., i], atol=0, rtol=1e-10)
