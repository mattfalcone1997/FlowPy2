from flowpy.flowstruct import FlowStructND
from flowpy.coords import CoordStruct
from flowpy.flow_type import CartesianFlow
import pytest
from math import prod
import numpy as np


@pytest.fixture
def reference_fstruct():
    data = {'x': np.linspace(0, 100, 200),
            'y': np.linspace(0, 2, 50),
            'z': np.linspace(0, 6, 100)}
    coords = CoordStruct(CartesianFlow,
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
    return CoordStruct(CartesianFlow,
                       data)


def test_valid_init(reference_fstruct):

    times = reference_fstruct.times
    comps = reference_fstruct.comps
    coords = reference_fstruct.coords
    array = reference_fstruct._array.squeeze()

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
    assert reference_fstruct.flow_type is CartesianFlow

    reference_fstruct.times = [200., 300., 400.]
    assert np.array_equal(reference_fstruct.times,
                          np.array([200., 300., 400.]))
    reference_fstruct.times += 100
    assert np.array_equal(reference_fstruct.times,
                          np.array([300., 400., 500.]))


def test_get_out_fs(reference_fstruct):
    pass


def test_get_array(reference_fstruct):
    pass


def test_reduce(reference_fstruct):
    pass


def test_ufuncs(reference_fstruct):
    pass


def test_functions(reference_fstruct):
    pass


def test_getitem(reference_fstruct):
    pass


def test_concat_comps(reference_fstruct):
    pass


def test_concat_times(reference_fstruct):
    pass


def test_copy(reference_fstruct):
    pass


def test_translate(reference_fstruct):
    pass


def test_plot_line(reference_fstruct):
    pass


def test_pcolormeah(reference_fstruct):
    pass


def test_contour(reference_fstruct):
    pass


def test_contourf(reference_fstruct):
    pass
