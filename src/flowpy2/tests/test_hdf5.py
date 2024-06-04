from flowpy2.io import hdf5
import h5py
import pytest
from tempfile import NamedTemporaryFile
import numpy as np


@pytest.fixture(autouse=True)
def test_filename():
    return NamedTemporaryFile(suffix='.h5').name


@pytest.fixture()
def test_group():
    name = NamedTemporaryFile(suffix='.h5').name
    return hdf5.make_group(name, 'w')


def test_make_group(test_filename):

    g = hdf5.hdfHandler(test_filename, 'w')
    assert g.filename == test_filename

    g1 = hdf5.hdfHandler(g)

    g2 = hdf5.hdfHandler(g, key="group1")

    assert g2.groupname == "group1"

    with pytest.raises(ValueError):
        g2 = hdf5.hdfHandler(g, 'w')

    g3 = hdf5.hdfHandler(test_filename, 'a', key='group2')
    hdf5.hdfHandler(test_filename, 'a', key='group1/group2')

    assert g3.filename == test_filename

    with pytest.raises(TypeError):
        hdf5.hdfHandler(1)


def test_set_type_tag(test_filename):

    g = hdf5.hdfHandler(test_filename, 'w')
    a = np.arange(10)
    g.set_type_tag(type(a))

    assert g.attrs['type_tag'] == "numpy.ndarray"


def test_validate_tag(test_filename):

    g = hdf5.hdfHandler(test_filename, 'w')
    g1 = hdf5.hdfHandler(g, key="group1")

    a = np.arange(10)
    g.set_type_tag(type(a))

    g.validate_tag(type(a), 'strict')
    g1.validate_tag(type(a), 'nocheck')

    class ndarray_subclass(np.ndarray):
        pass

    g.validate_tag(ndarray_subclass, 'weak')


    with pytest.raises(hdf5.HDF5TagError):
        g1.validate_tag(type(a), 'strict')

    with pytest.raises(hdf5.HDF5TagError):
        g.validate_tag(ndarray_subclass, 'strict')

    with pytest.raises(hdf5.HDF5TagError):
        g1.validate_tag(h5py.File, 'strict')

    with pytest.raises(hdf5.HDF5TagError):
        g1.validate_tag(h5py.File, 'strict')

    with pytest.raises(hdf5.HDF5TagError):
        g1.validate_tag(type(a), 'strict')

    with pytest.raises(hdf5.HDF5TagError):
        g1.validate_tag(type(a), 'warn')

    with pytest.warns(hdf5.HDF5TagWarning):
        g.validate_tag(h5py.File, 'warn')

    with pytest.raises(hdf5.HDF5TagError):
        g.validate_tag(h5py.File, 'weak')

    g2 = hdf5.hdfHandler(g, key="group2")
    g2.attrs['type_tag'] = "numpy.nddarray"
    with pytest.raises(hdf5.HDF5TagError):
        g2.validate_tag(np.ndarray, 'weak')

    g2.attrs['type_tag'] = "numpyy.ndarray"
    with pytest.raises(hdf5.HDF5TagError):
        g2.validate_tag(np.ndarray, 'weak')
        

def test_access_group(test_filename):

    g1 = hdf5.hdfHandler(test_filename, 'w', "group1")

    hdf5.hdfHandler(g1.filename, 'r')
    hdf5.hdfHandler(g1.filename, 'r', "group1")

def test_getitem(test_filename):
    g1 = hdf5.hdfHandler(test_filename, 'w', "group1/group2")

    g2 =  hdf5.hdfHandler(test_filename, 'r')

    assert g2['group1'].groupname == 'group1'

    assert g2['group1']['group2'].groupname == 'group2'

def test_dataset(test_filename):
    g1 = hdf5.hdfHandler(test_filename, 'w', "group1/group2")

    a = np.random.randn(100,200,300)
    g1.create_dataset('data',
                      data=a)

    a1 = g1.read_dataset('data')

    assert np.array_equal(a, a1)

    g1.create_dataset('data1', data=a, compression='gzip')

    a2 = g1.read_dataset('data1')

    assert np.array_equal(a, a2)

def test_dataset_str_array(test_filename):
    g1 = hdf5.hdfHandler(test_filename, 'w', "group1/group2")

    a = np.array(['a', 'bb', 'ccc', 'dddd'], dtype=np.string_)

    g1.create_dataset('data', data=a)
    a1 = g1.read_dataset('data')

    assert (a == a1).all()

