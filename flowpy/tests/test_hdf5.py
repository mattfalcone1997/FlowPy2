from flowpy.io import hdf5
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

    g = hdf5.make_group(test_filename, 'w')
    assert isinstance(g, h5py.File)
    assert g.filename == test_filename

    g1 = hdf5.make_group(g)

    assert g is g1

    g2 = hdf5.make_group(g, key="group1")
    assert isinstance(g2, h5py.Group)
    assert g2.name == "/group1"

    with pytest.raises(ValueError):
        g2 = hdf5.make_group(g, 'w')

    g.close()

    g3 = hdf5.make_group(test_filename, 'a', key='group2')

    assert isinstance(g3, h5py.Group)
    assert g3.file.filename == test_filename

    g3.file.close()

    with pytest.raises(ValueError):
        hdf5.make_group(test_filename)

    with pytest.raises(TypeError):
        hdf5.make_group(1)


def test_set_type_tag(test_filename):

    g = hdf5.make_group(test_filename, 'w')
    a = np.arange(10)
    hdf5.set_type_tag(type(a), g)

    assert g.attrs['type_tag'] == "numpy.ndarray"


def test_validate_tag(test_filename):

    g = hdf5.make_group(test_filename, 'w')
    g1 = hdf5.make_group(g, key="group1")

    a = np.arange(10)
    hdf5.set_type_tag(type(a), g)

    hdf5.validate_tag(type(a), g, 'strict')
    hdf5.validate_tag(type(a), g1, 'nocheck')

    with pytest.raises(hdf5.HDF5TagError):
        hdf5.validate_tag(type(a), g1, 'strict')

    with pytest.raises(hdf5.HDF5TagError):
        hdf5.validate_tag(h5py.File, g1, 'strict')

    with pytest.raises(hdf5.HDF5TagError):
        hdf5.validate_tag(h5py.File, g1, 'strict')

    with pytest.raises(hdf5.HDF5TagError):
        hdf5.validate_tag(type(a), g1, 'strict')

    with pytest.raises(hdf5.HDF5TagError):
        hdf5.validate_tag(type(a), g1, 'warn')

    with pytest.warns(hdf5.HDF5TagWarning):
        hdf5.validate_tag(h5py.File, g, 'warn')


def test_access_group(test_filename):

    g1 = hdf5.make_group(test_filename, 'w', "group1")

    hdf5.access_group(g1.file.filename)
    hdf5.access_group(g1.file.filename, "group1")
    g2 = hdf5.access_group(g1)
    g3 = hdf5.access_group(g1.file, "group1")

    assert g2 == g1
    assert g2 == g3
