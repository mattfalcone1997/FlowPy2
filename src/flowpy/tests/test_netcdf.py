from flowpy.io import netcdf
import netCDF4
import pytest
from tempfile import NamedTemporaryFile
import numpy as np


@pytest.fixture(autouse=True)
def test_filename():
    return NamedTemporaryFile(suffix='.nc').name


@pytest.fixture()
def test_group():
    name = NamedTemporaryFile(suffix='.nc').name
    return netcdf.make_dataset(name, 'w')


def walktree(top):
    yield top.groups.values()
    for value in top.groups.values():
        yield from walktree(value)


def test_make_dataset(test_filename):

    g = netcdf.make_dataset(test_filename, 'w')
    assert isinstance(g, netCDF4.Dataset)
    assert g.filepath() == test_filename

    g1 = netcdf.make_dataset(g)

    assert g is g1

    g2 = netcdf.make_dataset(g, key="group1")
    assert isinstance(g2, netCDF4.Group)
    assert g2.name == "group1"

    with pytest.raises(ValueError):
        netcdf.make_dataset(g, 'w')

    g.close()

    g3 = netcdf.make_dataset(test_filename, 'a', key='group2')

    assert isinstance(g3, netCDF4.Group)
    assert g3.filepath() == test_filename

    g3.parent.close()

    with pytest.raises(ValueError):
        netcdf.make_dataset(test_filename)

    with pytest.raises(TypeError):
        netcdf.make_dataset(1)


def test_set_type_tag(test_filename):

    g = netcdf.make_dataset(test_filename, 'w')
    a = np.arange(10)
    netcdf.set_type_tag(type(a), g)

    netcdf.set_type_tag(netCDF4.Dataset, g, "other_tag")

    assert g.type_tag == "numpy.ndarray"
    assert g.other_tag == "netCDF4._netCDF4.Dataset"


def test_validate_tag(test_filename):

    g = netcdf.make_dataset(test_filename, 'w')
    g1 = netcdf.make_dataset(g, key="group1")

    a = np.arange(10)
    netcdf.set_type_tag(type(a), g)
    netcdf.set_type_tag(netCDF4.Dataset, g, "other_tag")

    netcdf.validate_tag(type(a), g, 'strict')
    netcdf.validate_tag(netCDF4.Dataset, g, 'strict', "other_tag")

    netcdf.validate_tag(type(a), g1, 'nocheck')

    with pytest.raises(netcdf.netCDF4TagError):
        netcdf.validate_tag(type(a), g1, 'strict')

    with pytest.raises(netcdf.netCDF4TagError):
        netcdf.validate_tag(netCDF4.Dataset, g1, 'strict')

    with pytest.raises(netcdf.netCDF4TagError):
        netcdf.validate_tag(netCDF4.Dataset, g1, 'strict')

    with pytest.raises(netcdf.netCDF4TagError):
        netcdf.validate_tag(type(a), g1, 'strict')

    with pytest.raises(netcdf.netCDF4TagError):
        netcdf.validate_tag(type(a), g1, 'warn')

    with pytest.warns(netcdf.netCDF4TagWarning):
        netcdf.validate_tag(netCDF4.Dataset, g, 'warn')


def test_access_group(test_filename):

    g1 = netcdf.make_dataset(test_filename, 'w', "group1")
    g1.parent.close()

    netcdf.access_dataset(test_filename)
    netcdf.access_dataset(test_filename, "group1")

    g2 = netcdf.access_dataset(g1)
    g3 = netcdf.access_dataset(g1.filepath(), "group1")
