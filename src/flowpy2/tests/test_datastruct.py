import numpy as np
from flowpy2.datastruct import DataStruct
# from flowpy2.arrays_old import GroupArray

import pytest
from test_hdf5 import test_filename


def return_data_index():
    array = [np.arange(100, dtype='f8'),
             np.arange(50, dtype='f8'),
             np.arange(200, dtype='f8')]
    index = ['x', 'y', 'z']

    return array, index


def return_dstruct():
    array = [np.arange(100, dtype='f8'),
             np.arange(50, dtype='f8'),
             np.arange(200, dtype='f8')]
    index = ['x', 'y', 'z']

    return DataStruct(array, index=index)


test_dstruct = pytest.fixture(return_dstruct)


def check_data(ds1: DataStruct):
    array, index = return_data_index()

    for d1, d2 in zip(ds1._data, array):
        if not np.array_equal(d1, d2):
            return False

    for i, k in enumerate(ds1.index):
        if index[i] != k:
            return False

    return True


def test_array_ini():

    data, _ = return_data_index()
    ds = return_dstruct()
    check_data(ds)

    with pytest.raises(ValueError):
        DataStruct(data,
                   index=['x', 'x', 'y'])


def test_dict_ini():
    array, index = return_data_index()

    ds_dict = dict(zip(index, array))

    ds = DataStruct(ds_dict)

    check_data(ds)

    ds_dict2 = ds.to_dict()

    for k1, k2 in zip(ds_dict.keys(), ds_dict2.keys()):
        assert k1 == k2, "check keys of to_dict"

    for v1, v2 in zip(ds_dict.values(), ds_dict2.values()):
        assert np.array_equal(v1, v2), "check values of to_dict"


def test_getitem():
    array, index = return_data_index()
    data = return_dstruct()
    a = data['x']

    assert np.array_equal(a, array[0]), "check integer index"

    sub_ds = DataStruct(array[:2], index=index[:2])
    sub_ds2 = DataStruct(array[1:], index=index[1:])
    sub_ds3 = DataStruct([array[0]], index=[index[0]])

    assert np.array_equal(
        sub_ds, data['x':'y']), "check slice index with start and stop"
    assert np.array_equal(sub_ds, data[:'y']), "check slice index with stop"
    assert np.array_equal(sub_ds2, data['y':]), "check slice index with start"
    assert np.array_equal(
        sub_ds3, data['x':'x']), "check slice index with start"

    with pytest.raises(ValueError):
        data['x':'y': 1]

    with pytest.raises(KeyError):
        data['a']
    with pytest.raises(KeyError):
        data['x':'a']
    with pytest.raises(KeyError):
        data['a':'x']

    assert np.array_equal(sub_ds, data[['x', 'y']]), "check slice index"
    assert np.array_equal(sub_ds3, data[['x']]), "check slice index with start"

    with pytest.raises(KeyError):
        data[['a']]

    with pytest.raises(KeyError):
        data[['x', 'a']]

    with pytest.raises(KeyError):
        data[('x', 'a')]


def test_iter():
    array, index = return_data_index()
    data = return_dstruct()

    for i, (k, v) in enumerate(data):
        assert k == index[i], "Check key from iterator"
        assert np.array_equal(v, array[i]), "Check value from iterator"


def test_copy(test_dstruct):
    copy_test = test_dstruct.copy()
    assert copy_test == test_dstruct
    assert copy_test._data is not test_dstruct._data, \
        "Check it has actually copied"
    for d1, d2 in zip(test_dstruct._data, copy_test._data):
        assert d1 is not d2, "elements not the same object"
        assert np.array_equal(d1, d2), "check numeric values are the same"


def test_setitem():
    array, index = return_data_index()
    data = return_dstruct()
    data1 = data.copy()

    data1[:] = array
    assert np.array_equal(data1, data), "full slice setitem"
    data1[:'y'] = array[:2]
    assert np.array_equal(data1, data), "full slice setitem"

    data1['x'] = array[2]
    assert np.array_equal(data1['x'], array[2]), "single element setitem"

    data1[['x', 'y']] = array[1:]

    assert np.array_equal(data1['x'], array[1]), "list setitem"

    assert np.array_equal(data1['y'], array[2]), "list setitem"

    with pytest.raises(KeyError):
        data1[['x', 'a']] = array[1:]

    with pytest.raises(KeyError):
        data1['x':'a'] = array[1:]

    with pytest.raises(KeyError):
        data1['a'] = array[0]


def test_delitem():
    data = return_dstruct().copy()

    data1 = data.copy()
    del data1['x']
    assert np.array_equal(
        data1, data[['y', 'z']]), "test delitem single element"

    data1 = data.copy()
    del data1[['x', 'y']]
    assert np.array_equal(data1, data[['z']]), "test delitem list"

    data1 = data.copy()
    del data1['x':'y']
    assert np.array_equal(data1, data[['z']]), "test delitem slice"

    data1 = data.copy()
    with pytest.raises(KeyError):
        del data1['a']

    with pytest.raises(KeyError):
        del data1['a':'x']

    with pytest.raises(KeyError):
        del data1[['a', 'x']]


def test_concat():
    data = return_dstruct().copy()

    data1 = data.copy()
    data1.index.replace_keys({'x': 'a',
                              'y': 'b',
                              'z': 'c'})

    # test valid concat
    data1.concat(data)

    # test invalid concat
    with pytest.raises(ValueError):
        data1.concat(data1)


def test_remove():
    data = return_dstruct().copy()

    # remove element
    ref = data[['y', 'z']]
    data1 = data.copy()
    data1.remove('x')
    assert np.array_equal(ref, data1), "Test remove element"
    # remove slice
    data1 = data.copy()
    data1.remove(slice('x', 'x'))
    assert np.array_equal(ref, data1), "Test remove slice"

    # remove list
    data1 = data.copy()
    data1.remove(['x'])
    assert np.array_equal(ref, data1), "Test remove list"


def test_binary():
    array, index = return_data_index()
    data = return_dstruct().copy()

    data1 = data*data

    data2 = data + 2

    for i, (k, v) in enumerate(data1):
        assert np.array_equal(
            v, array[i]*array[i]), "Test multiplication of datastructs"
        assert k == index[i], "Test multiplication of datastructs"

    for i, (k, v) in enumerate(data2):
        assert np.array_equal(
            v, array[i]+2), "Test multiplication of datastructs"
        assert k == index[i], "Test multiplication of datastructs"


def test_unary():
    array, index = return_data_index()
    data = return_dstruct().copy()

    data = -data
    for i, (k, v) in enumerate(data):
        assert np.array_equal(
            v, -array[i]), "Test multiplication of datastructs"
        assert k == index[i], "Test multiplication of datastructs"


def test_sqrt():
    array, index = return_data_index()
    data = return_dstruct().copy()
    data = np.sqrt(data)
    for i, (k, v) in enumerate(data):
        assert np.array_equal(v, np.sqrt(
            array[i])), "Test multiplication of datastructs"
        assert k == index[i], "Test multiplication of datastructs"


def test_hdf(test_filename):
    dstruct = return_dstruct()

    dstruct.to_hdf(test_filename, 'w')

    dstruct2 = dstruct.__class__.from_hdf(test_filename)

    assert dstruct == dstruct2
