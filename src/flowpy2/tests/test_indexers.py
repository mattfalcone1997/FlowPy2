import numpy as np
from flowpy2.indexers import (CompIndexer,
                              TimeIndexer,
                              IndexInitialisationError,
                              RoundTruncationWarning,
                              DtypeTruncationWarning)
import pytest
from flowpy2.io import hdf5, netcdf
from test_hdf5 import test_filename


def test_CompIndexer_initialisation():
    # test valid inputs with different containers
    input_list = ['x', 'y', 'z']
    input_ndarray = np.array(input_list, dtype=object)
    input_tuple = tuple(input_list)

    assert CompIndexer(input_list)._index == input_list, \
        "Check initialisation with list of str"

    assert CompIndexer(input_ndarray)._index == input_list, \
        "Check initialisation with ndarray of str"

    assert CompIndexer(input_tuple)._index == input_list, \
        "Check initialisation with tuple of str"

    # test containers with invalid inputs
    invalid_input = [1, 'y', 'z']
    with pytest.raises(IndexInitialisationError):
        CompIndexer(invalid_input)

    assert CompIndexer(input_list) == CompIndexer(input_list), \
        "Check __eq__ works"
    assert input_list != CompIndexer(input_list), \
        "Check no equal works"

    # test invalid containers


def test_CompIndexer_get():

    index = CompIndexer(['x', 'y', 'z'])
    # test valid key
    assert index.get('x') == 0, "Checking index getter"
    assert index.get('y') == 1, "Checking index getter"
    assert index.get('z') == 2, "Checking index getter"

    assert index.get(['x', 'y']) == [0, 1], "Checking index list getter"
    assert index.get(slice('x', 'y')) == slice(0, 2), \
        "Checking index slice getter"

    # test invalid key
    with pytest.raises(KeyError):
        index.get('a')

    # test get_other
    assert index.get_other('x') == [
        1, 2], "Checking index getting inverse"


def test_CompIndexer_modifiers():
    index = CompIndexer(['x', 'y', 'z'])

    index.append('a')
    assert index.get('a') == 3, "Checking index append getter"

    with pytest.raises(TypeError):
        index.append(1)

    with pytest.raises(ValueError):
        index.append('a')

    other_index = ['b', 'c']
    index.extend(other_index)
    assert index == CompIndexer(['x', 'y', 'z', 'a', 'b', 'c']), \
        "Checking index extend"

    with pytest.raises(ValueError):
        index.extend(other_index)

    index.remove('x')
    assert index == CompIndexer(['y', 'z', 'a', 'b', 'c']), \
        "Checking index remove 1 element"
    index.remove(['y', 'z'])
    assert index == CompIndexer(['a', 'b', 'c']), \
        "Checking index remove many element"

    with pytest.raises(ValueError):
        index.remove('x')

    index.replace_keys({'a': 'x',
                        'b': 'y',
                        'c': 'z'})

    assert index == CompIndexer(['x', 'y', 'z']), \
        "Checking valid key replacement"

    with pytest.raises(KeyError):
        index.replace_keys({'a': 'x'})

    with pytest.raises(TypeError):
        index.replace_keys({'x': 1})

    with pytest.raises(KeyError):
        index.replace_keys({'x': 'y'})


def test_CompIndexer_to_hdf(test_filename):
    index = CompIndexer(['x', 'y', 'z'])
    f = hdf5.hdfHandler(test_filename, 'w')

    index.to_hdf(f, "index")

    index1 = CompIndexer.from_hdf(f, "index")

    assert index1 == index


def test_TimeIndexer_init():
    data = np.arange(100., dtype='f8')

    indexer = TimeIndexer(data)

    assert indexer._decimals is None, "Check default decimals"
    assert indexer._dtype == np.float64, "Check default type"
    assert indexer._index.dtype.type == np.float64, "Check user specified decimals"

    data1 = data + 0.000000001
    with pytest.warns(DtypeTruncationWarning):
        indexer = TimeIndexer(data1, dtype='f4')

    data1 = data + 0.01
    with pytest.warns(RoundTruncationWarning):
        indexer = TimeIndexer(data1, decimals_round=1)

    assert indexer._decimals is 1, "Check user specified decimals"
    assert np.array_equal(indexer._index, data.astype('f4'))

    indexer = TimeIndexer(data, dtype='f4')
    assert indexer._index.dtype.type == np.float32, "Check user specified decimals"


def test_TimeIndexer_get():
    data = np.arange(100., dtype='f8') + 0.01

    indexer = TimeIndexer(data)

    assert indexer.get(0.01) == 0, \
        "Check get with element with default decimals"

    assert indexer.get([0.01, 4.01, 5.01]) == [0, 4, 5], \
        "Check get with list with default decimals"

    assert indexer.get(slice(0.01, 70.01)) == slice(0, 71), \
        "Check get with slice with default decimals"

    indexer = TimeIndexer(data, decimals_round=4)
    assert indexer.get(0.01001) == 0, \
        "Check get with element with default decimals with extral decimal place"

    assert indexer.get([0.01001, 4.01001, 5.01001]) == [0, 4, 5], \
        "Check get with list with default decimals with extral decimal place"

    assert indexer.get(slice(0.01001, 70.01001)) == slice(0, 71), \
        "Check get with slice with default decimals with extral decimal place"

    with pytest.raises(KeyError):
        indexer.get(0.011)

    with pytest.raises(KeyError):
        indexer.get([0.011, 4.01001, 5.01001])

    with pytest.raises(KeyError):
        indexer.get(slice(0.011, 70.011))


def test_TimeIndex_get_other():
    data = np.arange(10., dtype='f8')
    indexer = TimeIndexer(data)

    assert indexer.get_other(0) == list(range(1, 10)), \
        "Test get_other with integer"

    assert indexer.get_other([0, 1, 2, 3]) == list(range(4, 10)), \
        "Test get_other with list"

    assert indexer.get_other(slice(0, 3)) == list(range(4, 10)), \
        "Test get_other with slice"

    with pytest.raises(KeyError):
        indexer.get_other(10)


def test_TimeIndexer_append():
    data = np.arange(10., dtype='f8')
    indexer = TimeIndexer(data)

    indexer.append(10)

    assert indexer == TimeIndexer(np.arange(11)), "Test append"

    with pytest.raises(TypeError):
        indexer.append('a')

    with pytest.raises(ValueError):
        indexer.append(10)


def test_TimeIndexer_extend():
    data = np.arange(10., dtype='f8')
    indexer = TimeIndexer(data)

    test_indexer = TimeIndexer(np.arange(20))
    indexer.extend(np.arange(10, 20))
    assert indexer == test_indexer, "Test extend"

    with pytest.raises(ValueError):
        indexer.extend(np.arange(19, 29))


def test_TimeIndexer_concat_indexers():
    data1 = np.arange(0, 20, 2)
    data2 = np.arange(1, 21, 2)
    index1 = TimeIndexer(data1)
    index2 = TimeIndexer(data2)

    sort1, sort2 = TimeIndexer.concat_indexers(index1, index2)

    assert sort1 == list(range(0, 20, 2)), "Check first list list index"
    assert sort2 == list(range(1, 21, 2)), "Check second list list index"


def test_TimeIndexer_ufunc():
    data = np.arange(0, 20)
    index = TimeIndexer(data)

    assert (index + 2) == TimeIndexer(data+2), "Check arithmetic with scalar"

    assert (index + index) == TimeIndexer(data+data), \
        "Check arithmetic with another indexer"

    assert np.sqrt(index) == TimeIndexer(np.sqrt(data)), "Check sqrt operation"


def test_TimeIndexer_to_hdf(test_filename):
    index = TimeIndexer(np.arange(0, 20))
    f = hdf5.hdfHandler(test_filename, 'w')

    index.to_hdf(f, "index")

    index1 = TimeIndexer.from_hdf(f, "index")

    assert index1 == index


def test_TimeIndexer_to_netcdf(test_filename):
    index = TimeIndexer(np.arange(0, 20))
    f = netcdf.make_dataset(test_filename, 'w')

    index.to_netcdf(f)
    f.close()
    f = netcdf.access_dataset(test_filename)
    index1 = TimeIndexer.from_netcdf(f)

    assert index1 == index
