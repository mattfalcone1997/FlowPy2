import numpy as np
import logging

import unittest


# class TestGroup(unittest.TestCase):
#     def assertObjectArrayEqual(self, array1, array2, *args, **kwargs):
#         for a1, a2 in zip(array1, array2):
#             self.assertTrue(np.array_equal(a1, a2), *args, **kwargs)


# class TestGroupInitialisation(TestGroup):
#     def test_basic(self):
#         self.assertTrue(npGroupArray._array_class, np.ndarray)
#         self.assertTrue(npGroupArray._array_creator, np.array)

#         array = np.array(
#             [np.array([1., 2., 3.]), np.array([1., 2.])], dtype=object)
#         group = npGroupArray(array)

#         self.assertObjectArrayEqual(array[0], group[0])
#         self.assertObjectArrayEqual(array[1], group[1])

#         self.assertTrue(np.shares_memory(array[0], group[0]))

#     def test_dtype(self):
#         array = [np.array([1., 2., 3.]), np.array([1., 2.])]
#         group = npGroupArray(array, dtype='f4')

#         self.assertEqual(group.dtype, np.dtype('f4'))

#     def test_initcopy(self):

#         array = [np.array([1., 2., 3.]), np.array([1., 2.])]
#         group = npGroupArray(array, copy=True)
#         self.assertTrue(~np.shares_memory(array[0], group[0]))


# class TestGroupItems(TestGroup):
#     def setUp(self) -> None:
#         self._array = np.array(
#             [np.array([1., 2., 3.]), np.array([1., 2.])], dtype=object)
#         self._group = npGroupArray(self._array)

#     def test_getitem(self):

#         for i in range(len(self._array)):
#             self.assertObjectArrayEqual(self._group[i], self._array[i],
#                                         msg="Test indexing by integer gives the same results")

#         self.assertObjectArrayEqual(self._group[:].data, self._array[:],
#                                     msg="Test indexing by slice gives the same results")

#         self.assertObjectArrayEqual(self._group[[0, 1]].data, self._array[[0, 1]],
#                                     msg="Test indexing by list gives the same results")

#         self.assertObjectArrayEqual(self._group[[True, False]].data, self._array[[True, False]],
#                                     msg="Test indexing by boolean gives the same results")

#     def test_setitem(self):
#         ref = self._array.copy()
#         ref[1] = self._array[0]

#         self._group[1] = ref[1]

#         self.assertObjectArrayEqual(self._group.data, ref,
#                                     msg="Test setitem")

#         with self.assertRaises(TypeError):
#             self._group[1] = 1

#         with self.assertRaises(ValueError):
#             self._group[:] = 1

#         with self.assertRaises(TypeError):
#             self._group[:] = [1, 2]

#     def test_iter(self):
#         for val1, val2 in zip(self._group, self._array):
#             self.assertObjectArrayEqual(val1, val2,
#                                         msg="Check iterator works correctly")

#         self.assertEqual(len(self._array), len(self._group))


# class TestGroupUfunc(TestGroup):
#     def setUp(self) -> None:
#         self._array = np.array(
#             [np.array([1., 2., 3.], dtype=np.int32), np.array([1., 2.], dtype=np.int32)], dtype=object)
#         self._group = npGroupArray(self._array)

#     def test_ufuncs(self):
#         # testing unary
#         self.assertObjectArrayEqual((-self._group).data,
#                                     -self._array,
#                                     msg="Test unary negation")

#         # testing binary with scalars
#         # check add
#         add = self._group + 2
#         self.assertObjectArrayEqual(add.data, self._array+2,
#                                     msg="Test addition with scalar group on the left")

#         # check radd
#         radd = 2+self._group
#         self.assertObjectArrayEqual(radd.data, self._array+2,
#                                     msg="Test addition with scalar group on the right")

#         # check binary with another group
#         mul = self._group*self._group
#         self.assertObjectArrayEqual(mul.data, self._array*self._array,
#                                     msg="Test addition with scalar group on the right")

#         # check binary with non broadcastable shapes fails
#         with self.assertRaises(ValueError):
#             self._group*self._group[::-1]

#         self.assertObjectArrayEqual(np.sqrt(self._group).data, np.array([np.sqrt(a) for a in self._array], dtype=object),
#                                     "Testing the np.sqrt")

#         new_array = np.array(
#             [np.array([1., 2., 3.]), np.array([1., 2., 3.])], dtype=object)
#         new_group = npGroupArray(new_array)
#         self.assertObjectArrayEqual(new_group*new_array[0], np.array([a*new_array[0] for a in new_array], dtype=object),
#                                     "Testing the broadcasting")


# def main():
#     logger.setLevel(logging.DEBUG)
#     suite = unittest.TestSuite()

#     suite.addTest(TestGroupInitialisation())
#     suite.addTest(TestGroupItems())
#     suite.addTest(TestGroupUfunc())
#     unittest.main()


# if __name__ == '__main__':
#     main()
