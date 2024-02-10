# cython: language=3, boundscheck=False, wraparound=False

cimport numpy as cnp
import numpy as np

from cython.parallel import parallel,prange
import cython

ctypedef fused float_type:
    double
    float

def gradient1_order2_dx(float_type[:,::1] array, float_type dx):
    gradient = np.zeros_like(array)
    
    cdef float_type idx = 1./dx
    cdef float_type[:,::1] grad_view = gradient
    cdef int i, j, n, m

    m = grad_view.shape[0]
    n = grad_view.shape[1]
    for i in prange(m, nogil=True):

        grad_view[i, 0] = idx * \
            (-1.5*array[i, 0] + 2.*array[i, 1] - 0.5*array[i, 2])

        for j in prange(1, n-1):
            grad_view[i, j] = 0.5*idx*(- array[i, j-1] + array[i, j+1])

        grad_view[i, n-1] = idx * \
            (0.5*array[i, n-3] - 2.*array[i, n-2] + 1.5*array[i, n-1])

    return gradient

def gradient1_order2_var_x(float_type[:,::1] array,float_type[::1] x):
    gradient = np.zeros_like(array)

    cdef float_type[:,::1] grad_view = gradient
    cdef float_type dx1, dx2
    cdef float_type a, b, c
    cdef int i, j, n, m

    m = grad_view.shape[0]
    n = grad_view.shape[1]

    for i in prange(m, nogil=True):
        dx1 = x[1] - x[0]
        dx2 = x[2] - x[1]

        a = -(2.*dx1 + dx2)/(dx1*(dx1 + dx2))
        b = (dx1 + dx2)/(dx1*dx2)
        c = -dx1/(dx2*(dx1 + dx2))

        grad_view[i, 0] = a*array[i, 0] + b*array[i, 1] + c*array[i, 2]

        for j in prange(1, n-1):
            dx1 = x[j] - x[j-1]
            dx2 = x[j+1] - x[j]

            a = -dx2/(dx1*(dx1 + dx2))
            b = (-dx1 + dx2)/(dx1*dx2)
            c = dx1/(dx2*(dx1 + dx2))
            grad_view[i, j] = a*array[i, j-1] + b*array[i, j] + c*array[i, j+1]

        dx1 = x[n-2] - x[n-3]
        dx2 = x[n-1] - x[n-2]

        a = dx2/(dx1*(dx1 + dx2))
        b = -(dx1 + dx2)/(dx1*dx2)
        c = (2*dx2 + dx1)/(dx2*(dx1 + dx2))

        grad_view[i, n-1] = a*array[i, n-3] + b*array[i, n-2] + c*array[i, n-1]

    return gradient


def gradient2_order2_dx(float_type[:,::1] array, float_type dx):
    
    gradient = np.zeros_like(array)
    
    cdef float_type idx2 = 1./(dx*dx)
    cdef float_type a, b, c
    cdef float_type[:,::1] grad_view = gradient
    cdef int i, j, n, m

    m = grad_view.shape[0]
    n = grad_view.shape[1]

    a = idx2
    b = -2.*idx2
    c = idx2

    for i in prange(m, nogil=True):
        grad_view[i, 0] = a*array[i, 0] + b*array[i, 1] + c*array[i, 2]

        for j in prange(1, n-1):
            grad_view[i, j] = a*array[i, j-1] + b*array[i, j] + c*array[i, j+1]

        grad_view[i, n-1] = a*array[i, n-3] + b*array[i, n-2] + c*array[i, n-1]

    return gradient

def gradient2_order2_var_x(float_type[:,::1] array, float_type[::1] x):
    
    gradient = np.zeros_like(array)

    cdef float_type[:,::1] grad_view = gradient
    cdef float_type dx1, dx2
    cdef float_type a, b, c
    cdef int i, j, n, m

    m = grad_view.shape[0]
    n = grad_view.shape[1]

    for i in prange(m, nogil=True):
        dx1 = x[1] - x[0]
        dx2 = x[2] - x[1]

        a = 2./(dx1*(dx1 + dx2))
        b = -2./(dx1*dx2)
        c = 2./(dx2*(dx1 + dx2))

        grad_view[i, 0] = a*array[i, 0] + b*array[i, 1] + c*array[i, 2]

        for j in prange(1, n-1):
            dx1 = x[j] - x[j-1]
            dx2 = x[j+1] - x[j]

            a = 2./(dx1*(dx1 + dx2))
            b = -2/(dx1*dx2)
            c = 2/(dx2*(dx1 + dx2))
            grad_view[i, j] = a*array[i, j-1] + b*array[i, j] + c*array[i, j+1]

        dx1 = x[n-3] - x[n-2]
        dx2 = x[n-2] - x[n-1]

        a = 2./(dx1*(dx1 + dx2))
        b = -2/(dx1*dx2)
        c = 2/(dx2*(dx1 + dx2))

        grad_view[i, n-1] = a*array[i, n-3] + b*array[i, n-2] + c*array[i, n-1]
        
    return gradient    