""" basic arithmetic operations """


import numpy as np


def floor_even(x):
    """ Returns largest even integer less than or equal to x """
    if not np.all(np.isfinite(x)):
        warnings.warn("Input is not finite: bizarre castings will occur")
    floor = np.floor(x).astype(int)
    ones = np.ones(floor.shape, dtype=floor.dtype)
    return np.bitwise_and(floor, ~ones) # this clears the ones bit


def floor_odd(x):
    """ Returns largest odd integer less than or equal to x """
    return floor_even(x + 1) - 1


def modf_even(x):
    """
    Returns n, r such that n + r = x and n is the largest
    even integer less than or equal to x (so 0 <= r < 2).
    """
    n = floor_even(x)
    r = x - n
    return n, r


def modf_odd(x):
    """
    Returns n, r such that n + r = x and n is the largest
    odd integer less than or equal to x (so 0 <= r < 2).
    """
    n = floor_odd(x)
    r = x - n
    return n, r