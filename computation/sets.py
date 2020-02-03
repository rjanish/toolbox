""" operations on sets """


import numpy as np


def cartesian_product(arrays, output_dtype=None):
    """
    Generate a Cartesian product of input arrays.

    Args:
    arrays - list
        List of arrays with which to form the Cartesian
        product.  Input arrays will be treated as 1D.
    output_dtype - numpy data type
        The dtype of the output array. If unspecified, will be
        taken to match the dtype of the first element of arrays.

    Returns:
    output - 2D array
        Array of shape (arrays[0].size*arrays[1].size..., len(arrays))
        containing the Cartesian product of the input arrays.

    Examples:
    >>> cartesian_product(([1, 2, 3, 4], [5.0], [6.2, 7.8]))
    array([[1, 5, 6],
           [1, 5, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 5, 6],
           [3, 5, 7],
           [4, 5, 6],
           [4, 5, 7]])
    >>> cartesian_product([['boil', 'mash', 'stew'],
                           ['potatoes', 'rabbit']], dtype='|S8')
    array([['boil', 'potatoes'],
           ['boil', 'rabbit'],
           ['mash', 'potatoes'],
           ['mash', 'rabbit'],
           ['stew', 'potatoes'],
           ['stew', 'rabbit']], dtype='|S8')
    """
    # make output container
    arrays = [np.array(x).flatten() for x in arrays]
    if output_dtype is None:
        output_dtype = arrays[0].dtype
    num_output_elements = np.prod([x.size for x in arrays])
    output_element_size = len(arrays)
    output_shape = (num_output_elements, output_element_size)
    output = np.zeros(output_shape, dtype=output_dtype)
    # form Cartesian product
    repetitions = num_output_elements/arrays[0].size
        # the number of times that each element of arrays[0] will
        # appear as the first entry in an element of the output
    output[:, 0] = np.repeat(arrays[0], repetitions)
        # for each block of output elements with identical first
        # entry, the remaining pattern of entries within each block
        # will be identical to that of any other block and is just the
        # Cartesian produced of the remaining arrays: recursion!
    arrays_remaining = bool(arrays[1:])
    if arrays_remaining:
        sub_output = cartesian_product(arrays[1:], output_dtype=output_dtype)
        for block_number in xrange(arrays[0].size):
            block_start = block_number*repetitions
            block_end = block_start + repetitions
            output[block_start:block_end, 1:] = sub_output
    return output