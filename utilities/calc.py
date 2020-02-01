""" Miscellaneous mathematical functions """


import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt

from utilities.local_warnings import warn_untested

def gaussian(x, mu, sigma):
    """
    Returns the normalized Gaussian G(x) evaluated element-wise on x.
    G(x) has a mean mu and standard deviation sigma, and is normalized
    to have a continuous integral from x=-inf to x=inf of 1.
    """
    x = np.asarray(x)
    mu, sigma = float(mu), float(sigma)
    arg = (x - mu)/sigma
    norm = sigma*np.sqrt((2*np.pi))
    return np.exp(-0.5*(arg**2))/norm


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


def in_linear_interval(array, interval):
    """
    Indicate the array elements that are contained in interval.

    Args:
    array - arraylike
        Array to test for containment, assumed to be numeric-valued.
    interval - 1D, 2-element arraylike
        A 1D interval expressed as [start, end], assumed start <= end.

    Returns:
    contained - boolean arraylike
        A boolean array of the same shape as the input array, valued
        True if the the input array values are in the passed interval.
    """
    array = np.asarray(array, dtype=float)
    start, end = np.asarray(interval, dtype=float)
    if start > end:
        raise ValueError("Invalid interval: ({}, {})".format(start, end))
    return (start < array) & (array < end)


def in_periodic_interval(array, interval, period):
    """
    Indicate the array elements that are contained in the passed
    interval, taking into account the given periodicity.

    For a passed interval = [a, b], the interval is defied as the
    region starting at a and continuing in the direction of explicitly
    increasing values to b, all subject to the passed identification
    x ~ x + period.  I.e., if a and b are interpreted as angles on the
    circle, the interval is the arc from a to b counterclockwise.
    There is no restriction a < b: the intervals [a, b] and [b, a] are
    both sensible, representing the two paths along the circle
    connecting points a and b.

    Args:
    array - arraylike
        Array to test for containment, assumed to be numeric-valued.
    interval - 1D, 2-element arraylike
        A 1D interval expressed as [start, end], see convention above.

    Returns:
    contained - boolean arraylike
        A boolean array of the same shape as the input array, valued
        True if the the input array values are in the passed interval.
    """
    period = float(period)
    array = np.asarray(array, dtype=float)
    start, end = np.asarray(interval, dtype=float)
    dist_to_data = (array - start) % period
    dist_to_interval_end = (end - start) % period
    return dist_to_data < dist_to_interval_end


def in_union_of_intervals(values, intervals, inclusion=in_linear_interval):
    """
    Are values contained in a union of inclusive 1D intervals?

    This function implements only the union logic. It can be used with
    any notion of point-in-interval via the passed inclusion function.

    Args:
    values - arraylike
        The values to be tested for inclusion.
    ranges - iterable
        An iterable of 1D inclusive intervals, each interval
        specified as a range [lower, upper].
    inclusion - func
        A function that determines if a value is in a given interval,
        accepting trial array as the first argument, an interval
        [lower, upper] as the second, and returning a boolean array.

    Returns: in_union
    in_union - bool array, shape matching values
        A boolean of the same shape as the input values, valued True
        if the input values are in the union of the passed intervals.
    """
    in_union = np.zeros(values.shape, dtype=np.bool)
    for interval in intervals:
        in_interval = inclusion(values, interval)
        in_union = in_union | in_interval
    return in_union


def interval_contains_interval(interval1, interval2,
                               inclusion=in_linear_interval):
    """
    Is interval2 contained in interval1? Boolean output.

    This function implements the notion of an interval contained in
    another interval using only an abstracted notion of
    point-in-interval containment: interval2 is contained in interval1
    if both of the endpoints of interval2 are contained in interval1.
    The point-in-interval notion is defined via the user-supplied
    function inclusion, allowing this function to be used with any
    sort of values - linear, periodic, etc. By default, it assumes
    containment in the usual sense of the real number line.

    Args:
    interval1 - 1D, 2-element arraylike
    interval2 - 1D, 2-element arraylike
        Will test if interval2 is contained in interval1.
        The intervals should have the form [start, end], with start
        and end possibly subject to some criteria depending on the
        containment function used. The default linear containment
        requires that start <= end.
    inclusion - func
        A function that determines if a value is in a given interval,
        accepting trial array as the first argument, an interval
        [lower, upper] as the second, and returning a boolean array.

    Returns: in_union
    in_union - bool value
        True if interval2 is contained in interval1.
    """
    interval1 = np.asarray(interval1, dtype=float)
    start2, end2 = np.asarray(interval2, dtype=float)
    return inclusion(start2, interval1) & inclusion(end2, interval1)


def min_max(array, axis=None):
    """ Return the minimum and maximum of passed array """
    array = np.asarray(array)
    return np.asarray([array.min(axis=axis), array.max(axis=axis)])


def quartiles(array, axis=None):
    """ Return the quartile boundaries of passed array """
    array = np.asarray(array)
    zero, one = min_max(array, axis=axis)
    quarter = np.percentile(array, 25, axis=axis)
    half = np.median(array, axis=axis)
    threequarters = np.percentile(array, 75, axis=axis)
    return np.asarray([zero, quarter, half, threequarters, one])


def clipped_mean(data, weights=None, noise=None,
                 mask=None, fill_value=None, clip=5, max_iters=5,
                 max_fractional_remove=0.02, converge_fraction=0.02):
    """
    Compute a clipped, weighted mean of each column in the passed 2d
    array.  This is the weighted mean excluding any data points that
    differ from the unweighted median by greater than the passed clip
    factor times the standard deviation. This is iterative.

    Args:
    data - 2d ndarray
        Array on which to compute the clipped, weighted column mean.
    weights - 1d or 2d ndarray
        The weighting factors to be used.  Will be normalized so that
        each column of weights sums to 1.  If a 1d array is passed,
        the same weights will be used for each column. If unspecified,
        then uniform weights are used.
    noise - 2d ndarray, defalt=None
        The estimated noise in the passed data, will be ignored if not
        passed or combined in quadrature if an array is given
    fill_value - float, default=None
        If given, will be used to fill all mask values in the final
        output, otherwise masked values are allowed to float.
    clip - float, default=3
        All data differing from the unweighted median by more than
        clip*standard_deviation will be ignored
    max_iters - int, default=10
        maximum number of iterations before clipping is halted
    max_fractional_remove - float, default=0.02
        maximum fraction number of data points
        that can be clipped before clipping is halted
    mask - 2d boolean ndarray
        True for any pixels to be ignored in the computation.

    Returns: clipped_data_mean, clipped_data_noise (optional),
             mean_mask, clipped
    clipped_data_mean - 1d ndarray
        The clipped, weighted mean of each column of data
    clipped_data_noise - 1d ndarray
        The noise estimate in the clipped mean of each column, only
        returned if a noise array is passed
    mean_mask - 1d ndarray
        The masking array for the mean data, indicated any columns
        for which all values were either masked or clipped
    clipped_points - 2d boolean ndarray
        An array of the same size as the input data, with a True for
        every data point that was clipped
    """
    data = np.asarray(data, dtype=float)
    if mask is None:
        masked = np.zeros(data.shape, dtype=bool)
    else:
        masked = np.asarray(mask, dtype=bool)
    total_num_points = float(data.size)  # float for fractional divisions
    if total_num_points == 0:
        raise ValueError("Data arrays must be non-empty.")
            # division by total number of points is needed below
    masked_data = np.ma.array(data, mask=masked)
    clipped = np.zeros(data.shape, dtype=bool)
    # determine values to clip
    for iter in xrange(max_iters):
        sigma = np.ma.std(masked_data, axis=0)
        central = np.ma.median(masked_data, axis=0)
        distance = np.ma.absolute(masked_data - central)/sigma
            # default broadcasting is to copy vector along each row
        new_clipped = (distance > clip).data
            # a non-masked array, any data already masked in distance are set
            # False by default in size compare - this finds new clipped only
        num_old_nonclipped = np.sum(~clipped)
        clipped = clipped | new_clipped  # all clipped points
        masked_data.mask = clipped | masked  # actual clipping
        total_frac_clipped = np.sum(clipped)/total_num_points
        delta_nonclipped = np.absolute(np.sum(~clipped) - num_old_nonclipped)
        delta_frac_nonclipped = delta_nonclipped/float(num_old_nonclipped)
        if ((delta_frac_nonclipped <= converge_fraction) or  # convergence
            (total_frac_clipped > max_fractional_remove)):
            break
    # compute mean
    bad_pixels = masked | clipped
    if weights is None:
        weights = np.ones(data.shape)
    else:
        weights = np.array(weights, dtype=float)
    if weights.ndim == 1:
        weights = np.vstack((weights,)*masked_data.shape[1]).T
            # array with each row a constant value
    weights[bad_pixels] = 0.0  # do not include clipped or masked in norm
    total_weight = weights.sum(axis=0)
    all_bad = np.all(bad_pixels, axis=0)
    total_weight[all_bad] = 1.0
        # set nonzero fiducial total weight for wavelengths with no un-masked
        # values to avoid division errors; normalized weight is still zero
    weights = weights/total_weight  # divides each col by const
    clipped_data_mean = np.ma.sum(masked_data*weights, axis=0).data
    mean_mask = np.all(bad_pixels, axis=0)
    if fill_value is not None:
        clipped_data_mean[mean_mask] == fill_value
    if noise is not None:
        noise = np.asarray(noise, dtype=float)
        masked_noise = np.ma.masked_array(noise, mask=bad_pixels)
        clipped_variance = np.ma.sum((masked_noise*weights)**2, axis=0).data
        clipped_data_noise = np.sqrt(clipped_variance)
        if fill_value is not None:
            clipped_data_noise[mean_mask] == fill_value
        return clipped_data_mean, clipped_data_noise, mean_mask, clipped
    else:
        return clipped_data_mean, mean_mask, clipped


def interp1d_constextrap(*args, **kwargs):
    """
    This is a wrapper for scipy.interpolate.interp1d that allows
    a constant-valued extrapolation on either side of the input data.

    The arguments are the same as interp1d, and the return is a
    function analogous to that returned by interp1d. The only
    difference is that upon extrapolation the returned function will
    not throw an exception, but rather extend the interpolant as a
    flat line with value given by the nearest data value.
    """
    interp_func = interp.interp1d(*args, **kwargs)
    x, y = args[:2]  # convention from interp1d
    xmin_index = np.argmin(x)
    xmin = x[xmin_index]
    ymin = y[xmin_index]
    xmax_index = np.argmax(x)
    xmax = x[xmax_index]
    ymax = y[xmax_index]
    def interp_wrapper(z):
        """ 1d interpolation with constant extrapolation """
        z = np.asarray(z, dtype=float)
        is_lower = z <= xmin
        is_upper = xmax <= z
        valid = (~is_lower) & (~is_upper)
        output = np.nan*np.ones(z.shape)
        output[is_lower] = ymin
        output[is_upper] = ymax
        output[valid] = interp_func(z[valid])
        return output
    return interp_wrapper


def compute_projected_confidences(prob_draws, fraction=0.683):
    """
    Given a set of draws from an Gaussian distribution in D dimensions,
    this computes the D-ellipsoid containing the passed probability
    percentile, and then projects that ellipsoid onto the directions
    of each individual dimension.

    Args:
    prob_draws - 2D arraylike
        A set of N draws from a D-dimension distribution, with each
        draw occupying a row, i.e. the shape of prob_draws is (N, D)
    fraction - float, default=0.683
        The fraction of probability weight to be enclosed by the
        D-ellipse.  Default is 0.683, but note that this does not quite
        give a '1-sigma' ellipsoid: the weight enclosed by the covariance
        ellipsoid of a Gaussian distribution depends on dimension and is
        decreasing.  In 1D, 1 sigma corresponds to 68.3% confidence, but
        in higher dimension 1 sigma encloses less than 68.3% of the
        probability weight.  This code works off of percentiles rather
        than sigma-levels, so the ellipsoid returned is in general going 
        to be some larger multiple of the 1 sigma ellipse than naively
        expected from the 1D case.

    Returns: covariance, intervals
    covariance - 2D arraylike
        The covariance matrix of the Gaussian describing the samples.
    intervals - 1D arraylike
        The half-width of the projected confidence intervals
    """
    # get 6D confidence ellipse
    covariance = np.cov(prob_draws.T)  # normalized
    metric = np.linalg.inv(covariance)  # Mahalanobis metric
    center = np.median(prob_draws, axis=0)
    mdist_sq = []
    for num, row in enumerate(prob_draws):
        # compute Mahalanobis distance of each draw
        # would be nice to vectorize, but do not see how with a matrix mult
        shifted = row - center
        mdist_sq.append(np.dot(shifted.T, np.dot(metric, shifted)))
    conf_mdist_sq = np.percentile(mdist_sq, fraction*100)
    intervals = np.sqrt(np.diag(covariance*conf_mdist_sq))
        # this gets the max extent of the ellipsoid (scaled to the passed
        # prob weight) after projecting into each dimension. 
        # Derivation is in log notebook - see October 2014
    return covariance, intervals


def find_local_min(func, interval, atol=10**-7):
    """
    Find a local minimum in the passed interval (a, b), with a < b. 
    If no such minimum is found, returns None. Otherwise, the location 
    of the minimum and its function value are returned (in that order).

    Optimization is done with scipy's "bounded" Brent method. No minima 
    cases are identified by convergence onto an endpoint of the 
    interval. Tolerance (absolute) to accept a minima set by atol.
    """
    out = opt.minimize_scalar(func, bounds=interval,
                              method="bounded", tol=atol*(10**-2))
        # need to find minima at better tol than the endpoint compare tol
        # out is a dict; out['x'] is min location, out['fun'] is value
    if np.isclose(out['x'], interval, atol=atol).any():
        return None
    return [out['x'], out['fun']]


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


@warn_untested
def cart_to_polar(cart):
    """
    Convert 2D Cartesian coordinates to polar coordinates.

    Args:
    cart - arraylike, shape (..., 2)
        Array of 2D Cartesian coordinate pairs (x, y).
        Shape can be anything, provided the final axis has length 2. 
        Each element in the array is taken to represent one point,
        with the final axis giving the point's 2D coordinates.  

    Returns:
    polar - ndarray, shape matching input
        Array of 2D polar coordinate pairs (r, phi), matching
        the shape and format of the input. Angle is in radians.
    """
    cart = np.array(cart, dtype=float)
    x = cart[..., 0]
    y = cart[..., 1]
    polar = np.full(cart.shape, np.nan)
    polar[..., 0] = np.sqrt(x**2 + y**2) # r
    polar[..., 1] = np.arctan2(y, x) % (2*np.pi) # phi
    return polar


@warn_untested
def polar_to_cart(polar):
    """
    Convert 2D polar coordinates to Cartesian coordinates.

    Args:
    polar - arraylike, shape (..., 2)
        Array of 2D polar coordinate pairs (r, phi), angle in radians.
        Shape can be anything, provided the final axis has length 2. 
        Each element in the array is taken to represent one point,
        with the final axis giving the point's 2D coordinates.  

    Returns:
    cart - ndarray, shape matching input
        Array of 2D Cartesian coordinate pairs (x, y),
        matching the shape and format of the input.
    """
    polar = np.array(polar, dtype=float)
    r = polar[..., 0]
    phi = polar[..., 1]
    cart = np.full(polar.shape, np.nan)
    cart[..., 0] = r*np.cos(phi) # x
    cart[..., 1] = r*np.sin(phi) # y
    return cart 


@warn_untested
def rotate2d(coords, angles, form):
    """
    Rotate 2D coordinates by the given angle.

    Args:
    coords - arraylike, shape (..., 2)
        Array of 2D coordinates, given either as all Cartesian pairs (x, y)
        or all polar pairs(r, phi). The shape of the array may be anything,
        provided the final axis has length 2. Each element in the array is
        taken to represent one point, with the final axis giving the point's
        2D coordinate pair.
    angle - float or arraylike
        Angle to rotate, measured from x-axis towards y-axis in radians.
        If float, all points are rotated by the passed angle.
        If array, the shape must match the shape of all but the final axis
        of coords, i.e. angle.shape = coords.shape[:-1]. In this case, for
        each point there is one corresponding angle with which it is rotated.
    form - string, either 'cart' or 'polar'
        Specifes whether the passed coordinates are in Cartesian or poalr form. 

    Returns:
    rotated - arraylike, mathcing input
        An array of 2D coordinate pairs, matching the shape and format of the
        input, and rotated from the input values ccw by the passed angle.
    """
    coords = np.array(coords, dypte=float)
    if np.isscalar(angles):
        angles = float(angles) % (2*np.pi)
    else:
        angles = np.asarray(angles) % (2*np.pi)
    if form == 'cart':
        cart = True
    elif form == 'polar':
        cart = False
    else:
        raise ValueError("Invalid coordinate form {}".format(form))
    if cart:
        coords = to_polar(coords)
    # assume polar coords from here
    coords[..., 1] += angles
    coords[..., 1] = coords[..., 1] % (2*np.pi)
    if cart:
        coords = to_cart(coords)
    return coords


@warn_untested
def spherical_to_cart(spherical):
    """
    Convert 3D spherical coordinates to Cartesian coordinates.

    Args:
    spherical - arraylike, shape (..., 2)
        Array of 3D spherical coordinate triples (r, theta, phi), 
        with theta the polar angle [0, pi] and phi the azimuthal 
        angle [0, 2pi).  Shape can be anything, provided the final
        axis has length 3.  Each element in the array is taken to
        represent one point, the final axis giving the coordinates.  

    Returns:
    cart - ndarray, shape matching input
        Array of 3D Cartesian coordinate pairs (x, y, z),
        matching the shape and format of the input.
    """
    spherical = np.array(spherical, dtype=float)
    r = spherical[..., 0]
    theta = spherical[..., 1]
    phi = spherical[..., 2]
    cart = np.full(spherical.shape, np.nan)
    cart[..., 0] = r*np.sin(theta)*np.cos(phi) # x
    cart[..., 1] = r*np.sin(theta)*np.sin(phi) # y
    cart[..., 2] = r*np.cos(theta) # z
    return cart 


@warn_untested
def cart_to_spherical(cartesian):
    """
    Convert 3D Cartesian coordinates to spherical coordinates.

    Args:
    cart - arraylike, shape (..., 2)
        Array of 3D Cartesian coordinate pairs (x, y, x).
        Shape can be anything, provided the final axis has length 3. 
        Each element in the array is taken to represent one point,
        with the final axis giving the point's coordinates.  

    Returns:
    spherical - ndarray, shape matching input
        Array of 3D spherical coordinate pairs (r, theta, phi),
        matching the shape and format of the input.  Theta the polar
        angle [0, pi] and phi the azimuthal angle [0, 2pi).
    """
    cartesian = np.array(cartesian, dtype=float)
    x = cartesian[..., 0]
    y = cartesian[..., 1]
    z = cartesian[..., 2]
    spherical = np.full(cartesian.shape, np.nan)
    spherical[..., 0] = np.sqrt(x**2 + y**2 + z**2)    # r
    spherical[..., 1] = np.arccos(z/spherical[..., 0]) # theta
    spherical[..., 2] = np.arctan2(y, x)               # phi
    return spherical 


@warn_untested
def rotate3d(points, angles, axes):
    """
    Rotate 3D coordinates around the given axis by the given angle.

    Args:
    points - arraylike, shape (N, 3)
        Array of N 3D Cartesian coordinates
    angles - float or arraylike shape (N,)
        Angle (radian) to rotate, in the right-hand sense about the
        rotation axis.  If float, all points are rotated by the passed
        angle.  If array, one angle must be given per passed point,
        and each point will be rotated by its corresponding angle.
    axes - arraylike shape (3,) or (N, 3)
        Rotation axis. If dimension 1, each point is rotated about the
        passes axis.  If dimension 2, each point is rotated about its
        corresponding axis. 

    Returns:
    rotated - arraylike, mathcing input
        An array of 3D coordinate pairs for the rotated points, 
        matching the shape and format of the input points.
    """
    points = np.array(points, dtype=float)
    axes = np.array(axes, dtype=float)
    norm = np.sqrt(np.sum(axes**2, axis=-1))
    axes = (axes.T/norm).T  # axes now unit vectors
    if axes.ndim == 1:
        axes = np.tile(axes, (points.shape[0], 1))
    cross = np.cross(axes, points)
    dot = np.sum(axes*points, axis=-1)
    rotated = (points.T*np.cos(angles) + 
               cross.T*np.sin(angles) + 
               axes.T*dot*(1.0 - np.cos(angles))).T  # Rodrigues's Formula
    return rotated 


@warn_untested
def draw_from_sphere(shape):
    """ 
    Draw points uniformly over the unit sphere.  Draws are returned
    in two arrays, the first giving the polar and the second the
    azimuthal angles.  Both arrays will have the passed shape. 
    """
    # The area element on the sphere is:
    #   dA = r^2 sin(theta) d_theta d_phi = - r^2 d_cos(theta) d_phi
    # So to be uniform over the surface area we need to draw points
    # with phi and cos(theta) distributed uniformly over their ranges,
    # [0, 2pi) and [-1, 1].
    azimuth = np.random.uniform(0, 2*np.pi, shape)
    cos_polar = np.random.uniform(-1, 1, shape)  # cos(polar)
    polar = np.arccos(cos_polar)
    return polar, azimuth


@warn_untested
def draw_from_hemisphere(shape):
    """ 
    Draw points uniformly over the Northern (theta > 0) half of the
    unit sphere.  Draws are returned in two arrays, the first giving
    the polar and the second the azimuthal angles.  Both arrays will
    have the passed shape. 
    """
    # see "draw_from_sphere"
    # the polar range is now cos(theta) uniform in [0, 1] 
    azimuth = np.random.uniform(0, 2*np.pi, shape)
    cos_polar = np.random.uniform(0, 1, shape)  # cos(polar)
    polar = np.arccos(cos_polar)
    return polar, azimuth