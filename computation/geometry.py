""" computational geometry """


import numpy as np

from utilities.local_warnings import warn_untested


def bounding_radius(polygon, center=None):
    """
    Return the radius of the smallest circle with the passed center
    that completely encloses the passed polygon.  If no center is
    given, the polygon's centroid is used (in this case, the returned
    result is the polygon's circumradius).  The input is assumed to
    be a shapely polygon object.
    """
    verticies = np.asarray(polygon.boundary.coords, dtype=float)
    if center is None:
        center = np.asarray(polygon.centroid.coords, dtype=float)
    sq_dists = np.sum((verticies - center)**2, axis=1)
    return np.sqrt(sq_dists.max())


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