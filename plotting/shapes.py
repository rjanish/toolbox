"""
Construction of various shapes as matplotlib patches.

These are mostly wrappers for the constructors of primitive patches,
allowing input arguments that I find more useful, along with a few
new constructors for shapes not in matplotlib.patches's defaults.

To eliminate the need for separate imports of matplotlib.pathces and
this module, some wrappers are included that exactly preserve the
original matplotlib.patches syntax (e.g., Circle).
"""


import numpy as np
import matplotlib.patches as mpatch
import shapely.geometry as geo
import descartes as desc


def square(center, side_length, *args, **kwargs):
    """ Create a square with the given center and side length. """
    center = np.asarray(center)
    side_length = float(side_length)
    lower_left = center - 0.5*side_length
    return patch.Rectangle(lower_left, side_length, side_length,
    					   *args, **kwargs)


def circle(center, radius, *args, **kwargs):
    """ Create a circle with the given center and radius. """
    return patch.Circle(center, radius, *args, **kwargs)


def polar_box(rad_min, rad_max, ang_min, ang_max, **kwargs):
    """
    Returns a box with boundaries given by constant polar.
    coordinates. Angle arguments are to be given in degrees.
    The first four agruments must specify the box dimensions,
    followed by optional matplotlib patch keyword arguments.
    """
    delta_angle = (ang_max - ang_min) % 360.0
    ang_max_sequential = ang_min + delta_angle
    maxarc_coords = list(approxArc((0.0, 0.0), rad_max, ang_min,
    							   ang_max_sequential).coords)
    minarc_coords = list(approxArc((0.0, 0.0), rad_min, ang_min,
    							   ang_max_sequential).coords)
    minarc_coords.reverse()
    poly = geo.Polygon(maxarc_coords + minarc_coords)
    return desc.PolygonPatch(poly, **kwargs)
