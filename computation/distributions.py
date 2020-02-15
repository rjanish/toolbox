""" probability distributions """


import numpy as np

from toolbox.utilities import warn_untested


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
    return np.exp(-0.5*(arg**2))/normalized


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