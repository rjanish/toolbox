""" operations on functions """


import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt


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