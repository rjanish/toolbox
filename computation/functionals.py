""" operations on functions """


import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt

from toolbox.utilities import warn_untested


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


@warn_untested
def track_period(x, y, frequency_guess, padding_factor=10):
    """ 
    Given data y(x) which has the form of an oscillation with a slow
    frequency modulation, extract the frequency evolution $\omega(x)$.
    This uses *angular frequency*, so the period is 2\pi/frequency.

    This begins with an initial guess for the frequency omega0 and an 
    initial data pair x0, y0, then steps to $x1 = x0 + 2\pi/omega0$. It 
    expects to find near x1 a recurrence of the value y0, and solves 
    y0 = y(x1 + delta) for small $\delta$ to find the true period T0. 
    It then continues with the just-computed period T0 as the guess 
    used to search for the next recurrence of y0 to find T1, and so on. 

    x, y - 1D ndarrays of the same length
    frequency_guess - float
      This is a guess for the initial angular frequency, $\omega(x[0])$
    padding - float
      Something 

    Returns an array of sample points xn (not necessarily the same as 
    were passed) and the frequency computed at those points \omega(xn).
    """
    x, y = np.asarray(x), np.asarray(y)
    period_guess = 2*np.pi/frequency_guess
    ytarget = y[0]
    xrecur_guess = x[0] + period_guess
    xrecur_previous = x[0]
    times, periods = [], [] 
    while xrecur_guess < x[-1]:
        # Find the x values immediately to the right and left of the 
        # expected recurrence point, and fit a line between these points. 
        # Solve for the true recurrence using this linear approximation.
        xleft = np.max(x[x < xrecur_guess])
        xleft_subindex = np.argmax(xleft)
        xright = np.min(x[x > xrecur_guess])
        xright_subindex = np.argmax(xright)
        yleft = y[x < xrecur_guess][xleft_subindex]
        yright = y[x > xrecur_guess][xright_subindex]
        xrecur = xleft + (ytarget - yleft)*(xright - xleft)/(yright - yleft)
        periods.append(xrecur - xrecur_previous)
        times.append((xrecur + xrecur_previous)*0.5)
        xrecur_guess = xrecur + periods[-1]
        xrecur_previous = xrecur
    # better idea: (?)
    # start with x[1], and compute z = y - y[1]. Every zero of z is a 
    # possible recurrence spot, and y either has a positive or negative
    # slope there. Find all x where z(x) = 0 and z_slope(x) = z_slope[1],
    # these are the recurrences x_recur and they give the periods. 
    # Repeat starting with x[2], up to x_recur[0] to get a finer sample.
    frequencies = 2*np.pi/np.asarray(period)    
    return np.asarray(times), frequencies