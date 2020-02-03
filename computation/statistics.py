""" statistical computations """


import numpy as np


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