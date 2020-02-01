""" 
warnings to add to in-module code
"""


import warnings


def warn_untested(func):
    """ Add to untested functions"""
    def wrapper(*args, **kwargs):
        warnings.warn("{} is untested; see source".format(func.__name__))
        return func(*args, **kwargs)
    return wrapper