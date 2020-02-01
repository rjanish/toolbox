"""
Physical constants, scales, and unit conversions. 
"""

Msolar = 1.12*(10**57) # GeV 

Mp = 0.932 # GeV
Me = 5.11*(10**-4) # GeV

def GeV_to_cm(x):
    """ 
    Convert length value in GeV^-1 to cm
    """
    return x*2*10**(-14)

def GeV_to_km(x):
    """ 
    Convert length value in GeV^-1 to km
    """
    return x*2*10**(-19)