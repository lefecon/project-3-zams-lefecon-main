# 
# <Team name, Connor Lefebvre, Brandon Grimes>
# 

import numpy as np

def pp_rate(T,rho,XH,pp_factor=1.0):
    """
    Specific heating rate from pp chain hydrogen burning. Approximate rate 
    taken from Hansen, Kawaler, & Trimble.
    
    Arguments
        T, rho
            temperature [K] and density [kg/m**3]
        XH
            mass fraction hydrogen
        pp_factor
            multiplicative factor for rate
    Returns
        heating rate from the pp-reaction chain [W/kg]
    """
    
    # fill this out
    rate = (2.4*10**-3*rho*XH**2)/T**(2/3)*np.e(-3.380/T**(1/3))*pp_factor
    return rate
