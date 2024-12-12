""" 
Routines for computing structure of fully convective star of a given mass and 
radius.

<Team name, Connor Lefebvre, Brandon Grimes>
"""

import numpy as np
from eos import get_rho_and_T, mean_molecular_weight
from ode import rk4
from astro_const import G, Msun, Rsun, Lsun, kB, m_u, fourpi, delta_m, eta, xi
from reactions import pp_rate

def central_thermal(m,r,mu):
    """ 
    Computes the central pressure, density, and temperature from the polytropic
    relations for n = 3/2.

    Arguments
        m
            mass in solar units
        r
            radius is solar units
        mu
            mean molecular weight
    Returns
        Pc, rhoc, Tc
            central pressure, density, and temperature in solar units
    """
    # fill this in
    Pc = 0.77*(G*m*2)/r**4
    rhoc = 5.99*(3*m/(4*np.pi*r**3))
    Tc = 0.54*(mu*m_u/kB)(G*m/r)
    
    return Pc, rhoc, Tc

# The following should be modified versions of the routines you wrote for the 
# white dwarf project
def stellar_derivatives(m,z,rho,rate):
    """
    RHS of Lagrangian differential equations for radius and pressure
    
    Arguments
        m
            current value of the mass
        z (array)
            current values of (radius, pressure)
        mue
            ratio, nucleons to electrons.  For a carbon-oxygen white dwarf, 
            mue = 2.
        rate
            pp_rate , epsilon value
        
    Returns
        dzdm (array)
            Lagrangian derivatives dr/dm, dP/dm
    """
    r, P = z
    drdm = 1/(4*np.pi*r**2*rho) #lagrangian form of eq 2, eq4
    dPdm = -G * m / (4 * np.pi * r**4) #lagrangian form of eq 3, eq5
    dLdm = rate
    # evaluate dzdm
    dzdm = np.zeros_like(z)
    dzdm[0] = drdm
    dzdm[1] = dPdm
    dzdm[2] = dLdm 
    return dzdm

def central_values(Pc,delta_m,rho,T_c,rho_c, XH):
    """
    Constructs the boundary conditions at the edge of a small, constant density 
    core of mass delta_m with central pressure P_c
    
    Arguments
        Pc
            central pressure (units = Pascal)
        delta_m
            infinitesimal core mass (units = kg)
        mue
            nucleon/electron ratio
    
    Returns
        z = array([ r, p ])
            central values of radius and pressure (units = m, Pa)
    """
    
    r = (3 * delta_m / (4 * np.pi *rho))**(1/3)  # meters, from eq 9 in instructions
    Lc = pp_rate(T_c, rho_c, XH, pp_factor=1.0) * delta_m
    z = np.zeros(3)
    # compute initial values of z = [ r, p ]
    z = np.array([r,Pc,Lc])
    return z

    def lengthscales(m,z,rho,rate):
     """
    Computes the radial length scale H_r and the pressure length H_P
    
    Arguments
        m
            current mass coordinate (units = kg)
        z (array)
           [ r, p ] (units = m, Pascals)
        mue
            mean electron weight
    
    Returns
        z/|dzdm| (units = m and kg)
    """
    r, P = z

    #length scales
    Hr = 4*pi*r**3*rho # eq 10
    Hp = P*4*pi*r**4/(G*m) #eq 11
    HL = L/(rate)
    h = min(Hr, Hp, HL)
    return h

    def integrate(P_c,delta_m,eta,xi,rho,max_steps=10000):
     """
    Integrates the scaled stellar structure equations

    Arguments
        Pc
            central pressure (units = Pascal)
        delta_m
            initial offset from center (units = kg)
        eta
            The integration stops when P < eta * Pc
        xi
            The stepsize is set to be xi*min(p/|dp/dm|, r/|dr/dm|)
        mue
            mean electron mass
        max_steps
            solver will quit and throw error if this more than max_steps are 
            required (default is 10000)
                        
    Returns
        m_step, r_step, p_step
            arrays containing mass coordinates, radii and pressures during 
            integration (what are the units?) -> should be kg, m, and Pa
    """
        
    m_step = np.zeros(max_steps)
    r_step = np.zeros(max_steps)
    p_step = np.zeros(max_steps)
    L_step = np.zeros(max_steps)

    # set starting conditions using central values
    z = central_values(Pc, delta_m, mue)
    m_step[0] = delta_m
    r_step[0], p_step[0], L_step[0] = z 
    
    Nsteps = 0
    for step in range(max_steps):
        radius = z[0]
        pressure = z[1]
        luminosity = z[2]
        # are we at the surface?
        if (pressure < eta*Pc):
            break
        # store the step
        m_step[Nsteps] = m_step[Nsteps - 1] + h if Nsteps > 0 else delta_m
        r_step[Nsteps] = radius
        p_step[Nsteps] = pressure
        L_step[Nsteps] = luminosity
        # set the stepsize
        h = lengthscales(m_step[Nsteps], z, mue) * xi # eq 12
        
        # take a step
        z = rk4(f=stellar_derivatives, t=m_step[Nsteps], z=z, h=h, args=(mue,))
        # increment the counter
        Nsteps += 1
    # if the loop runs to max_steps, then signal an error
    else:
        raise Exception('too many iterations')
        
    return m_step[0:Nsteps],r_step[0:Nsteps],p_step[0:Nsteps],L_step[0:Nsteps]