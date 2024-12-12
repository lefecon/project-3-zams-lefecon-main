import numpy as np
import matplotlib.pyplot as plt
import astro_const as ac
from zams import Teff, surface_luminosity, lum_diff, find_radius
from reactions import pp_rate
from eos import mean_molecular_weight, get_rho_and_T
import structure as struc


m, r, P, L = struc.integrate(P_c,ac.delta_m,ac.eta,ac.xi,rho,max_steps=10000)

mue = mean_molecular_weight

P_c = 0.77(ac.G*m**2/r**4)
rho_c = 5.99(3*m/(4*np.pi*r**3))
T_c = 0.54((mue*ac.m_u/ac.kB)*(ac.G*m/r))

rho, T = get_rho_and_T(P,P_c,rho_c,T_c)

teff = Teff(m)

# part 10a
plt.plot(np.log(L/ac.Lsun),np.log(Teff))
# part 10b
plt.plot(np.log(T_c),np.log(rho_c))
