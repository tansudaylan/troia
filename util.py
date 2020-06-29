import numpy as np
import matplotlib.pyplot as plt
from astropy import constants

'''
Equations taken from:
Rahvar, S., Mehrabi, A., & Dominik, M. 2011, MNRAS,410, 912v
'''
# t_star = 0.051
# t_start: t_0 - 5.5 * t_star,
# t_stop: t_0 + 5.5 * t_star,
# subtract_2460000: True,

# speed of light, c, (au/day)
# orbital radius, a, (au)
# mass of source star, m_star, mass of lens, M (M_sun)
# inclination angle with respect to observer-lens line of sight, φ, in arcsec

t_0 = 100.0
φ = 0.33   # 1.59989 * (10 ** (-6)) radians
M = 8.5
m_star = 0.35
a = 17

# rho_star = 22.7 * ((m_star) ** 0.8) * ((M) ** -0.5) * ((a) ** -0.5)

def calculate(t, t_0, φ, M, m_star, a):

    c = 173.145 # (au/day)
    G = 0.000295913010 # au * 1/solar mass * (au/day)^2
    R_S = (2 * M * G)/(c ** 2) # au

    # equation 17
    t_E = ((2 * a)/c) * (np.sqrt(M/(m_star + M)))

    # equation 18
    tao = (t - t_0)/t_E
    u_0 = np.sqrt(φ * (a / ((2 * R_S))))
    u_t = np.sqrt((u_0 ** 2) + (tao ** 2))

    # equaiton 14
    Amplitude = (2/np.pi) * ((1 + 1/(u_t ** 2)) * np.arcsin(1/(np.sqrt(1 + 1/(u_t ** 2)))) + 1/u_t)

    return Amplitude

t_vals = np.linspace(0.0, 200.0, 201)

amp_vals = np.zeros(201)

for element in t_vals:
    amp_vals[int(element)] = (calculate(element, t_0, φ, M, m_star, a))

plt.plot(t_vals, amp_vals)
plt.xlabel("Time")
plt.ylabel("Magnfication")
plt.show()
