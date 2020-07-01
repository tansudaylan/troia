import numpy as np
import matplotlib.pyplot as plt
from astropy import constants

'''
Equations taken from:
Rahvar, S., Mehrabi, A., & Dominik, M. 2011, MNRAS, 410, 912
and
Poleski, R., & Yee, J. C. 2019, Astronomy and Computing, 26, 35
'''
# t_star = 0.051
# t_start: t_0 - 5.5 * t_star,
# t_stop: t_0 + 5.5 * t_star,
# subtract_2460000: True,

# speed of light, c, (au/day)
# orbital radius, a, (au)
# mass of source star, m_star, mass of lens, M (M_sun)
# inclination angle with respect to observer-lens line of sight, φ, in radians


def mag_max(M, m_star, a):

    rho_star = 22.7 * ((M) ** -0.5) * ((m_star) ** 0.8) * ((a) ** -0.5)

    mu_max = np.sqrt(1 + 4/(rho_star ** 2))

    return mu_max

M_vals = np.linspace(1.0, 100000.0, 100000)
mstar_vals = np.linspace(1.0, 100000.0, 100000)
orbradius_vals = np.linspace(1.0, 100000.0, 100000)

mumax_vals = np.zeros(100000)

def poleski_main(t, t_0, u_0, t_E):
    # eq5
    tao = (t - t_0)/t_E

    # eq4
    u = np.sqrt((u_0 ** 2) + (tao ** 2))

    # eq5
    Aofu = ((u ** 2) + 2) / (u * np.sqrt((u ** 2) + 4))

    return Aofu

t_01 = 20.0
u_01 = 9.0
t_E1 = 0.008

t_vals1 = np.linspace(3.0, 37.0, 35)

amp_vals1 = np.zeros(35)

def rahvar_main(t, t_0, φ, M, m_star, a):

    c = 173.145 # (au/day)
    G = 0.000295913010 # au * 1/solar mass * (au/day)^2
    R_S = (2 * M * G)/(c ** 2) # au

    # equation 17
    t_E = ((2 * a)/c) * (np.sqrt(M/(m_star + M))) * 1/24

    # equation 18
    tao = (t - t_0)/t_E
    u_0 = np.sqrt(φ * (a / ((2 * R_S))))
    u_t = np.sqrt((u_0 ** 2) + (tao ** 2))

    # equaiton 14
    Amplitude = (2/np.pi) * ((1 + 1/(u_t ** 2)) * np.arcsin(1/(np.sqrt(1 + 1/(u_t ** 2)))) + 1/u_t)

    return Amplitude

t_02 = 20.0
φ = 1.59989 * (10.0 ** (-6)) # radians
M = 8.5
m_star = 0.35
a = 17

t_vals2 = np.linspace(3.0, 37.0, 35)

amp_vals2 = np.zeros(35)

for element in t_vals2:
    amp_vals2[int(element - 3)] = (rahvar_main(element, t_02, φ, M, m_star, a))
plt.figure(1)
plt.plot(t_vals2, amp_vals2, label="Rahvar")
plt.xlabel("Time (hours)")
plt.ylabel("Magnfication")

for element in t_vals1:
    amp_vals1[int(element - 3)] = (poleski_main(element, t_01, u_01, t_E1))

plt.plot(t_vals1, amp_vals1, label="Poleski")
plt.xlabel("Time")
plt.ylabel("Magnification")
plt.legend()





for element in M_vals:
    mumax_vals[int(element - 1)] = (mag_max(element, 1, 1))

plt.figure(2)
plt.plot(M_vals, mumax_vals)
plt.xlabel("Mass of Lens")
plt.ylabel("Maximum Magnification")
plt.title("Variable Mass of Lens, m_star = 1, a = 1")
plt.legend()

mumax_vals = np.zeros(100000)

for element in mstar_vals:
    mumax_vals[int(element - 1)] = (mag_max(1, element, 1))

plt.figure(3)
plt.plot(mstar_vals, mumax_vals)
plt.xlabel("Mass of Source")
plt.ylabel("Maximum Magnification")
plt.title("M = 1, Variable m_star, a = 1")
plt.legend()

mumax_vals = np.zeros(100000)

for element in orbradius_vals:
    mumax_vals[int(element - 1)] = (mag_max(1, 1, element))

plt.figure(4)
plt.plot(orbradius_vals, mumax_vals)
plt.xlabel("Orbital Radius")
plt.ylabel("Maximum Magnification")
plt.title("M = 1, m_star = 1, Variable Orbital Radius")
plt.legend()

plt.show()
