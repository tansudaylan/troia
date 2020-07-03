import numpy as np
import matplotlib.pyplot as plt
#from astropy import constants

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
mstar_vals = np.logspace(1.0, 2.0, 100000)
print(mstar_vals)
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
u_01 = 3.325
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



additive = 0

for element in t_vals1:
    amp_vals1[additive] = (rahvar_main(element, t_02, φ, M, m_star, a))
    additive += 1

plt.figure(1)
plt.plot(t_vals1, amp_vals1, label="Rahvar")
plt.xlabel("Time [Hours]")
plt.ylabel("Magnfication")

additive = 0

for element in t_vals1:
    amp_vals1[additive] = (poleski_main(element, t_01, u_01, t_E1))
    additive += 1

plt.plot(t_vals1, amp_vals1, label="Poleski")
plt.xlabel("Time [Hours]")
plt.ylabel("Magnification")
plt.legend()

# second set

additive = 0

for element in M_vals:
    mumax_vals[additive] = (mag_max(element, 1, 1))
    additive += 1

plt.figure(2)
plt.plot(M_vals, mumax_vals)
plt.xlabel("Mass of Lens [$M_{\odot}$]")
plt.ylabel("Maximum Magnification")
plt.title("m_star = 1 [$M_{\odot}$], a = 1 [AU]")
plt.legend()

mumax_vals = np.zeros(100000)

additive = 0

for element in mstar_vals:
    mumax_vals[additive] = (mag_max(1, element, 1))
    additive += 1

plt.figure(3)
plt.plot(mstar_vals, mumax_vals)
plt.xlabel("Mass of Star [$M_{\odot}$]")
plt.ylabel("Maximum Magnification")
plt.title("M = 1 [$M_{\odot}$], a = 1 [AU]")
plt.legend()

mumax_vals = np.zeros(100000)

additive = 0

for element in orbradius_vals:
    mumax_vals[additive] = (mag_max(1, 1, element))
    additive += 1

plt.figure(4)
plt.plot(orbradius_vals, mumax_vals)
plt.xlabel("Orbital Radius [AU]")
plt.ylabel("Maximum Magnification")
plt.title("M = 1 [$M_{\odot}$], m_star = 1 [$M_{\odot}$]")
plt.legend()

plt.show()
