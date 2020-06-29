import numpy as np
import matplotlib.pyplot as plt

'''
Equations taken from:
Poleski, R., & Yee, J. C. 2019, Astronomy and Computing,26, 35

'''

# t_start: t_0 - 5.5 * t_star,
# t_stop: t_0 + 5.5 * t_star,
# subtract_2460000: True,


t_0 = 100.0
u_0 = 0.06
t_E = 200.0
t_star = 0.051
def light_curve(t, t_0, u_0, t_E):
    # eq5
    tao = (t - t_0)/t_E

    # eq4
    u = np.sqrt((u_0 ** 2) + (tao ** 2))

    # eq5
    Aofu = ((u ** 2) + 2) / (u * np.sqrt((u ** 2) + 4))

    return Aofu

t_vals = np.linspace(0.0, 200.0, 201)

amp_vals = np.zeros(201)

for element in t_vals:
    amp_vals[int(element)] = (light_curve(element, t_0, u_0, t_E))
print(amp_vals)
plt.plot(t_vals, amp_vals)
plt.xlabel("time")
plt.ylabel("magnfication")
plt.show()




#u=rho_*
#rho_star_set = np.zeros((1, 5))
#rho_star = 0.81
#A = (2 / np.pi) * (1 + 1/np.square(rho_star) * np.arcsin(1 / np.sqrt(1 + 1/np.square(rho_star))) + 1/rho_star)
#print(A)
