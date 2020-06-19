import math
import numpy as np
import matplotlib.pyplot as plt

def generate_light_curve(P, i, M_BH, M_S=1, R_S=1, rho_S=1.41, snr=7.1):
    '''
    Takes in various input parameters and generates a 1000 dimensional light curve
    based on EV, SL and beam signals.

    P: Orbital Period (days)
    M_BH: Black Hole Mass (solar masses)
    M_S: Stellar Mass (solar masses)
    R_S: Stellar Radius (solar radii)
    rho_S: Stellar Density (g/cm^3)
    i: orbital inclination (angle)

    returns: size 1000 array containing the light curve
    '''

    s_ev = .0189 * math.sin(i)**2 * P**(-2) * (1/rho_S) * (1/(1 + M_S/M_BH))    # unitless 
    s_beam = .0028 * math.sin(i) * P**(-1/3) * (M_BH + M_S)**(-2/3) * M_BH      # unitless
    s_sl = 7.15*10**(-5) * R_S**(-2) * P**(2/3) * M_BH * (M_BH + M_S)**(1/3)    # unitless
    tau_sl = .075 * math.pi/4 * P**(1/3) * (M_BH + M_S)**(-1/3) * R_S           # days

    # signal will represent 27 days
    bin_size = 27/1000
    signal = np.zeros(1000)
    for j in range(1000):
        t = j * bin_size
        ev =  -s_ev * math.cos(4*math.pi*t/P)
        beam = s_beam * math.sin(2*math.pi*t/P)
        sl = s_sl if (t%P) < tau_sl else 0
        signal[j] += ev + beam + sl + 1 + np.random.normal(0,.0005)
    
    plt.plot(signal)
    
    return signal
