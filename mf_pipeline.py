from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import mock_light_curves as mlc
import os

#fits_file = "./Light Curves/tess2019357164649-s0020-0000000004132863-0165-s_lc.fits"
fits_file = "https://archive.stsci.edu/missions/tess/tid/s0001/0000/0000/2515/5310/tess2018206045859-s0001-0000000025155310-0120-s_lc.fits"
with fits.open(fits_file, mode="readonly") as hdulist:
    tess_bjds = hdulist[1].data['TIME']
    sap_fluxes = hdulist[1].data['SAP_FLUX']
    pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
    var = hdulist[1].data['PSF_CENTR1']
    
#print(1440*(tess_bjds[-1] - tess_bjds[0])/len(tess_bjds))

# Start figure and axis.
plt.figure()

# Find average flux from valid data points
total = 0
x = 0

for i in range(len(pdcsap_fluxes)):
    flux = pdcsap_fluxes[i]
    if not np.isnan(flux):
        total += flux
        x += 1
        
average = total/x

# Replace invalid data points normalize light curve
for i in range(len(pdcsap_fluxes)):
    if np.isnan(pdcsap_fluxes[i]):
        pdcsap_fluxes[i] = 0
    else:
        pdcsap_fluxes[i] = (pdcsap_fluxes[i] - average)/average
        
ss_fluxes = -mlc.supersample(pdcsap_fluxes)

# Let's label the axes and define a title for the figure.
#fig.suptitle("WASP-126 b Light Curve - Sector 1")
plt.ylabel("PDCSAP Flux (e-/s)")
plt.xlabel("Time (TBJD)")

# Plot the timeseries in black circles.
plt.plot(tess_bjds, ss_fluxes, 'ko')

# Adjust the left margin so the y-axis label shows up.
plt.subplots_adjust(left=0.15)
plt.savefig("Light Curve")
plt.close()

def run_filters(lc):
    best = {"corr": 0, 
            "corrs": None, 
            "amplitude": None, 
            "width": None, 
            "threshold": None,
            "result": False}
    
    for amp in [.0015 + .001*i for i in range(14)]:
        for w in [5 + 5*j for j in range(12)]:
            template = mlc.generate_template(amp, w)
            result, top_corr, corrs, template, threshold = mlc.match_filter(lc, template, threshold=w*amp**2/3)
            if (top_corr > best["corr"] and best["result"] == result) or (not best["result"] and result):
                best["corr"], best["corrs"] , best["result"] = top_corr, corrs, result
                best["amplitude"], best["width"], best["threshold"] = amp, w, threshold
                
    return best

best = run_filters(ss_fluxes)
plt.figure()
plt.ylabel("Correlation")
plt.plot(best["corrs"])
plt.savefig("Correlation")
plt.close()
print(best)