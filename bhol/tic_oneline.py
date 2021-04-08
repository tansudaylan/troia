import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs



def tic_query(integer):
    tic_ID = "TIC " + str(integer)
    bhol = Catalogs.query_object(tic_ID, catalog="TIC")
    gaia_ID = bhol[0]["GAIA"]
    radius = bhol[0]["rad"]
    temperature = bhol[0]["Teff"]
    Tmag = bhol[0]["Tmag"]
    mass = bhol[0]["mass"]
    ra = bhol[0]["ra"]
    dec = bhol[0]["dec"]
    distance = bhol[0]["d"]
    TESS_info = [int(gaia_ID), radius, temperature, Tmag, mass, ra, dec, distance]
    return TESS_info


TIC_ID = 260647166 # input any TIC ID as an integer
TESS_info = tic_query(TIC_ID)
print(TESS_info)
#print(TESS_Info)

string = "TIC:" + str(TIC_ID) + "  $R_{*}$:" + str(round(TESS_info[1], 2)) + "  Teff:" + str(round(TESS_info[2], 0)) + "  \nTmag:" + str(round(TESS_info[3], 1)) + "  $m_{*}$:" + str(TESS_info[4]) + "  Ra:" + str(round(TESS_info[5], 2)) + "  Dec:" + str(round(TESS_info[6], 2)) + "  d:" + str(round(TESS_info[7],1))

print(string) #S tring in its raw form

plt.plot()
plt.xlim(-5, 5) # Values be changed based on preferncce
plt.ylim(-5, 5)
plt.text(-4, 0, string, fontsize=8)

# String in a matplotlib pdf
plt.savefig("/Users/kartikpingle/Dropbox/BHOL/data/TIC_string_test.pdf") # Change this path, but leave it as a pdf
