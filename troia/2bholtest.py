import astropy.units as u
from astropy.coordinates import SkyCoord
#from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n/#p.set_printoptions(suppress=True)

path1 = "/Users/kartikpingle/Dropbox/BHOL/Data/gaiaRV/GaiaSource_6714230465835878784_6917528443525529728.csv"
path2 = "/Users/kartikpingle/Dropbox/BHOL/Data/gaiaRV/GaiaSource_5933051914143228928_6714230117939284352.csv"
path3 = "/Users/kartikpingle/Dropbox/BHOL/Data/gaiaRV/GaiaSource_5502601873595430784_5933051501826387072.csv"
path4 = "/Users/kartikpingle/Dropbox/BHOL/Data/gaiaRV/GaiaSource_4475722064104327936_5502601461277677696.csv"
path5 = "/Users/kartikpingle/Dropbox/BHOL/Data/gaiaRV/GaiaSource_3650805523966057472_4475721411269270528.csv"
path6 = "/Users/kartikpingle/Dropbox/BHOL/Data/gaiaRV/GaiaSource_2851858288640_1584379458008952960.csv"
path7 = "/Users/kartikpingle/Dropbox/BHOL/Data/gaiaRV/GaiaSource_2200921875920933120_3650804325670415744.csv"
path8 = "/Users/kartikpingle/Dropbox/BHOL/Data/gaiaRV/GaiaSource_1584380076484244352_2200921635402776448.csv"

#   LINES 46-47 MODEL
#catalog_data = Catalogs.query_criteria(catalog="Tic", GAIA="6714230465835878784",objType="STAR")
#print(catalog_data)

data11through18 = [path1, path2, path3, path4, path5, path6, path7, path8]
''''
li = []
additive = 0
for filename in data11through18:
    df = pd.read_csv(filename, index_col=None, header=0)
    print(additive)
    additive += 1
    li.append(df)

gaia_data1 = pd.concat(li, axis = 0, ignore_index = True)
'''

gaia_data1 = pd.read_csv(path8, index_col=None, header=0)

print(gaia_data1.size)


#print(gaia_data["radial_velocity_error"])
gaia_data1a = np.array(gaia_data1["radial_velocity_error"])
gaia_data1b = np.array(gaia_data1["source_id"])

#print(np.where(gaia_data >= 15))
rv_index = np.where(gaia_data1a >= 15)[0]
#print(rv_index)
#print((gaia_data1["dec"])[rv_index])
#ra_vals = (gaia_data1["ra"])[rv_index]
#dec_vals = (gaia_data1["dec"])[rv_index]
gaia_IDs = (gaia_data1b)[rv_index]

ticIDs = np.zeros(1265)

additive = 0
#print(ra_vals[[0]])
#ra_vals = np.array(ra_vals)
#dec_vals = np.array(dec_vals)



while additive < 1265:
        mast_table = Catalogs.query_criteria(catalog="Tic", GAIA=str(gaia_IDs[additive]),objType="STAR")
        print(additive)
        ticIDs[additive] = int(mast_table["ID"][0])

        additive += 1

print(ticIDs)

'''
for element in ra_vals:
    print(element)
    print(np.where(ra_vals == element)[0][0])
    print(ra_vals[np.where(ra_vals == element)[0][0]])
    additive += 1
    if additive == 10:
        break
'''

'''
while additive < 1265:
    #icrscoords = str()
    #print(str(ra_vals[additive]) + " " + str(dec_vals[additive]))
    #if additive == 10:
    #    break
    mast_table = Catalogs.query_region(str(ra_vals[additive]) + " " + str(dec_vals[additive]), radius = 0.001, catalog="TIC")


    ticIDs[additive] = int(mast_table["ID"][0])
#    mastRAs[additive] = mast_table["ra"][0]
#    mastDECs[additive] = mast_table["dec"][0]

    #print(mast_table["dec"][0])
    print(additive)
    additive += 1

print(ticIDs)

#print(rv_index[0][0])

#print(catalog_data["ID"])
'''


charar = np.chararray((1, 2), itemsize=27)
table_data = np.zeros((1265, 2))



additive = 0
charar[0] = "TIC ID", "Radial Velocity Uncertainty"
while additive < 1265:
    table_data[additive] = (ticIDs[additive]), (gaia_data1a[rv_index])[additive]
    additive += 1
#table_data = table_data.astype("|S6")
table_data = np.concatenate((charar, table_data))


ascii.write(table_data, 'Gaia_rv_error_8.csv', format='csv', fast_writer=False)
