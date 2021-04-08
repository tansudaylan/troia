import os, fnmatch

from tdpy.util import summgene

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import astroquery
from astroquery.mast import Catalogs

import datetime
import astropy

from astropy.io import fits

pathbase = os.environ['BHOL_DATA_PATH'] + '/'
pathdata = pathbase + 'data/'
pathimag = pathbase + 'imag/'

# read WTV output
path = pathdata + 'wtv-WTVT.csv'
print('Reading from %s...' % path)
data = np.loadtxt(path, skiprows=45, delimiter=',')
rasc = data[:, 0]
decl = data[:, 1]
boolsect = data[:, 2:]
# determine visible targets
indxoldd = np.where(np.sum(boolsect[:, 26:], 1) > 0)[0]

indxneww = []
for k in indxoldd:
    if len(indxneww) == 0 or (np.sqrt((rasc[k] - rasc[np.array(indxneww)])**2 + (decl[k] - decl[np.array(indxneww)])**2) > 2.).all():
        indxneww.append(k)
indxneww = np.array(indxneww)
indx = np.array(indxneww)
print('indxneww')
summgene(indxneww)
print('indxoldd')
summgene(indxoldd)
print('data')
summgene(data)
print('indx')
summgene(indx)

# read TIC xmat
path = pathdata + 'datafull.csv'
print('Reading from %s...' % path)
datafull = np.loadtxt(path, delimiter=',', skiprows=1)
 

# write the output
pathsave = pathdata + 'outp.csv'
print('Writing to %s...' % pathsave)
fmt = ['%16d', '%8.3f', '%8.3f', '%8.3f', '%8.3f', '%8.3f']
np.savetxt(pathsave, np.nan_to_num(datafull[indx, :]), delimiter=',', fmt=fmt)

pathsave = pathdata + 'outp.tex'
print('Writing to %s...' % pathsave)
np.savetxt(pathsave, datafull[indx, :], delimiter=' & ', header='TIC ID & RA & DEC & PM RA & PM DEC & TESS Magnitude', fmt=fmt, newline='\\\\ \n \\hline \n')

