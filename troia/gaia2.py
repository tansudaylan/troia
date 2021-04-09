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

## read PCAT path environment variable
pathbase = os.environ['BHOL_DATA_PATH'] + '/'
pathdata = pathbase + 'data/'
pathimag = pathbase + 'imag/'

# read gaia RA, DEC
path = pathdata + 'gaia.csv'
print('Reading from %s...' % path)
datagaia = np.loadtxt(path, skiprows=5, delimiter=',')

# read TIC xmat from MAST
path = pathdata + 'MAST_Crossmatch_TIC.csv'
print('Reading from %s...' % path)
datatemp = np.loadtxt(path, skiprows=5, delimiter=',')
data = np.copy(datatemp)
data[:, 0] = datatemp[:, 2]
data[:, 1] = datatemp[:, 0]
data[:, 2] = datatemp[:, 1]

# write to CSV for viewing tool
pathsave = pathdata + 'datafull.csv'
print('Writing to %s...' % pathsave)
np.savetxt(pathsave, data, delimiter=',', header='TIC ID, RA, DEC, PM_RA, PM_DEC, Tmag')

# write to CSV for Web TESS Viewing Tool (WTVT)
pathsave = pathdata + 'WTVT.csv'
print('Writing to %s...' % pathsave)
np.savetxt(pathsave, data[:, 1:3], delimiter=',')
