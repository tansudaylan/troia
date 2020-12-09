import os, fnmatch, sys, json

from tdpy.util import summgene

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import astroquery
from astroquery.mast import Catalogs


from urllib.parse import quote as urlencode
import http.client as httplib 

import datetime
import astropy

from astropy.io import fits

def mastQuery(request):

    server='mast.stsci.edu'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head,content

## read PCAT path environment variable
pathbase = os.environ['BHOL_DATA_PATH'] + '/'
pathdata = pathbase + 'data/'
pathimag = pathbase + 'imag/'

# read Gaia IDs with high radial velocity errors from Saul
print('Reading Sauls Gaia high RV catalog...')
path = pathdata + 'Gaia_high_RV_errors.txt'
for line in open(path):
    listnamesaul = line[:-1].split('\t')
    break
print('Reading from %s...' % path)
dataradv = np.loadtxt(path, skiprows=1)
rascradv = dataradv[:, 0]
declradv = dataradv[:, 1]
stdvradv = dataradv[:, -4]

print('listnamesaul')
print(listnamesaul)
print('rascradv')
summgene(rascradv)
print('declradv')
summgene(declradv)
print('stdvradv')
summgene(stdvradv)
print

# get Gaia RVs
path = pathdata + 'GaiaSource.csv'
liststrgextn = fnmatch.filter(os.listdir(pathdata), 'GaiaSource_[0-9]*.csv')
listobjtarch = []
for strgextn in liststrgextn:
    path = pathdata + strgextn
    print('Reading from %s...' % path)
    for line in open(path):
        listcols = line[:-1].split(',')
        break
    objtarch = pd.read_csv(path, skiprows=0)
    objtarch = objtarch.to_numpy()
    
    listobjtarch.append(objtarch)
    
    # temp
    break

print('listcols')
print(listcols)

arrytotl = np.concatenate(listobjtarch, 0)
print('Total number of entries: %d' % arrytotl.shape[0])
listcols = np.array(listcols)
indxpmra = np.where(listcols == 'pmra')[0][0]
indxpmde = np.where(listcols == 'pmdec')[0][0]
indxradi = np.where(listcols == 'radius_val')[0][0]
indxstdvradv = np.where(listcols == 'radial_velocity_error')[0][0]

prmo = np.sqrt(arrytotl[:, indxpmra].astype(float)**2 + arrytotl[:, indxpmde].astype(float)**2)

minmstdvradv = 15.
maxmradi = 1.
minmprmo = 20.

print('np.where(arrytotl[:, indxstdvradv] > minmstdvradv)[0]')
summgene(np.where(arrytotl[:, indxstdvradv] > minmstdvradv)[0])
print('np.where(arrytotl[:, indxradi] < maxmradi)[0]')
summgene(np.where(arrytotl[:, indxradi] < maxmradi)[0])
print('np.where(prmo > minmprmo)[0]')
summgene(np.where(prmo > minmprmo)[0])

## filter based on RV error
listindxradv = np.where((arrytotl[:, indxstdvradv] > minmstdvradv) & (arrytotl[:, indxradi] < maxmradi))[0]
print('Number of Gaia DR2 sources after the cuts: %d' % listindxradv.size)
arry = np.empty((listindxradv.size, 2))
arry[:, 0] = arrytotl[listindxradv, 5]
arry[:, 1] = arrytotl[listindxradv, 7]
print('arry')
summgene(arry)

numbgaia = arry.shape[0]
indxgaia = np.arange(numbgaia)
dictcoor = [[] for k in indxgaia]
for k in indxgaia:
    dictcoor[k] = dict()
    dictcoor[k]['ra'] = arry[k, 0]
    dictcoor[k]['dec'] = arry[k, 1]

# crossmatch with TIC to get TIC IDs
crossmatchInput = {"fields":[{"name":"ra","type":"float"},
                             {"name":"dec","type":"float"}],
                   #"data":[{"ra":210.8,"dec":54.3}]}
                   "data":dictcoor}
                   #"data":[dictcoor]}

request =  {"service":"Mast.GaiaDR2.Crossmatch",
            "data":crossmatchInput,
            "params":{
                "raColumn":"ra",
                "decColumn":"dec",
                "radius":0.1
            },
            "format":"json"}


headers,outString = mastQuery(request)

outData = json.loads(outString)




# make plot of the Gaia DR2 catalog
if False:
    for k, cols in enumerate(listcols):
        if isinstance(arrytotl[0, k], float) and k != 4:
            figr, axis = plt.subplots(figsize=(12, 4))
            temptotl = arrytotl[:, k].astype(float)
            hist, bins, ptch = axis.hist(temptotl[~np.isnan(temptotl)], bins=100)
            
            #tempsele = arrytotl[listindxarry, k].astype(float)
            #axis.hist(tempsele[~np.isnan(tempsele)], bins=bins, label='Selected')
            
            print('cols')
            print(cols)
            indx = np.where(listnamesaul == cols)[0]
            if indx.size == 1:
                axis.hist(dataradv[:, indx[0]], bins=bins, label='Saul')
    
            axis.set_ylabel('N')
            axis.set_xlabel(cols)
            axis.legend()
            plt.tight_layout()
            axis.set_yscale('log')
            path = pathimag + 'histgaia_%s.png' % (cols)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

numbradv = listindxradv.size
print('numbradv')
print(numbradv)

# write radial velocity errors
pathsave = pathdata + 'gaia.csv'
print('Writing to %s...' % pathsave)
gaia = np.empty((listindxradv.size, 2))
gaia[:, 0] = arrytotl[listindxradv, 5]
gaia[:, 1] = arrytotl[listindxradv, 7]
#gaia[:, 2] = arrytotl[listindxradv, indxstdvradv]
np.savetxt(pathsave, gaia, delimiter=',')

# write RA and DEC for MAST cross match
pathsave = pathdata + 'mast.csv'
print('Writing to %s...' % pathsave)
np.savetxt(pathsave, arry, delimiter=',', header='RAJ2000,DECJ2000')


# plot scatter
figr, axis = plt.subplots(figsize=(5, 3))


xdat = arrytotl[:, indxradi].astype(float)
ydat = arrytotl[:, indxstdvradv].astype(float)
indx = np.where((~np.isnan(xdat)) & (~np.isnan(ydat)))
xdat = xdat[indx]
ydat = ydat[indx]
print('xdat')
summgene(xdat)
bins = [np.logspace(np.amin(np.log10(xdat)), np.amax(np.log10(xdat)), 100), np.logspace(np.amin(np.log10(ydat)), np.amax(np.log10(ydat)), 100)]
axis.hist2d(xdat, ydat, cmap='Blues', bins=bins, norm=mpl.colors.LogNorm())
#axis.scatter(arrytotl[:, indxradi], arrytotl[:, indxstdvradv], marker='o', s=0.05, color='k')
axis.axvline(3., ls='--', color='y')
#axis.axhline(7., ls='--', color='y')
axis.axvline(maxmradi, ls='--', color='r')
axis.axhline(minmstdvradv, ls='--', color='r')
axis.set_ylabel(r'$\sigma_{rv}$ [km/s]')
axis.set_xlabel(r'R [R$_\odot$]')
axis.set_xscale('log')
axis.set_yscale('log')
plt.tight_layout()
path = pathimag + 'stdvradvradi.pdf'
print('Writing to %s...' % path)
plt.savefig(path)
plt.close()



raise Exception('')



listindxarry = []
numbsourrradv = arry.size
indxsourrradv = np.arange(numbsourrradv)
for a in range(2):
    if a == 0:
        strgexpr = 'XMM'
        path = pathdata + 'asu.fit'
    else:
        strgexpr = 'ROSAT'
        path = pathdata + 'asu-2.fit'
    print('Reading the %s Gaia-AllWISE cross-match catalog...' % strgexpr)
    listhdun = astropy.io.fits.open(path)
    listhdun.info()
    dataxmat = listhdun[1].data
    listnamexmat = dataxmat.columns.names
    print('listnamexmat')
    print(listnamexmat)
    listlablxmat = listnamexmat
    listhdun.close()
    
    listindxxmat = []
    for n in indxsourrradv:
        indx = np.where(dataxmat.field('Gaia') == arry[n])[0]
        if indx.size > 1:
            raise Exception('')
        if indx.size > 0:
            print('found match(es)')
            for k, name in enumerate(listnamexmat):
                print(name)
                print(dataxmat[name][indx[0]])
            print
            listindxxmat.append(indx[0])
            listindxarry.append(n)
    listindxxmat = np.array(listindxxmat)
    
    # save cross-matches
    numbxmat = listindxxmat.size
    data = np.empty((numbxmat, 1))
    pathsave = 'data_xmat_%s.csv' % strgexpr
    np.savetxt(pathsave, data, delimiter=',')
    
    # plot all features
    for k, name in enumerate(listnamexmat):
        varb = dataxmat.field(name)
        if not isinstance(varb[0], str) and np.where(np.isfinite(varb))[0].size > 0:
            
            figr, axis = plt.subplots(figsize=(12, 4))
            temptotl = varb
            tempsele = varb[listindxxmat]
            hist, bins, ptch = axis.hist(temptotl[np.where(np.isfinite(temptotl))[0]], bins=100)
            axis.hist(tempsele[np.where(np.isfinite(tempsele))[0]], bins=bins)
            axis.set_ylabel('N')
            axis.set_xlabel(name)
            axis.set_yscale('log')
            plt.tight_layout()
            path = pathimag + 'hist_%s_%s.png' % (strgexpr, name)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

# list of xmat indices in the Gaia high radial velocity catalog 
listindxarry = np.array(listindxarry)

print('listindxarry')
summgene(listindxarry)
print('np.unique(listindxarry)')
summgene(np.unique(listindxarry))
if not np.unique(listindxarry).size == listindxarry.size:
    print('listindxarry is not unique...')
    print('listindxarry')
    print(listindxarry)
    print('np.unique(listindxarry)')
    print(np.unique(listindxarry))
    raise Exception('')


