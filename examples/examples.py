import sys, os

import numpy as np

import troia
from tdpy import summgene


def cnfg_Keck():
    '''
    Keck survey
    '''    

    path = os.environ['TROIA_DATA_PATH'] + '/data/Keck_20240311.txt'
    
    listtoiitarg = np.loadtxt(path)

    dictmileinptglob = dict()
    #dictmileinptglob['dictboxsperiinpt'] = dict()
    #dictmileinptglob['dictboxsperiinpt']['factosam'] = 0.1

    troia.init( \
               typesyst='CompactObjectStellarCompanion', \
               listtoiitarg=listtoiitarg, \
               listlablinst=[['TESS'], []], \
               #typepopl='prev', \
               dictmileinptglob=dictmileinptglob, \
              )

    
def cnfg_prev():
    '''
    Previous discoveries
    '''    
    
    dictmileinptglob = dict()
    dictmileinptglob['dictboxsperiinpt'] = dict()
    dictmileinptglob['dictboxsperiinpt']['factosam'] = 0.1

    troia.init( \
               typesyst='CompactObjectStellarCompanion', \
               liststrgmast=['V723 Mon', 'VFTS 243', 'HR 6819', 'A0620-00'], \
               listlablinst=[['TESS'], []], \
               typepopl='prev', \
               dictmileinptglob=dictmileinptglob, \
              )

    
def cnfg_Flares():
    '''
    Simulated flaring stars observed by ULTRASAT
    '''
    
    troia.init( \
               typesyst='StarFlaring', \
               listlablinst=[['ULTRASAT'], []], \
               typepopl='CTL_S1_2min', \
               liststrgtypedata=[['simutargpartsynt'], []], \
              )


def cnfg_candidates_Rom():
    '''
    Targets from Rom's RNN
    '''
    
    dictmileinptglob = dict()
    dictmileinptglob['typelcurtpxftess'] = 'SPOC'
    
    listticitarg = [21266729, 14397653, 12967420, 3892500, 3664978, 1008024, 1066665, 2761086, 3542993]
    troia.init( \
               typesyst='CompactObjectStellarCompanion', \
               listticitarg=listticitarg, \
               typepopl='candidates_Rom', \
               dictmileinptglob=dictmileinptglob, \
               listlablinst=[['TESS'], []], \
              )


def cnfg_XRB():
    '''
    XRBs in Kartik's catalog
    '''

    troia.init( \
               typesyst='CompactObjectStellarCompanion', \
               liststrgmast=['HR6819', 'Vela X-1'], \
               typepopl='XRB', \
              )


def cnfg_TargetsOfInterest():
    
    # black hole candidate from Eli's list
    listticitarg = [281562429]
    # Rafael
    listticitarg = [356069146]


def cnfg_rvel():
    '''
    High RV targets from Gaia DR2
    '''   
    
    troia.init( \
               typesyst='CompactObjectStellarCompanion', \
               typepopl='rvel', \
              )


def cnfg_TESSGEO_PlanetPlanet():
    '''
    TESS-GEOSimulated analysis for simulatenaous data collected by TESS-GEO
    '''
    
    dictpoplsystinpt = dict()
    dictpoplsystinpt['booltrancomp'] = True
    
    strgcnfg = 'TESSGEO_PlanetPlanet'
    troia.init( \
               strgcnfg=strgcnfg, \
               typesyst='PlanetarySystem', \
               listlablinst=[['TESS-GEO-UV', 'TESS-GEO-VIS'], []], \
               liststrgtypedata=[['simutargsynt', 'simutargsynt'], []], \
               dictpoplsystinpt=dictpoplsystinpt, \
               #typepopl='CTL_prms_2min', \
              )


def cnfg_TESSGEO_WD( \
                    typesyst='PlanetarySystem', \
                    listlablinst=[['TESS-GEO-UV', 'TESS-GEO-VIS'], []], \
                    liststrgtypedata=[['simutargsynt', 'simutargsynt'], []], \
                    typepopl=None, \
                    #typepopl='Gaia_WD', \
                   ):
    '''
    generic function to call troia
    '''
    
    dictmileinptglob = dict()
    dictmileinptglob['dictboxsperiinpt'] = dict()
    #dictmileinptglob['dictboxsperiinpt']['factosam'] = 1.
    
    # oversampling factor (wrt to transit duration) when rebinning data to decrease the time resolution
    dictmileinptglob['dictboxsperiinpt']['factduracade'] = 2.
    # factor by which to oversample the frequency grid
    dictmileinptglob['dictboxsperiinpt']['factosam'] = 10.
    # spread in the logarithm of duty cycle
    dictmileinptglob['dictboxsperiinpt']['deltlogtdcyc'] = 0.5
    # epoc steps divided by trial duration
    dictmileinptglob['dictboxsperiinpt']['factdeltepocdura'] = 0.5

    dictpoplsystinpt = dict()
    dictpoplsystinpt['typestar'] = 'wdwf'
    dictpoplsystinpt['maxmnumbcompstar'] = 1
    
    strgcnfg = 'TESSGEO_WD'
    
    troia.init( \
               strgcnfg=strgcnfg, \
               typesyst=typesyst, \
               listlablinst=listlablinst, \
               typepopl=typepopl, \
               liststrgtypedata=liststrgtypedata, \
               dictpoplsystinpt=dictpoplsystinpt, \
               dictmileinptglob=dictmileinptglob, \
              )


def cnfg_TESS_BH( \
              typesyst='CompactObjectStellarCompanion', \
              listlablinst=[['TESS'], []], \
              typepopl='CTL_prms_2min', \
              liststrgtypedata=[['simutargpartsynt'], []], \
             ):
    '''
    generic function to call troia for self-lensing black holes
    '''
    
    dictmileinptglob = dict()
    dictmileinptglob['dictboxsperiinpt'] = dict()
    #dictmileinptglob['dictboxsperiinpt']['factosam'] = 1.
    
    # oversampling factor (wrt to transit duration) when rebinning data to decrease the time resolution
    dictmileinptglob['dictboxsperiinpt']['factduracade'] = 2.
    # factor by which to oversample the frequency grid
    dictmileinptglob['dictboxsperiinpt']['factosam'] = 10.
    # spread in the logarithm of duty cycle
    dictmileinptglob['dictboxsperiinpt']['deltlogtdcyc'] = 0.5
    # epoc steps divided by trial duration
    dictmileinptglob['dictboxsperiinpt']['factdeltepocdura'] = 0.5

    troia.init( \
               typesyst=typesyst, \
               listlablinst=listlablinst, \
               typepopl=typepopl, \
               liststrgtypedata=liststrgtypedata, \
               dictmileinptglob=dictmileinptglob, \
              )


def cnfg_LSST_PlanetarySystem():
    '''
    LSST transiting exoplanet survey
    '''
    
    typepopl = 'SyntheticPopulation'
    if typepopl == 'TIC':
        listticitarg = []
        k = 0
        for line in objtfile:
            if k != 0 and not line.startswith('#') and line != '\n':
                strgtici = line.split(',')[0]
                listticitarg.append(strgtici)
            k += 1
            if len(listticitarg) == 10:
                break
        listticitarg = np.array(listticitarg)
        listticitarg = listticitarg.astype(int)
        listticitarg = np.unique(listticitarg)
    
    liststrgtypedata = [[], []]
    listlablinst = [[], []]
    liststrglsst = ['u', 'g', 'r', 'i', 'z', 'y']
    for strglsst in liststrglsst:
        listlablinst[0].append('LSST %s band' % strglsst)
        liststrgtypedata[0].append('simutargsynt')
    
    dictpoplsystinpt = dict()
    dictpoplsystinpt['typestar'] = 'wdwf'
    
    dictmileinptglob = dict()
    dictmileinptglob['typepriocomp'] = 'outlperi'
    #dictmileinptglob['dictboxsperiinpt']['factosam'] = 0.1
    
    troia.init( \
               typesyst='PlanetarySystem', \
               typepopl=typepopl, \
               liststrgtypedata=liststrgtypedata, \
               listlablinst=listlablinst, \
               dictpoplsystinpt=dictpoplsystinpt, \
               dictmileinptglob=dictmileinptglob, \
              )


def cnfg_cycle3_G03254():
    '''
    Targets in TESS GI Cycle 3 proposal (PI: Tansu Daylan, G03254)
    '''
    
    path = os.environ['TROIA_DATA_PATH'] + '/data/G03254_Cycle3_GI.csv'
    objtfile = open(path, 'r')
    listticitarg = []
    k = 0
    for line in objtfile:
        if k != 0 and not line.startswith('#') and line != '\n':
            strgtici = line.split(',')[0]
            listticitarg.append(strgtici)
        k += 1
        if len(listticitarg) == 10:
            break
    listticitarg = np.array(listticitarg)
    listticitarg = listticitarg.astype(int)
    listticitarg = np.unique(listticitarg)
    troia.init( \
               typesyst='CompactObjectStellarCompanion', \
               typepopl='cycle3_G03254', \
               listticitarg=listticitarg, \
               listlablinst=[['TESS'], []], \
              )


globals().get(sys.argv[1])(*sys.argv[2:])
