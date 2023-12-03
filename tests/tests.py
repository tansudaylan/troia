import sys, os

import numpy as np

import troia
from tdpy import summgene


def cnfg_prev():
    '''
    Previous discoveries
    '''    
    
    dictmileinptglob = dict()
    dictmileinptglob['dictpboxinpt'] = dict()
    dictmileinptglob['dictpboxinpt']['factosam'] = 0.1

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


def cnfg_TESSGEO():
    '''
    Simulated analysis for simulatenaous data collected by TESS EM2 and TESS L5
    '''
    
    troia.init( \
               typesyst='PlanetarySystem', \
               listlablinst=[['TGEO-UV', 'TGEO-IR', 'TGEO-VIS'], []], \
               #typepopl='CTL_prms_2min', \
               liststrgtypedata=[['simutargsynt', 'simutargsynt', 'simutargsynt'], []], \
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


def cnfg_TESSGEO_WD( \
                    typesyst='PlanetarySystem', \
                    listlablinst=[['TESS-GEO'], []], \
                    #typepopl='Gaia_WD', \
                    typepopl=None, \
                    liststrgtypedata=[['simutargsynt'], []], \
                   ):
    '''
    generic function to call troia
    '''
    
    dictmileinptglob = dict()
    dictmileinptglob['dictpboxinpt'] = dict()
    #dictmileinptglob['dictpboxinpt']['factosam'] = 1.
    
    # oversampling factor (wrt to transit duration) when rebinning data to decrease the time resolution
    dictmileinptglob['dictpboxinpt']['factduracade'] = 2.
    # factor by which to oversample the frequency grid
    dictmileinptglob['dictpboxinpt']['factosam'] = 10.
    # number of duty cycle samples  
    dictmileinptglob['dictpboxinpt']['numbdcyc'] = 3
    # spread in the logarithm of duty cycle
    dictmileinptglob['dictpboxinpt']['deltlogtdcyc'] = 0.5
    # epoc steps divided by trial duration
    dictmileinptglob['dictpboxinpt']['factdeltepocdura'] = 0.5

    troia.init( \
               typesyst=typesyst, \
               listlablinst=listlablinst, \
               typepopl=typepopl, \
               liststrgtypedata=liststrgtypedata, \
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
    dictmileinptglob['dictpboxinpt'] = dict()
    #dictmileinptglob['dictpboxinpt']['factosam'] = 1.
    
    # oversampling factor (wrt to transit duration) when rebinning data to decrease the time resolution
    dictmileinptglob['dictpboxinpt']['factduracade'] = 2.
    # factor by which to oversample the frequency grid
    dictmileinptglob['dictpboxinpt']['factosam'] = 10.
    # number of duty cycle samples  
    dictmileinptglob['dictpboxinpt']['numbdcyc'] = 3
    # spread in the logarithm of duty cycle
    dictmileinptglob['dictpboxinpt']['deltlogtdcyc'] = 0.5
    # epoc steps divided by trial duration
    dictmileinptglob['dictpboxinpt']['factdeltepocdura'] = 0.5

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
    
    typepopl = 'Synthetic'
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
    
    troia.init( \
               typesyst='PlanetarySystem', \
               typepopl=typepopl, \
               #listticitarg=listticitarg, \
               liststrgtypedata=[['simutargsynt'], []], \
               listlablinst=[['LSST'], []], \
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
