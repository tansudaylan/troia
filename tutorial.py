import sys, os

import numpy as np

import troia

from tdpy import summgene

def cnfg_candidates_Rom():
    
    listticitarg = [21266729, 14397653, 12967420, 3892500, 3664978, 1008024, 1066665, 2761086, 3542993]
    troia.init( \
               #liststrgmast=['V723 Mon'], \
               listticitarg=listticitarg, \
               typepopl='candidates_Rom', \
              )


def cnfg_XRB():
   
    troia.init( \
               liststrgmast=['HR6819', 'Vela X-1'], \
               typepopl='XRB', \
              )


def cnfg_Rafael():
   
    troia.init( \
               liststrgmast=['TIC 356069146'], \
               typepopl='Rafael', \
              )


# rvel: High RV targets from Gaia DR2
def cnfg_rvel():
   
    troia.init( \
               typepopl='rvel', \
              )


def cnfg_tessnomi2minbulk():
   
    troia.init( \
               typepopl='tessnomi2minbulk', \
               typedata='mock', \
              )


def cnfg_cycle3_G03254():
    
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
               typepopl='cycle3_G03254', \
               listticitarg=listticitarg, \
              )


globals().get(sys.argv[1])()


