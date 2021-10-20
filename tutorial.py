import sys, os

import numpy as np

import troia
from tdpy import summgene


def cnfg_prev():
    '''
    Previous discoveries
    '''    
    
    dictmileinpt = dict()
    dictmileinpt['dictpboxinpt'] = dict()
    dictmileinpt['dictpboxinpt']['factosam'] = 0.1

    troia.init( \
               liststrgmast=['V723 Mon'], \
               typepopl='prev', \
               dictmileinpt=dictmileinpt, \
              )

    
def cnfg_candidates_Rom():
    '''
    Targets from Rom's RNN
    '''
    
    listticitarg = [21266729, 14397653, 12967420, 3892500, 3664978, 1008024, 1066665, 2761086, 3542993]
    troia.init( \
               listticitarg=listticitarg, \
               typepopl='candidates_Rom', \
              )


def cnfg_XRB():
    '''
    XRBs in Kartik's catalog
    '''

    troia.init( \
               liststrgmast=['HR6819', 'Vela X-1'], \
               typepopl='XRB', \
              )


def cnfg_Rafael():
    '''
    Targets from Rafael
    '''
    
    troia.init( \
               liststrgmast=['TIC 356069146'], \
               typepopl='Rafael', \
              )


def cnfg_rvel():
    '''
    High RV targets from Gaia DR2
    '''   
    
    troia.init( \
               typepopl='rvel', \
              )


def cnfg_tessnomi2min():
    '''
    2-minute targets from the nominal mission
    '''
    
    dictmileinpt = dict()
    dictmileinpt['dictpboxinpt'] = dict()
    dictmileinpt['dictpboxinpt']['factosam'] = 1.
    dictmileinpt['dictpboxinpt']['minmperi'] = 1.
    dictmileinpt['dictpboxinpt']['maxmperi'] = 40.
    
    dictlcurtessinpt = dict()
    dictlcurtessinpt['booltpxfonly'] = True
    
    troia.init( \
               typepopl='tessnomi2min', \
               typedata='mock', \
               dictmileinpt=dictmileinpt, \
               dictlcurtessinpt=dictlcurtessinpt, \
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
               typepopl='cycle3_G03254', \
               listticitarg=listticitarg, \
              )


globals().get(sys.argv[1])()


