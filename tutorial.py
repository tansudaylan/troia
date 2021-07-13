import sys
import troia


def cnfg_candidates():
   
    troia.init( \
               liststrgmast=['V723 Mon'], \
               typepopl='Candidates', \
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


def cnfg_2minsc17_mock():
   
    troia.init( \
               typepopl='2minsc17', \
               typedata='mock', \
              )


def cnfg_sc17():
   
    troi.init( \
              typepopl='2minsc17', \
             )


globals().get(sys.argv[1])()
