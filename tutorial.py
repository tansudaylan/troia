import sys
import troia

def cnfg_candidates():
   
    
    typepopl = 'cand'

    troia.init( \
         typepopl=typepopl, \
        )


def cnfg_candidates():
   
    troia.init( \
               liststrgmast=['V723 Mon'], \
               typepopl='Candidates', \
              )


def cnfg_XRB():
   
    troia.init( \
               liststrgmast=['HR6819'], \
               typepopl='XRB', \
              )


def cnfg_Rafael():
   
    troia.init( \
         liststrgmast=['TIC 356069146'], \
        )


# rvel: High RV targets from Gaia DR2
def cnfg_rvel():
   
    troia.init( \
         #listtsec=listtsec, \
         #liststrgmast=['Vela X-1'], \
        )


def cnfg_mock():
   
    listtsec = [9]
    troia.init( \
               #listtsec=listtsec, \
               typedata='mock', \
              )


def cnfg_tsec():
   
    init( \
         tsec=1, \
        )


globals().get(sys.argv[1])()
