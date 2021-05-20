import sys
import troia

def cnfg_candidates():
   
    
    typepopl = 'cand'

    troia.init( \
         typepopl=typepopl, \
        )


def cnfg_HR6819():
   
    troia.init( \
         liststrgmast=['HR6819'], \
        )


def cnfg_Rafael():
   
    troia.init( \
         liststrgmast=['TIC 356069146'], \
        )


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
