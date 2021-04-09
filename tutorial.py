import sys

import bhol.main

def cnfg_HR6819():
   
    init( \
         liststrgmast=['HR6819'], \
         bdtrtype='medi', \
        )


def cnfg_Rafael():
   
    init( \
         liststrgmast=['TIC 356069146'], \
        )


def cnfg_rvel():
   
    init( \
         #listtsec=listtsec, \
         #liststrgmast=['Vela X-1'], \
        )


def cnfg_mock():
   
    listtsec = [9]
    init( \
         #listtsec=listtsec, \
         typedata='mock', \
        )


def cnfg_tsec():
   
    init( \
         tsec=1, \
        )


globals().get(sys.argv[1])()


