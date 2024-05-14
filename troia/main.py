import os, sys, datetime, copy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.interpolate

from tdpy.util import summgene
import tdpy
import ephesos
import miletos
import pergamon
import nicomedia
import chalcedon


def retr_dictderi_effe(para, gdat):
    
    radistar = para[0]
    peri = para[1]
    masscomp = para[2]
    massstar = para[3]
    masstotl = massstar + masscomp

    amplslenmodl = chalcedon.retr_amplslen(peri, radistar, masscomp, massstar)
    duratrantotlmodl = ephesos.retr_duratrantotl(peri, radistar, masscomp, massstar, incl)
    smax = ephesos.retr_smaxkepl(peri, masstotl) * 215. # [R_S]
    radischw = 4.24e-6 * masscomp # [R_S]

    dictvarbderi = None

    dictparaderi = dict()
    dictparaderi['amplslenmodl'] = np.array([amplslenmodl])
    dictparaderi['duratrantotlmodl'] = np.array([duratrantotlmodl])
    dictparaderi['smaxmodl'] = np.array([smax])
    dictparaderi['radischw'] = np.array([radischw])

    return dictparaderi, dictvarbderi
    

def mile_work(gdat, i):
    
    for n in gdat.listindxtarg[i]:
        
        if gdat.boolsimusome:
            for v in gdat.indxtyperele:
                if n in gdat.dictindxtarg['rele'][v]:
                    gdat.boolreletarg[v][n] = True
                else:
                    gdat.boolreletarg[v][n] = False

        if gdat.typepopl == 'SyntheticPopulation':
            #listarrytser = dict()
            #listarrytser['raww'] = gdat.listarrytser['data'][n]
            
            rasctarg = None
            decltarg = None
            strgmast = None
            labltarg = gdat.labltarg[n]
            strgtarg = gdat.strgtarg[n]
            toiitarg = None
            if len(strgtarg) == 0:
                raise Exception('')
        
        else:
            if gdat.typepopl == 'MASTKeywords':
                rasctarg = None
                decltarg = None
                strgmast = gdat.liststrgmast[n]
                labltarg = None
                strgtarg = None
                toiitarg = None

            elif gdat.booldataobsv and gdat.typepopl == 'TOIs':
                rasctarg = None
                decltarg = None
                strgmast = None
                toiitarg = gdat.listtoiitarg[n]
                labltarg = None
                strgtarg = None
        
        gdat.dictmileinpttarg = copy.deepcopy(gdat.dictmileinptglob)

        if n < gdat.maxmnumbtargplot:
            gdat.dictmileinpttarg['boolplot'] = gdat.boolplotmile
        else:
            gdat.dictmileinpttarg['boolplot'] = False
        gdat.dictmileinpttarg['rasctarg'] = rasctarg
        gdat.dictmileinpttarg['decltarg'] = decltarg
        gdat.dictmileinpttarg['strgtarg'] = strgtarg
        gdat.dictmileinpttarg['toiitarg'] = toiitarg
        gdat.dictmileinpttarg['labltarg'] = labltarg
        gdat.dictmileinpttarg['strgmast'] = strgmast
        gdat.dictmileinpttarg['boolanls'] = True
        
        #gdat.dictmileinpttarg['listarrytser'] = listarrytser
        
        dictmagtsyst = dict()
        
        dicttrue = dict()
        dicttrue['numbyearlsst'] = 1
        dicttrue['typemodl'] = 'PlanetarySystem'
        
        if gdat.boolsimusome:
            for namepara in gdat.dicttroy['true']['PlanetarySystem']['listnamefeatbody']:
                dicttrue[namepara] = gdat.dicttroy['true']['PlanetarySystem']['dictpopl']['star'][gdat.namepoplstartotl][namepara][0][n]
            for namepara in gdat.dicttroy['true']['PlanetarySystem']['listnamefeatlimbonly']:
                dicttrue[namepara] = gdat.dicttroy['true']['PlanetarySystem']['dictpopl']['comp'][gdat.namepoplcomptotl][namepara][0][gdat.indxcompsyst[n]]
            
            gdat.dictmileinpttarg['dicttrue'] = dicttrue
        
            for b in gdat.indxdatatser:
                for strginst in gdat.listlablinst[b]:
                    dictmagtsyst[strginst] = dicttrue['magtsyst' + strginst]
            gdat.dictmileinpttarg['dictmagtsyst'] = dictmagtsyst
        
        # call miletos to analyze data
        print('Calling miletos...')
        dictmileoutp = miletos.init( \
                                    **gdat.dictmileinpttarg, \
                                   )
        if n == 0:
            gdat.listlablposi = []
            gdat.listlablnega = []
            if dictmileoutp['boolcalclspe']:
                gdat.listlablposi.append('High LS power')
                gdat.listlablnega.append('Weak LS power')
            if dictmileoutp['boolsrchboxsperi']:
                gdat.listlablposi.append('High BLS power')
                gdat.listlablnega.append('Weak BLS power')
            if dictmileoutp['boolsrchoutlperi']:
                gdat.listlablposi.append('Low min$_k$ $f_k$')
                gdat.listlablnega.append('High min$_k$ $f_k$')
            
            if gdat.booldiag:
                if len(gdat.listlablposi) != len(gdat.listlablnega):
                    print('')
                    print('')
                    print('')
                    raise Exception('len(gdat.listlablposi) != len(gdat.listlablnega)')
            
            gdat.numbtypeposi = len(gdat.listlablposi)
            gdat.indxtypeposi = np.arange(gdat.numbtypeposi)
            gdat.boolpositarg = [np.empty(gdat.numbtarg, dtype=bool) for u in gdat.indxtypeposi]

            if gdat.boolsimusome:
                gdat.boolreleposi = [[[] for v in gdat.indxtyperele] for u in gdat.indxtypeposi]
                gdat.boolposirele = [[[] for v in gdat.indxtyperele] for u in gdat.indxtypeposi]
        
            # output features of miletos
            gdat.dictstat = dict()
            
            # list of statistics to be collected
            gdat.listnamefeatstat = []
            if dictmileoutp['boolcalclspe']:
                gdat.listnamefeatstat += ['perilspeprim', 'powrlspeprim']
            if dictmileoutp['boolsrchboxsperi']:
                gdat.listnamefeatstat += ['peripboxprim', 's2nrpboxprim']
            if dictmileoutp['boolsrchoutlperi']:
                gdat.listnamefeatstat += ['minmfrddtimeoutlsort']
                
            for namefeat in gdat.listnamefeatstat:
                gdat.dictstat[namefeat] = [np.empty(gdat.numbtarg), '']
        
        if dictmileoutp['boolcalclspe']:
            gdat.dictstat['perilspeprim'][0][n] = dictmileoutp['perilspempow']
            gdat.dictstat['powrlspeprim'][0][n] = dictmileoutp['powrlspempow']
        if dictmileoutp['boolsrchboxsperi']:
            gdat.dictstat['s2nrpboxprim'][0][n] = dictmileoutp['dictboxsperioutp']['s2nr'][0]
            gdat.dictstat['peripboxprim'][0][n] = dictmileoutp['dictboxsperioutp']['peri'][0]
        if dictmileoutp['boolsrchoutlperi']:
            gdat.dictstat['minmfrddtimeoutlsort'][0][n] = dictmileoutp['dictoutlperi']['minmfrddtimeoutlsort'][0]
        
        # taking the fist element, which belongs to the first TCE
        for u in gdat.indxtypeposi:
            gdat.boolpositarg[u][n] = dictmileoutp['boolposianls'][u]
        
        if gdat.boolsimusome:
            for u in gdat.indxtypeposi:
                for v in gdat.indxtyperele:
                    if gdat.boolreletarg[v][n]:
                        if gdat.boolpositarg[u][n]:
                            gdat.boolposirele[u][v].append(True)
                        else:
                            gdat.boolposirele[u][v].append(False)
                    if gdat.boolpositarg[u][n]:
                        if gdat.boolreletarg[v][n]:
                            gdat.boolreleposi[u][v].append(True)
                        else:
                            gdat.boolreleposi[u][v].append(False)
                
    return gdat


def init( \
        
        # type of the system
        typesyst, \

        # a string distinguishing the run to be used in the file names
        strgcnfg=None, \
         
        # type of population of target sources
        typepopl=None, \

        # list of target TIC IDs
        listticitarg=None, \
        
        # list of target TOIs
        listtoiitarg=None, \
        
        # type of instrument
        listlablinst=None, \

        # list of MAST keywords
        liststrgmast=None, \

        # common with miletos
        # type of data for each data kind, instrument, and chunk
        ## 'simutargsynt': simulated data on a synthetic target
        ## 'simutargpartsynt': simulated data on a particular target over a synthetic temporal footprint
        ## 'simutargpartfprt': simulated data on a particular target over the observed temporal footprint 
        ## 'simutargpartinje': simulated data obtained by injecting a synthetic signal on observed data on a particular target with a particular observational baseline 
        ## 'obsd': observed data on a particular target
        liststrgtypedata=None, \
            
        # list of GAIA IDs
        listgaid=None, \

        # Boolean flag to turn on multiprocessing
        boolprocmult=False, \
        
        # the path in which the run folder will be placed
        pathbase=None, \
        
        # input dictionary to miletos
        dictmileinptglob=dict(), \

        # input dictionary to retr_lcurtess()
        #dictlcurtessinpt=None, \

        # Boolean flag to make plots
        boolplot=True, \
        
        # Boolean flag to make initial plots
        boolplotinit=None, \
        
        # Boolean flag to make initial plots
        boolplotmile=None, \
        
        # input dictionary to the population generator
        dictpoplsystinpt=None, \

        # Boolean flag to turn on diagnostic mode
        booldiag=True, \

        # Boolean flag to force rerun and overwrite previous data and plots 
        boolwritover=True, \
        
        # type of verbosity
        ## -1: absolutely no text
        ##  0: no text output except critical warnings
        ##  1: minimal description of the execution
        ##  2: detailed description of the execution
        typeverb=1, \

        ):
    
    # construct global object
    gdat = tdpy.gdatstrt()
    
    # copy locals (inputs) to the global object
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
   
    print('troia initialized at %s...' % gdat.strgtimestmp)
    
    if gdat.boolplotinit is None:
        gdat.boolplotinit = gdat.boolplot
    
    if gdat.boolplotmile is None:
        gdat.boolplotmile = gdat.boolplot
    
    # check inputs
    if not gdat.boolplot and (gdat.boolplotinit or gdat.boolplotmile):
        raise Exception('')

    if gdat.liststrgmast is not None and gdat.listticitarg is not None or \
       gdat.liststrgmast is not None and gdat.listtoiitarg is not None or \
       gdat.listticitarg is not None and gdat.listtoiitarg is not None:
        raise Exception('liststrgmast, listticitarg and listtoiitarg cannot be defined simultaneously.')

    gdat.booltargusertoii = gdat.listtoiitarg is not None
    gdat.booltargusertici = gdat.listticitarg is not None
    gdat.booltargusermast = gdat.liststrgmast is not None
    gdat.booltargusergaid = gdat.listgaid is not None
    gdat.booltarguser = gdat.booltargusertici or gdat.booltargusermast or gdat.booltargusergaid or gdat.booltargusertoii
    
    # prepare miletos gdat for setup
    gdat.boolsimutargpartfprt = None

    if gdat.listlablinst is None:
        print('')
        print('')
        print('')
        raise Exception('gdat.listlablinst is None. gdat.listlablinst must be defined.')

    gdat.listlablinst = miletos.retr_strginst(gdat.listlablinst)

    miletos.setup_miletos(gdat)

    miletos.setup1_miletos(gdat)
    
    gdat.booltargsynt = not gdat.booltarguser

    if gdat.booldiag:
        if gdat.booltargsynt:
            if gdat.liststrgtypedata[0][0] == 'simutargpartinje':
                print('')
                print('')
                print('')
                print('gdat.booltarguser')
                print(gdat.booltarguser)
                print('gdat.listtoiitarg')
                print(gdat.listtoiitarg)
                print('gdat.listticitarg')
                print(gdat.listticitarg)
                print('gdat.liststrgmast')
                print(gdat.liststrgmast)
                print('gdat.listgaid')
                print(gdat.listgaid)
                print('gdat.booltargsynt')
                print(gdat.booltargsynt)
                print('gdat.liststrgtypedata')
                print(gdat.liststrgtypedata)
                print('gdat.typepopl')
                print(gdat.typepopl)
                print('gdat.booltargsynt and (gdat.liststrgtypedata[0][0] == simutargpartinje or gdat.typepopl != SyntheticPopulation.')
                raise Exception('Either gdat.listtoiitarg, gdat.listticitarg, gdat.liststrgmast, or gdat.listgaid should be defined.')

            if gdat.booltarguser or not gdat.booltargsynt and not gdat.booltarguser:
                print('')
                print('')
                print('')
                print('gdat.booltargsynt')
                print(gdat.booltargsynt)
                print('gdat.booltarguser')
                print(gdat.booltarguser)
                print('gdat.booltarguser')
                print(gdat.booltarguser)
                print('gdat.typepopl')
                print(gdat.typepopl)
                raise Exception('gdat.booltargsynt and gdat.booltarguser or not gdat.booltargsynt and not gdat.booltarguser and gdat.typepopl is None')

    if (gdat.liststrgmast is not None or listticitarg is not None) and gdat.typepopl is None:
        raise Exception('The type of population, typepopl, must be defined by the user when the target list is provided by the user')
    
    if gdat.typepopl is None:
        if gdat.booltarguser:
            if gdat.booltargusertoii:
                gdat.typepopl = 'TOIs'
            elif gdat.booltargusertici:
                gdat.typepopl = 'TICs'
            elif gdat.booltargusermast:
                gdat.typepopl = 'MASTKeywords'
            elif gdat.booltargusergaid:
                gdat.typepopl = 'GaiaIDs'
            else:
                raise Exception('')
        else:
            gdat.typepopl = 'SyntheticPopulation'
    
    #        gdat.typepopl = 'CTL_prms_2min'
    #        gdat.typepopl = 'CTL_prms_2min'
    
    print('gdat.typepopl')
    print(gdat.typepopl)

    # paths
    ## path of the troia data folder
    gdat.pathbasetroy = os.environ['TROIA_DATA_PATH'] + '/'
    ## base path of the run
    if gdat.pathbase is None:
        gdat.pathbase = gdat.pathbasetroy
    
    gdat.pathdatapipe = gdat.pathbase + 'data/'
    gdat.pathvisupipe = gdat.pathbase + 'visuals/'
    
    gdat.strginstconc = ''
    k = 0
    for b in gdat.indxdatatser:
        for p in range(len(gdat.listlablinst[b])):
            if k > 0:
                gdat.strginstconc += '_'
            gdat.strginstconc += '%s' % gdat.listlablinst[b][p]
            k += 1
    
    if gdat.strgcnfg is None:
        gdat.strgextncnfg = ''
    else:
        gdat.strgextncnfg = '%s_' % gdat.strgcnfg
    gdat.strgextn = '%s%s_%s' % (gdat.strgextncnfg, gdat.typepopl, gdat.strginstconc)
    
    gdat.pathpopl = gdat.pathbase + gdat.strgextn + '/'
    gdat.pathvisucnfg = gdat.pathpopl + 'visuals/'
    gdat.pathdatacnfg = gdat.pathpopl + 'data/'

    # make folders
    for attr, valu in gdat.__dict__.items():
        if attr.startswith('path'):
            os.system('mkdir -p %s' % valu)

    # settings
    ## seed
    np.random.seed(0)
    
    ## plotting
    gdat.typefileplot = 'png'
    gdat.timeoffs = 2457000.
    gdat.boolanimtmpt = False
    
    if not gdat.booltarguser and not gdat.booltargsynt:
        dicttic8 = nicomedia.retr_dictpopltic8(typepopl=gdat.typepopl)
        
    # number of time-series data sets
    gdat.numbdatatser = 2
    gdat.indxdatatser = np.arange(gdat.numbdatatser)

    # maximum number of targets to plot
    gdat.maxmnumbtargplot = 50

    gdat.numbinst = np.empty(gdat.numbdatatser, dtype=int)
    gdat.indxinst = [[] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        gdat.numbinst[b] = len(gdat.listlablinst[b])
        gdat.indxinst[b] = np.arange(gdat.numbinst[b])
    
    # determine number of targets
    ## number of targets
    if gdat.booltarguser:
        if gdat.booltargusertici:
            gdat.numbtarg = len(gdat.listticitarg)
        elif gdat.booltargusertoii:
            gdat.numbtarg = len(gdat.listtoiitarg)
        elif gdat.booltargusermast:
            gdat.numbtarg = len(gdat.liststrgmast)
        elif gdat.booltargusergaid:
            gdat.numbtarg = len(gdat.listgaidtarg)
        else:
            raise Exception('')
    else:
        if gdat.boolsimusome:
            gdat.numbtarg = 30000
        else:
            gdat.numbtarg = dicttic8['TICID'].size
            
        gdat.numbtarg = 100
    
    print('Number of targets: %s' % gdat.numbtarg)
    gdat.indxtarg = np.arange(gdat.numbtarg)
    
    if not gdat.booltarguser and not gdat.booltargsynt:
        size = dicttic8['TICID'].size
        indx = np.random.choice(np.arange(dicttic8['TICID'].size), replace=False, size=size)
        for name in dicttic8.keys():
            dicttic8[name] = dicttic8[name][indx]
    
    gdat.timeexectarg = 120.
    print('Expected execution time: %g seconds (%.3g days, %.3g weeks for 20M targets)' % (gdat.numbtarg * gdat.timeexectarg, gdat.numbtarg * gdat.timeexectarg / 3600. / 24., \
                                                                                                                        20e6 * gdat.timeexectarg / 3600. / 24. / 7.))
    
    if gdat.listticitarg is None:
        gdat.listticitarg = [[] for k in gdat.indxtarg]
    
    if not gdat.booltarguser and not gdat.booltargsynt:
        gdat.listticitarg = dicttic8['TICID']
    
    print('gdat.boolplot')
    print(gdat.boolplot)
    print('gdat.boolplotinit')
    print(gdat.boolplotinit)
    
    # target labels and file name extensions
    gdat.strgtarg = [[] for n in gdat.indxtarg]
    if gdat.liststrgmast is None:
        gdat.liststrgmast = [[] for n in gdat.indxtarg]
    gdat.labltarg = [[] for n in gdat.indxtarg]
    
    for n in gdat.indxtarg:
        if gdat.booltargsynt:
            gdat.strgtarg[n] = 'SimulatedSyntheticTarget%04d' % n
            gdat.labltarg[n] = 'Simulated target'
        else:
            if gdat.typepopl[4:12] == 'prms2min':
                gdat.strgtarg[n] = 'TIC%d' % (gdat.listticitarg[n])
                gdat.labltarg[n] = 'TIC %d' % (gdat.listticitarg[n])
                gdat.liststrgmast[n] = gdat.labltarg[n]
            elif gdat.booltargusertici:
                gdat.labltarg[n] = 'TIC ' + str(gdat.listticitarg[n])
                gdat.strgtarg[n] = 'TIC' + str(gdat.listticitarg[n])
            elif gdat.booltargusertoii:
                gdat.labltarg[n] = 'TOI ' + str(gdat.listtoiitarg[n])
                gdat.strgtarg[n] = 'TOI' + str(gdat.listtoiitarg[n])
            elif gdat.booltargusermast:
                gdat.labltarg[n] = gdat.liststrgmast[n]
                gdat.strgtarg[n] = ''.join(gdat.liststrgmast[n].split(' '))
            elif gdat.booltargusergaid:
                gdat.labltarg[n] = 'GID=' % (dictcatlrvel['rasc'][n], dictcatlrvel['decl'][n])
                gdat.strgtarg[n] = 'R%.4gDEC%.4g' % (dictcatlrvel['rasc'][n], dictcatlrvel['decl'][n])
            else:
                raise Exception('')

    gdat.listarrytser = dict()
    
    if gdat.boolsimusome:
    
        #gdat.listlablrele = ['Simulated %s' % gdat.typesyst, 'Simulated tr. COSC' % gdat.typesyst]
        #gdat.listlablirre = ['Simulated QS or SB', 'Simulated QS, SB or non-tr. COSC']
        
        # labels of the relevant classes
        gdat.listlablrele = ['Simulated %s' % gdat.typesyst]
        
        # labels of the irrelevant classes
        gdat.listlablirre = ['Simulated no signal']
    
        # number of relevant types
        gdat.numbtyperele = len(gdat.listlablrele)
        gdat.indxtyperele = np.arange(gdat.numbtyperele)
        gdat.boolreletarg = [np.empty(gdat.numbtarg, dtype=bool) for v in gdat.indxtyperele]
    
    gdat.dictindxtarg = dict()
    gdat.dicttroy = dict()
    
    if gdat.boolsimusome:
        
        listcolrtypetrue = np.array(['g', 'b', 'orange', 'olive', 'yellow'])
        
        gdat.dictprobtypetrue = dict()
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            gdat.dictprobtypetrue['CompactObjectStellarCompanion'] = [0.70, 'Compact object with Stellar Companion']
            gdat.dictprobtypetrue['StellarBinary'] = [0.25, 'Stellar Binary']
            gdat.dictprobtypetrue['Asteroid'] = [0.05, 'Asteroid']
        elif gdat.typesyst == 'PlanetarySystem':
            gdat.dictprobtypetrue['PlanetarySystem'] = [0.75, 'Planetary System']
            gdat.dictprobtypetrue['StellarBinary'] = [0.25, 'Stellar Binary']
        elif gdat.typesyst == 'StarFlaring':
            gdat.dictprobtypetrue['StellarFlare'] = [1., 'Stellar binary']
        else:
            print('')
            print('')
            print('')
            print('gdat.typesyst')
            print(gdat.typesyst)
            raise Exception('')
        
        # names of simulated classes of systems
        gdat.listnameclastype = list(gdat.dictprobtypetrue.keys())
        
        # number of simulated classes
        gdat.numbtypetrue = len(gdat.listnameclastype)
        indxtypetrue = np.arange(gdat.numbtypetrue)
        
        # probabilities of simulated classes
        gdat.probtypetrue = np.empty(gdat.numbtypetrue)
        gdat.listlabltypetrue = [[] for k in indxtypetrue]
        for k, name in enumerate(gdat.listnameclastype):
            gdat.probtypetrue[k] = gdat.dictprobtypetrue[name][0]
            gdat.listlabltypetrue[k] = gdat.dictprobtypetrue[name][1]

        listcolrtypetrue = listcolrtypetrue[indxtypetrue]
        
        #numbpoplcomp = numbnameincl * numbtypetrue
        #indxpoplcomp = np.arange(numbpoplcomp)

        #listnameincl = ['All', 'Transiting']
        #numbnameincl = len(listnameincl)
        #indxnameincl = np.arange(numbnameincl)
        #listnamepoplcomp = [[] for rh in indxpoplcomp]
        #for k in indxtypetrue: 
        #    for oi in indxnameincl:
        #        rh = k * numbnameincl + oi
        #        listnamepoplcomp[rh] = '%s_%s_%s' % (listnametypetrue[k], gdat.typepopl, listnameincl[k])
        
        listdictlablcolrpopl = []
        
        gdat.dicttroy['true'] = dict()
        
        listdictlablcolrpopl.append(dict())
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            listdictlablcolrpopl[-1]['PlanetarySystem_%s_All' % gdat.typepopl] = ['All', 'black']
            listdictlablcolrpopl[-1]['PlanetarySystem_%s_Transiting' % gdat.typepopl] = ['Transiting', 'b']
        elif gdat.typesyst == 'PlanetarySystem':
            listdictlablcolrpopl[-1]['PlanetarySystem_%s_All' % gdat.typepopl] = ['All', 'black']
            listdictlablcolrpopl[-1]['PlanetarySystem_%s_Transiting' % gdat.typepopl] = ['Transiting', 'b']
        elif gdat.typesyst == 'StarFlaring':
            listdictlablcolrpopl[-1]['StarFlaring_%s_All' % gdat.typepopl] = ['All', 'black']
            listdictlablcolrpopl[-1]['StarFlaring_%s_Mdwarfs' % gdat.typepopl] = ['M dwarfs', 'red']
        else:
            raise Exception('')

        #for namepoplcomm in listnametypetrue:
        #    gdat.dicttroy['true'][namepoplcomm] = dict()

        gdat.numbtypetrue = len(gdat.listnameclastype)
        gdat.indxtypetrue = np.arange(gdat.numbtypetrue)
        
        gdat.typetruetarg = np.random.choice(gdat.indxtypetrue, size=gdat.numbtarg, p=gdat.probtypetrue)
            
        gdat.indxtypetruetarg = [[] for r in gdat.indxtypetrue]
        for r in gdat.indxtypetrue:
            gdat.indxtypetruetarg[r] = np.where(gdat.typetruetarg == r)[0]
        
        gdat.booltypetargtrue = dict()
        gdat.numbtargtype = dict()
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            gdat.booltypetargtrue['CompactObjectStellarCompanion'] = gdat.typetruetarg == 0
            gdat.booltypetargtrue['StellarBinary'] = gdat.typetruetarg == 1
            
            gdat.dictindxtarg['CompactObjectStellarCompanion'] = gdat.indxtypetruetarg[0]
            gdat.dictindxtarg['StellarBinary'] = gdat.indxtypetruetarg[1]
            gdat.dictindxtarg['Asteroid'] = gdat.indxtypetruetarg[2]
            
        elif gdat.typesyst == 'PlanetarySystem':
            gdat.dictindxtarg['PlanetarySystem'] = np.arange(gdat.numbtarg)
            gdat.dictindxtarg['StellarBinary'] = np.arange(gdat.numbtarg)
            
        print('gdat.numbtargtype')
        for name in gdat.listnameclastype:
            gdat.numbtargtype[name] = gdat.dictindxtarg[name].size
            print(name)
            print(gdat.numbtargtype[name])

        print('gdat.numbtarg')
        print(gdat.numbtarg)
        #if gdat.typesyst == 'CompactObjectStellarCompanion':
        #    
        #    cntrcosc = 0
        #    cntrsbin = 0
        #    cntrssys = 0
        #    for k in gdat.indxtarg:
        #        if gdat.boolcosctrue[k] or gdat.boolsbintrue[k]:
        #            if gdat.boolcosctrue[k]:
        #                gdat.dictindxcosc['StellarBinary'][cntrssys] = cntrcosc
        #                gdat.indxssyscosc[cntrcosc] = cntrssys
        #                cntrcosc += 1
        #            if gdat.boolsbintrue[k]:
        #                gdat.indxsbinssys[cntrssys] = cntrsbin
        #                gdat.indxssyssbin[cntrsbin] = cntrssys
        #                cntrsbin += 1
        #            gdat.indxssystarg[k] = cntrssys
        #            cntrssys += 1
        
        if gdat.dictpoplsystinpt is None:
            gdat.dictpoplsystinpt = dict()
        if not 'booltrancomp' in gdat.dictpoplsystinpt:
            gdat.dictpoplsystinpt['booltrancomp'] = True
        gdat.dictpoplsystinpt['typepoplsyst'] = gdat.typepopl
        gdat.dictpoplsystinpt['minmnumbcompstar'] = 1
        gdat.dictpoplsystinpt['liststrgband'] = gdat.listlablinst[0]
        
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            gdat.dicttroy['true']['CompactObjectStellarCompanion'] = nicomedia.retr_dictpoplstarcomp('CompactObjectStellarCompanion', numbsyst=gdat.numbtarg, **gdat.dictpoplsystinpt)
            gdat.dicttroy['true']['StellarBinary'] = nicomedia.retr_dictpoplstarcomp('StellarBinary', numbsyst=gdat.numbtarg, **gdat.dictpoplsystinpt)
            #gdat.indxcompcosc = gdat.dicttroy['true']['CompactObjectStellarCompanion']['dictindx']['comp']['star']
            #gdat.indxcompsbin = gdat.dicttroy['true']['StellarBinary']['dictindx']['comp']['star']
        elif gdat.typesyst == 'PlanetarySystem':
            gdat.dicttroy['true']['PlanetarySystem'] = nicomedia.retr_dictpoplstarcomp('PlanetarySystem', numbsyst=gdat.numbtarg, **gdat.dictpoplsystinpt)
            gdat.indxcompsyst = gdat.dicttroy['true']['PlanetarySystem']['dictindx']['comp']['star']
        elif gdat.typesyst == 'StarFlaring':
            gdat.dicttroy['true']['StarFlaring'] = nicomedia.retr_dictpoplstarcomp('StarFlaring', numbsyst=gdat.numbtarg, **gdat.dictpoplsystinpt)
        else:
            raise Exception('')

        if gdat.booldiag:
            if gdat.boolsimusome:
                for b in gdat.indxdatatser:
                    for strginst in gdat.listlablinst[b]:
                        for strgpopl in gdat.dicttroy['true'][gdat.typesyst]['dictpopl']['star']:
                            if not 'magtsyst' + strginst in gdat.dicttroy['true'][gdat.typesyst]['dictpopl']['star'][strgpopl]:
                                print('')
                                print('')
                                print('')
                                print('gdat.listlablinst[b]')
                                print(gdat.listlablinst[b])
                                print('gdat.dicttroy[true][gdat.typesyst][dictpopl][star][strgpopl].keys()')
                                print(gdat.dicttroy['true'][gdat.typesyst]['dictpopl']['star'][strgpopl].keys())
                                raise Exception('not magtsyst + strginst in gdat.dicttroy[true][gdat.typesyst][dictpopl][star][strgpopl]')
        
        gdat.namepoplstartotl = 'star_%s_All' % gdat.typepopl
        gdat.namepoplstartran = 'star_%s_Transiting' % gdat.typepopl
        if gdat.typesyst == 'StarFlaring':
            strglimb = 'flar'
        else:
            strglimb = 'comp'
        gdat.namepoplcomptotl = '%sstar_%s_All' % (strglimb, gdat.typepopl)
        gdat.namepoplcomptran = '%sstar_%s_Transiting' % (strglimb, gdat.typepopl)
        
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            for namepoplextn in ['All', 'Transiting']:
                gdat.namepoplcomp = 'compstar_%s_%s' % (gdat.typepopl, namepoplextn)
                
                if gdat.booldiag:
                    if gdat.namepoplcomp.endswith('_tran'):
                        raise Exception('')

                summgene(gdat.dicttroy, boolshowlong=False)
                
                # list of features for stellar systems
                listname = np.intersect1d(np.array(list(gdat.dicttroy['true']['CompactObjectStellarCompanion']['dictpopl']['comp'][gdat.namepoplcomp].keys())), \
                                                          np.array(list(gdat.dicttroy['true']['StellarBinary']['dictpopl']['comp'][gdat.namepoplcomp].keys())))
            
            #if gdat.typedata == 'simutargpartinje':
            #    boolsampstar = False
            #    gdat.dicttroy['true']['StellarSystem']['radistar'] = dicttic8['radistar']
            #    gdat.dicttroy['true']['StellarSystem']['massstar'] = dicttic8['massstar']
            #    indx = np.where((~np.isfinite(gdat.dicttroy['true']['StellarSystem']['massstar'])) | (~np.isfinite(gdat.dicttroy['true']['StellarSystem']['radistar'])))[0]
            #    gdat.dicttroy['true']['StellarSystem']['radistar'][indx] = 1.
            #    gdat.dicttroy['true']['StellarSystem']['massstar'][indx] = 1.
            #    gdat.dicttroy['true']['totl']['magtsystTESS'] = dicttic8['magtsystTESS']
            
            # merge the features of the simulated COSCs and SBs
            gdat.dicttroy['true']['totl'] = dict()
            for namefeat in ['magtsystTESS']:
                if gdat.booldiag:
                    if not gdat.namepoplcomptotl in gdat.dicttroy['true']['CompactObjectStellarCompanion']['dictpopl']['comp']:
                        print('')
                        print('')
                        print('')
                        raise Exception('not gdat.namepoplcomptotl in gdat.dicttroy[true][CompactObjectStellarCompanion][dictpopl][comp]')

                gdat.dicttroy['true']['totl']['magtsystTESS'] = \
                        np.concatenate([gdat.dicttroy['true']['CompactObjectStellarCompanion']['dictpopl']['comp'][gdat.namepoplcomptotl][namefeat], \
                                                        gdat.dicttroy['true']['StellarBinary']['dictpopl']['comp'][gdat.namepoplcomptotl][namefeat]])

        # grab the photometric noise of TESS as a function of TESS magnitude
        #if gdat.typedata == 'simutargsynt':
        #    gdat.stdvphot = nicomedia.retr_noistess(gdat.dicttroy['true']['totl']['magtsystTESS']) * 1e-3 # [dimensionless]
        #    
        #    if not np.isfinite(gdat.stdvphot).all():
        #        raise Exception('')
        
        
        print('Visualizing the features of the simulated population...')

        listboolcompexcl = [False]
        listtitlcomp = ['']
        
        gdat.dictpopltrue = dict()
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            gdat.dictpopltrue['CompactObjectStellarCompanion' + gdat.typepopl + 'totl'] = gdat.dicttroy['true']['CompactObjectStellarCompanion']['dictpopl']['comp'][gdat.namepoplcomptotl]
            gdat.dictpopltrue['StellarBinary' + gdat.typepopl + 'totl'] = gdat.dicttroy['true']['StellarBinary']['dictpopl']['comp'][gdat.namepoplcomptotl]
            gdat.dictpopltrue['CompactObjectStellarCompanion' + gdat.typepopl + 'Transiting'] = gdat.dicttroy['true']['CompactObjectStellarCompanion']['dictpopl']['comp'][gdat.namepoplcomptran]
            gdat.dictpopltrue['StellarBinary' + gdat.typepopl + 'Transiting'] = gdat.dicttroy['true']['StellarBinary']['dictpopl']['comp'][gdat.namepoplcomptran]
        elif gdat.typesyst == 'PlanetarySystem':
            gdat.dictpopltrue['PlanetarySystem_%s_All' % gdat.typepopl] = gdat.dicttroy['true']['PlanetarySystem']['dictpopl']['comp'][gdat.namepoplcomptotl]
            gdat.dictpopltrue['PlanetarySystem_%s_Transiting' % gdat.typepopl] = gdat.dicttroy['true']['PlanetarySystem']['dictpopl']['comp'][gdat.namepoplcomptran]
        elif gdat.typesyst == 'StarFlaring':
            gdat.dictpopltrue['StarFlaring_%s_All' % gdat.typepopl] = gdat.dicttroy['true']['StarFlaring']['dictpopl']['flar'][gdat.namepoplcomptotl]
            gdat.dictpopltrue['StarFlaring_%s_Mdwarfs' % gdat.typepopl] = gdat.dicttroy['true']['StarFlaring']['dictpopl']['flar'][gdat.namepoplcomptotl]
        else:
            raise Exception('')
        
        # check if gdat.dictpopltrue is properly defined, which should be a list of two items (of values and labels, respectively)
        if gdat.booldiag:
            for namepopl in gdat.dictpopltrue:
                for namefeat in gdat.dictpopltrue[namepopl]:
                    if len(gdat.dictpopltrue[namepopl][namefeat]) != 2 or \
                                        len(gdat.dictpopltrue[namepopl][namefeat][1]) > 0 and not isinstance(gdat.dictpopltrue[namepopl][namefeat][1], str):
                        print('')
                        print('')
                        print('')
                        print('gdat.dictpopltrue[namepopl][namefeat]')
                        print(gdat.dictpopltrue[namepopl][namefeat])
                        raise Exception('gdat.dictpopltrue is not properly defined.')
        
        typecnfg = '%s_%s_%s' % (gdat.typesyst, gdat.strginstconc, gdat.typepopl)

        pathvisu = gdat.pathvisucnfg + 'True_Features/'
        pathdata = gdat.pathdatacnfg + 'True_Features/'
        
        lablnumbsamp = 'Number of systems'

        pergamon.init( \
                      typecnfg, \
                      dictpopl=gdat.dictpopltrue, \
                      listdictlablcolrpopl=listdictlablcolrpopl, \
                      lablnumbsamp=lablnumbsamp, \
                      listboolcompexcl=listboolcompexcl, \
                      listtitlcomp=listtitlcomp, \
                      pathvisu=pathvisu, \
                      pathdata=pathdata, \
                      boolsortpoplsize=False, \
                     )
        
        # relevant targets
        gdat.dictindxtarg['rele'] = [[] for v in gdat.indxtyperele]
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            indx = np.where(np.isfinite(gdat.dicttroy['true']['StellarBinary']['dictpopl']['comp'][gdat.namepoplcomptran]['duratrantotl'][gdat.indxssyscosc]))
            gdat.dictindxtarg['cosctran'] = gdat.dictindxtarg['CompactObjectStellarCompanion'][indx]
            # relevants are all COSCs
            gdat.dictindxtarg['rele'][0] = gdat.dictindxtarg['CompactObjectStellarCompanion']
            # relevants are those transiting COSCs
            gdat.dictindxtarg['rele'][1] = gdat.dictindxtarg['cosctran']
        elif gdat.typesyst == 'PlanetarySystem':
            
            # this check is probably wrong
            if False and gdat.booldiag:
                if len(gdat.dictindxtarg['rele']) != 2:
                    print('')
                    print('')
                    print('')
                    print('gdat.indxtyperele')
                    print(gdat.indxtyperele)
                    print('gdat.dictindxtarg[rele]')
                    print(gdat.dictindxtarg['rele'])
                    raise Exception('len(gdat.dictindxtarg[rele]) != 2')
            
            # relevants are Planetary Systems
            gdat.dictindxtarg['rele'][0] = gdat.dictindxtarg['PlanetarySystem']
        
        elif gdat.typesyst == 'StarFlaring':
            #indx = np.where(np.isfinite(gdat.dicttroy['true']['StarFlaring']['dictpopl']['flar'][gdat.namepoplcomptran]['duratrantotl'][gdat.indxssyscosc]))
            gdat.dictindxtarg['rele'][0] = gdat.dictindxtarg['StarFlaring']
        else:
            raise Exception('')
        gdat.numbtargrele = np.empty(gdat.numbtyperele, dtype=int)
        
        gdat.dictindxtarg['irre'] = [[] for v in gdat.indxtyperele]
        for v in gdat.indxtyperele:
            gdat.dictindxtarg['irre'][v] = np.setdiff1d(gdat.indxtarg, gdat.dictindxtarg['rele'][v])
            gdat.numbtargrele[v] = gdat.dictindxtarg['rele'][v].size
        
        gdat.indxssysrele = [[] for v in gdat.indxtyperele]
        for v in gdat.indxtyperele:
            cntrssys = 0
            cntrrele = 0
            gdat.indxssysrele[v] = np.empty(gdat.numbtargrele[v], dtype=int)
            for n in gdat.dictindxtarg['rele'][v]:
                if n in gdat.dictindxtarg['StellarBinary']:
                    gdat.indxssysrele[v][cntrrele] = cntrssys
                    cntrssys += 1
                cntrrele += 1

    #if gdat.boolsimusome:
        # move TESS magnitudes from the dictinary of all systems to the dictionaries of each types of system
        #for namepoplcomm in listnametypetrue:
        #    if namepoplcomm != 'totl':
        #        gdat.dicttroy['true'][namepoplcomm]['magtsystTESS'] = gdat.dicttroy['true']['totl']['magtsystTESS'][gdat.dictindxtarg[namepoplcomm]]
    
    ## fill miletos input dictionary
    ### path to put target data and visuals
    gdat.dictmileinptglob['booldiag'] = gdat.booldiag
    gdat.dictmileinptglob['typeverb'] = gdat.typeverb
    gdat.dictmileinptglob['pathbase'] = gdat.pathpopl
    ### Boolean flag to use PDC data
    gdat.dictmileinptglob['listtimescalbdtr'] = [0.5]
    gdat.dictmileinptglob['typefileplot'] = gdat.typefileplot
    gdat.dictmileinptglob['boolplotpopl'] = False
    gdat.dictmileinptglob['boolwritover'] = gdat.boolwritover
    gdat.dictmileinptglob['liststrgtypedata'] = gdat.liststrgtypedata
    gdat.dictmileinptglob['listlablinst'] = gdat.listlablinst
    dictfitt = dict()
    dictfitt['typemodl'] = gdat.typesyst
    if gdat.typesyst == 'CompactObjectStellarCompanion':
        dictfitt['typemodllens'] = gdat.typemodllens
    else:
        dictfitt['typemodllens'] = None
        
    gdat.dictmileinptglob['dictfitt'] = dictfitt
    gdat.dictmileinptglob['maxmfreqlspe'] = 1. / 0.1 # minimum period is 0.1 day
    #gdat.dictmileinptglob['boolsrchsingpuls'] = True
    #### define SDE threshold for periodic box search
    if not 'dictboxsperiinpt' in gdat.dictmileinptglob:
        gdat.dictmileinptglob['dictboxsperiinpt'] = dict()
    
    # inputs to the periodic box search pipeline
    ## make periodic box search single-process because each targets gets its own process
    gdat.dictmileinptglob['dictboxsperiinpt']['boolprocmult'] = False
    
    if boolprocmult:
        import multiprocessing
        from functools import partial
        multiprocessing.set_start_method('spawn')

        if __name__ == '__main__':
            if platform.system() == "Darwin":
                multiprocessing.set_start_method('spawn')

        numbproc = min(multiprocessing.cpu_count() - 1, gdat.numbtarg)
        
        print('Generating %d processes...' % numbproc)
        
        objtpool = multiprocessing.Pool(numbproc)
        numbproc = objtpool._processes
        indxproc = np.arange(numbproc)

        gdat.listindxtarg = [[] for p in indxproc]
        indxproctarg = np.linspace(0, numbproc - 1, gdat.numbtarg).astype(int)
        for p in indxproc:
            gdat.listindxtarg[p] = np.where(indxproctarg == p)[0]
        listgdat = objtpool.map(partial(mile_work, gdat), indxproc)
    else:
        gdat.listindxtarg = [gdat.indxtarg]
        temp = mile_work(gdat, 0)
    
    if gdat.boolsimusome:
        for u in gdat.indxtypeposi:
            for v in gdat.indxtyperele:
                gdat.boolposirele[u][v] = np.array(gdat.boolposirele[u][v], dtype=bool)
                gdat.boolreleposi[u][v] = np.array(gdat.boolreleposi[u][v], dtype=bool)
    
    gdat.dictindxtarg['posi'] = [[] for u in gdat.indxtypeposi]
    gdat.dictindxtarg['nega'] = [[] for u in gdat.indxtypeposi]
    for u in gdat.indxtypeposi:
        gdat.dictindxtarg['posi'][u] = np.where(gdat.boolpositarg[u])[0]
        gdat.dictindxtarg['nega'][u] = np.setdiff1d(gdat.indxtarg, gdat.dictindxtarg['posi'][u])
    
    # for each positive and relevant type, estimate the recall and precision
    gdat.indxtypeposiiter = np.concatenate((np.array([-1]), gdat.indxtypeposi))
    if gdat.boolsimusome:
        gdat.indxtypereleiter = np.concatenate((np.array([-1]), gdat.indxtyperele))
    else:
        gdat.indxtypereleiter = np.array([-1])
    
    print('gdat.indxtypeposiiter')
    print(gdat.indxtypeposiiter)
    print('gdat.indxtypereleiter')
    print(gdat.indxtypereleiter)

    for u in gdat.indxtypeposiiter:
        for v in gdat.indxtypereleiter:
            
            if u == -1 and v == -1:
                continue
        
            if gdat.booldiag:
                if v >= len(gdat.listlablrele):
                    print('')
                    print('')
                    print('')
                    print('gdat.indxtypereleiter')
                    print(gdat.indxtypereleiter)
                    print('v')
                    print(v)
                    print('gdat.listlablrele')
                    print(gdat.listlablrele)
                    raise Exception('v >= len(gdat.listlablrele)')

            # for relevant type v
            if u == -1:
                strguuvv = 'v%d' % (v)
                labluuvv = '(v = %d)' % (v)
            # for positive type u
            elif v == -1:
                strguuvv = 'u%d' % (u)
                labluuvv = '(u = %d)' % (u)
            # for positive type u and relevant type v
            else:
                strguuvv = 'u%dv%d' % (u, v)
                labluuvv = '(u = %d, v = %d)' % (u, v)

            gdat.dictindxtargtemp = dict()
            gdat.dicttarg = dict()

            if u == -1:
                gdat.dictindxtargtemp[strguuvv + 're'] = gdat.dictindxtarg['rele'][v]
                gdat.dictindxtargtemp[strguuvv + 'ir'] = gdat.dictindxtarg['irre'][v]
            elif v == -1:
                gdat.dictindxtargtemp[strguuvv + 'ne'] = gdat.dictindxtarg['nega'][u]
                gdat.dictindxtargtemp[strguuvv + 'po'] = gdat.dictindxtarg['posi'][u]
            else:
                gdat.dictindxtargtemp[strguuvv + 'trpo'] = np.intersect1d(gdat.dictindxtarg['posi'][u], gdat.dictindxtarg['rele'][v])
                gdat.dictindxtargtemp[strguuvv + 'trne'] = np.intersect1d(gdat.dictindxtarg['nega'][u], gdat.dictindxtarg['irre'][v])
                gdat.dictindxtargtemp[strguuvv + 'flpo'] = np.intersect1d(gdat.dictindxtarg['posi'][u], gdat.dictindxtarg['irre'][v])
                gdat.dictindxtargtemp[strguuvv + 'flne'] = np.intersect1d(gdat.dictindxtarg['nega'][u], gdat.dictindxtarg['rele'][v])
            
            for strgkeyy in gdat.dictindxtargtemp:
                if len(gdat.dictindxtargtemp[strgkeyy]) > 0:
                    gdat.dicttarg[strgkeyy] = dict()
                    for namefeat in gdat.listnamefeatstat:
                        tdpy.setp_dict(gdat.dicttarg[strgkeyy], namefeat, gdat.dictstat[namefeat][0][gdat.dictindxtargtemp[strgkeyy]])
                    for namefeat in gdat.dictpopltrue.keys():
                        tdpy.setp_dict(gdat.dicttarg[strgkeyy], namefeat, gdat.dictpopltrue[namefeat][0][gdat.dictindxtargtemp[strgkeyy]])
            
            listdictlablcolrpopl = []
            listboolcompexcl = []
            listtitlcomp = []
            listnamepoplcomm = list(gdat.dicttarg.keys())
            strgtemp = 'stat' + strguuvv
            
            print('u, v')
            print(u, v)
            print('strguuvv')
            print(strguuvv)
            print('listnamepoplcomm')
            print(listnamepoplcomm)
            print('gdat.listlablrele')
            print(gdat.listlablrele)
            print('gdat.listlablposi')
            print(gdat.listlablposi)
            print('listdictlablcolrpopl')
            print(listdictlablcolrpopl)
            print('gdat.indxtyperele')
            print(gdat.indxtyperele)
            
            boolgood = False
            for namepoplcomm in listnamepoplcomm:
                if strguuvv + 're' in namepoplcomm:
                    boolgood = True
                if strguuvv + 'ir' in namepoplcomm:
                    boolgood = True
            if boolgood:
                listdictlablcolrpopl.append(dict())
                listboolcompexcl.append(True)
                listtitlcomp.append(None)
            for namepoplcomm in listnamepoplcomm:
                if strguuvv + 're' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablirre[v], 'blue']
                if strguuvv + 'ir' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablirre[v], 'orange']
            
            boolgood = False
            for namepoplcomm in listnamepoplcomm:
                if strguuvv + 'po' in namepoplcomm:
                    boolgood = True
                if strguuvv + 'ne' in namepoplcomm:
                    boolgood = True
            if boolgood:
                listdictlablcolrpopl.append(dict())
                listboolcompexcl.append(True)
                listtitlcomp.append(None)
            for namepoplcomm in listnamepoplcomm:
                if strguuvv + 'po' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablposi[u], 'violet']
                if strguuvv + 'ne' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablnega[u], 'brown']
            
            boolgood = False
            for namepoplcomm in listnamepoplcomm:
                if strguuvv + 'trpo' in namepoplcomm:
                    boolgood = True
                if strguuvv + 'trne' in namepoplcomm:
                    boolgood = True
                if strguuvv + 'flpo' in namepoplcomm:
                    boolgood = True
                if strguuvv + 'flne' in namepoplcomm:
                    boolgood = True
            if boolgood:
                listdictlablcolrpopl.append(dict())
                listboolcompexcl.append(True)
                listtitlcomp.append(None)
            for namepoplcomm in listnamepoplcomm:
                if strguuvv + 'trpo' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablrele[v] + ', ' + gdat.listlablposi[u], 'green']
                if strguuvv + 'trne' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablirre[v] + ', ' + gdat.listlablnega[u], 'blue']
                if strguuvv + 'flpo' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablirre[v] + ', ' + gdat.listlablposi[u], 'red']
                if strguuvv + 'flne' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablrele[v] + ', ' + gdat.listlablnega[u], 'orange']
            
            typecnfg = '%s' % (gdat.strgextn)
            
            print('listdictlablcolrpopl')
            print(listdictlablcolrpopl)
            print('listboolcompexcl')
            print(listboolcompexcl)
            print('listtitlcomp')
            print(listtitlcomp)

            for dictlablcolrpopl in listdictlablcolrpopl:
                if len(dictlablcolrpopl) == 0:
                    raise Exception('')

            pathvisu = gdat.pathvisucnfg + 'Features/'
            pathdata = gdat.pathdatacnfg + 'Features/'
            pergamon.init( \
                          dictpopl=gdat.dicttarg, \
                          listdictlablcolrpopl=listdictlablcolrpopl, \
                          listboolcompexcl=listboolcompexcl, \
                          listtitlcomp=listtitlcomp, \
                          pathvisu=pathvisu, \
                          pathdata=pathdata, \
                          lablsampgene='exoplanet', \
                          boolsortpoplsize=False, \
                         )
            
            print('Will plot precision and recall...')
            if gdat.boolplot and gdat.boolsimusome and u != -1 and v != -1:
                listvarbreca = []
                
                listvarbreca.append(gdat.dicttroy['true'][gdat.typesyst]['dictpopl']['comp'][gdat.namepoplcomptotl]['pericomp'][0][gdat.indxssysrele[v]])
                #listvarbreca.append(gdat.dicttroy['true'][gdat.typesyst]['dictpopl']['comp'][gdat.namepoplcomptotl]['masscomp'][gdat.indxssysrele[v]])
                #listvarbreca.append(gdat.dicttroy['true'][gdat.typesyst]['dictpopl']['comp'][gdat.namepoplcomptotl]['magtsystTESS'][0][gdat.dictindxtarg['rele'][v]])
                listvarbreca = np.vstack(listvarbreca).T
                
                liststrgvarbreca = []
                liststrgvarbreca.append('trueperi')
                #liststrgvarbreca.append('truemasscomp')
                #liststrgvarbreca.append('truemagtsystTESS')
                
                listlablvarbreca, listscalvarbreca, _, _, _ = tdpy.retr_listlablscalpara(liststrgvarbreca)
                
                listtemp = []
                for namefeat in gdat.listnamefeatstat:
                    listtemp.append(gdat.dictstat[namefeat][0][gdat.dictindxtarg['posi'][u]])
                listvarbprec = np.vstack(listtemp).T
                #listvarbprec = np.vstack([gdat.lists2nr, gdat.listpowrlspe]).T
                liststrgvarbprec = gdat.listnamefeatstat#['s2nr', 'powrlspe']
                listlablvarbprec, listscalvarbprec, _, _, _ = tdpy.retr_listlablscalpara(liststrgvarbprec)
                #print('listvarbreca')
                #print(listvarbreca)
                #print('listvarbprec')
                #print(listvarbprec)
                #print('gdat.boolreletarg[v]')
                #print(gdat.boolreletarg[v])
                #print('gdat.boolpositarg[u]')
                #print(gdat.boolpositarg[u])
                #print('gdat.boolposirele[u][v]')
                #print(gdat.boolposirele[u][v])
                #print('gdat.boolreleposi[u][v]')
                #print(gdat.boolreleposi[u][v])

                strgextn = '%s_%s' % (gdat.typepopl, strguuvv)
                tdpy.plot_recaprec(gdat.pathvisucnfg, strgextn, listvarbreca, listvarbprec, liststrgvarbreca, liststrgvarbprec, \
                                        listlablvarbreca, listlablvarbprec, gdat.boolposirele[u][v], gdat.boolreleposi[u][v])


        #if gdat.typecnfg == 'CompactObjectStellarCompanion' or gdat.typecnfg == 'psys' or gdat.typecnfg == 'plan':
        #    
        #        # calculate photometric precision for the star population
        #        if typeinst.startswith('tess'):
        #            gdat.dictpopl[namepoplcomptran]['nois'] = nicomedia.retr_noistess(gdat.dictpopl[namepoplcomptran]['magtsystTESS'])
        #        elif typeinst.startswith('lsst'):
        #            gdat.dictpopl[namepoplcomptran]['nois'] = nicomedia.retr_noislsst(gdat.dictpopl[namepoplcomptran]['rmag'])
        #    
        #        # expected BLS signal detection efficiency
        #        if typeinst.startswith('lsst'):
        #            numbvisi = 1000
        #            gdat.dictpopl[namepoplcomptran]['sdee'] = gdat.dictpopl[namepoplcomptran]['depttrancomp'] / 5. / gdat.dictpopl[namepoplcomptran]['nois'] * \
        #                                                                                                 np.sqrt(gdat.dictpopl[namepoplcomptran]['dcyc'] * numbvisi)
        #        if typeinst.startswith('tess'):
        #            if gdat.typecnfg == 'plan':
        #                gdat.dictpopl[namepoplcomptran]['sdee'] = np.sqrt(gdat.dictpopl[namepoplcomptran]['duratrantotl']) * \
        #                                                                    gdat.dictpopl[namepoplcomptran]['depttrancomp'] / gdat.dictpopl[namepoplcomptran]['nois']
        #            if gdat.typecnfg == 'CompactObjectStellarCompanion':
        #                gdat.dictpopl[namepoplcomptran]['sdee'] = np.sqrt(gdat.dictpopl[namepoplcomptran]['duratrantotl']) * \
        #                                                                                        gdat.dictpopl[namepoplcomptran]['amplslen'] \
        #                                                                                                               / gdat.dictpopl[namepoplcomptran]['nois']
        #        
        #        # expected detections
        #        #gdat.dictpopl[namepoplcomptran]['probdeteusam'] = np.exp(-(0.01 * gdat.dictpopl[namepoplcomptran]['pericomp'] * gdat.dictpopl[namepoplcomptran]['numbtsec']))
        #        #booldeteusam = np.random.rand(gdat.dictpopl[namepoplcomptran]['pericomp'].size) < gdat.dictpopl[namepoplcomptran]['probdeteusam']
        #        
        #        #indx = (gdat.dictpopl[namepoplcomptran]['sdee'] > 5) & booldeteusam
        #        indx = (gdat.dictpopl[namepoplcomptran]['sdee'] > 5)
        #        retr_subp(gdat.dictpopl, gdat.dictnumbsamp, gdat.dictindxsamp, namepoplcomptran, 'compstar' + typepoplsyst + 'tranposi', indx)

        #        # expected non-detections
        #        #indx = (gdat.dictpopl[namepoplcomptran]['sdee'] < 5) | (~booldeteusam)
        #        indx = (gdat.dictpopl[namepoplcomptran]['sdee'] < 5)
        #        retr_subp(gdat.dictpopl, gdat.dictnumbsamp, gdat.dictindxsamp, namepoplcomptran, 'compstar' + typepoplsyst + 'trannega', indx)




