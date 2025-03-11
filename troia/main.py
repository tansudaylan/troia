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
            for v in gdat.indxtypeclastrue:
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

        if n < gdat.numbtargplot:
            # determine whether to make miletos plots of the analysis
            gdat.dictmileinpttarg['boolplot'] = gdat.boolplotmile
            
            # determine whether to make ephesos plots of the simulated data
            if gdat.boolsimusome:
                gdat.dictmileinpttarg['boolmakeplotefestrue'] = True
        else:
            gdat.dictmileinpttarg['boolplot'] = False
        
        # determine whether to make animations of simulated data 
        if gdat.boolsimusome:
            if n < gdat.numbtarganimtrue:
                gdat.dictmileinpttarg['boolmakeanimefestrue'] = True

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
                
                print('gdat.listindxtarg[i]')
                summgene(gdat.listindxtarg[i])
                print('n')
                print(n)
                print('gdat.indxcompsyst')
                summgene(gdat.indxcompsyst)

                dicttrue[namepara] = gdat.dicttroy['true']['PlanetarySystem']['dictpopl']['comp'][gdat.namepoplcomptotl][namepara][0][gdat.indxcompsyst[n]]
            
            gdat.dictmileinpttarg['dicttrue'] = dicttrue
        
            for b in gdat.indxdatatser:
                for strginst in gdat.listlablinst[b]:
                    dictmagtsyst[strginst] = dicttrue['magtsyst' + strginst]
            gdat.dictmileinpttarg['dictmagtsyst'] = dictmagtsyst
        
        # call miletos to analyze data
        print('Calling miletos...')
        print('temp')
        #dictmileoutp = miletos.init( \
        #                            **gdat.dictmileinpttarg, \
        #                           )
        dictmileoutp['boolcalclspe'] = False
        dictmileoutp['boolsrchboxsperi'] = False
        dictmileoutp['boolsrchoutlperi'] = True

        if n == 0:
            gdat.listlablclasdisp = []
            if dictmileoutp['boolcalclspe']:
                gdat.listlablclasdisp.append('High LS power')
                gdat.listlablclasdisp.append('Low LS power')
            if dictmileoutp['boolsrchboxsperi']:
                gdat.listlablclasdisp.append('High BLS power')
                gdat.listlablclasdisp.append('Low BLS power')
            if dictmileoutp['boolsrchoutlperi']:
                gdat.listlablclasdisp.append('Low min$_k$ $f_k$')
                gdat.listlablclasdisp.append('High min$_k$ $f_k$')
            
            gdat.numbtypeclasdisp = len(gdat.listlablclasdisp)
            gdat.indxtypeclasdisp = np.arange(gdat.numbtypeclasdisp)
            gdat.boolpositarg = [np.empty(gdat.numbtarg, dtype=bool) for u in gdat.indxtypeclasdisp]
            
            gdat.listnameclasdispposi = ''
            
            gdat.listnameclasdisp = [[] for u in gdat.indxtypeclasdisp]
            for u in gdat.indxtypeclasdisp:
                gdat.listnameclasdisp[u] = ''.join(gdat.listlablclasdisp[u].split(' '))
            
            if gdat.boolsimusome:
                gdat.boolreleposi = [[[] for v in gdat.indxtypeclastrue] for u in gdat.indxtypeclasdisp]
                gdat.boolposirele = [[[] for v in gdat.indxtypeclastrue] for u in gdat.indxtypeclasdisp]
        
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
            
            for u in gdat.indxtypeclasdisp:
                gdat.dictstat[gdat.listnameclasdisp[u]] = dict()
                for namefeat in gdat.listnamefeatstat:
                    gdat.dictstat[gdat.listnameclasdisp[u]][namefeat] = [np.empty(gdat.numbtarg), '']
        
        print('gdat.indxtypeclasdisp')
        print(gdat.indxtypeclasdisp)
        print('gdat.listnameclasdisp')
        print(gdat.listnameclasdisp)
        print('gdat.dictstat')
        print(gdat.dictstat)
        for u in gdat.indxtypeclasdisp:
            if dictmileoutp['boolcalclspe']:
                gdat.dictstat[gdat.listnameclasdisp[u]]['perilspeprim'][0][n] = dictmileoutp['perilspempow']
                gdat.dictstat[gdat.listnameclasdisp[u]]['powrlspeprim'][0][n] = dictmileoutp['powrlspempow']
            if dictmileoutp['boolsrchboxsperi']:
                gdat.dictstat[gdat.listnameclasdisp[u]]['s2nrpboxprim'][0][n] = dictmileoutp['dictboxsperioutp']['s2nr'][0]
                gdat.dictstat[gdat.listnameclasdisp[u]]['peripboxprim'][0][n] = dictmileoutp['dictboxsperioutp']['peri'][0]
            if dictmileoutp['boolsrchoutlperi']:
                gdat.dictstat[gdat.listnameclasdisp[u]]['minmfrddtimeoutlsort'][0][n] = dictmileoutp['dictoutlperi']['minmfrddtimeoutlsort'][0]
        
        # taking the fist element, which belongs to the first TCE
        for u in gdat.indxtypeclasdisp:
            gdat.boolpositarg[u][n] = dictmileoutp['boolposianls'][u]
        
        if gdat.boolsimusome:
            for u in gdat.indxtypeclasdisp:
                for v in gdat.indxtypeclastrue:
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

    # number of targets for which to produce miletos plots
    gdat.numbtargplot = 5

    # number of targets for which to animate true system via ephesos
    gdat.numbtarganimtrue = 0

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
        gdat.indxtypeclastrue = np.arange(gdat.numbtyperele)
        gdat.boolreletarg = [np.empty(gdat.numbtarg, dtype=bool) for v in gdat.indxtypeclastrue]
    
    gdat.dictindxtarg = dict()
    gdat.dicttroy = dict()
    
    if gdat.boolsimusome:
        
        gdat.dictprobclastruetype = dict()
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            gdat.dictprobclastruetype['CompactObjectStellarCompanion'] = [0.70, 'Compact object with Stellar Companion']
            gdat.dictprobclastruetype['StellarBinary'] = [0.25, 'Stellar Binary']
            gdat.dictprobclastruetype['Asteroid'] = [0.05, 'Asteroid']
        elif gdat.typesyst == 'PlanetarySystem':
            gdat.dictprobclastruetype['PlanetarySystem'] = [0.75, 'Planetary System']
            gdat.dictprobclastruetype['StellarBinary'] = [0.25, 'Stellar Binary']
        elif gdat.typesyst == 'StarFlaring':
            gdat.dictprobclastruetype['StellarFlare'] = [1., 'Stellar binary']
        else:
            print('')
            print('')
            print('')
            print('gdat.typesyst')
            print(gdat.typesyst)
            raise Exception('')
        
        gdat.namepoplstartotl = 'star_%s_All' % gdat.typepopl
        gdat.namepoplstartran = 'star_%s_Transiting' % gdat.typepopl
        if gdat.typesyst == 'StarFlaring':
            strglimb = 'flar'
        else:
            strglimb = 'comp'
        
        gdat.namepoplcomptotl = '%sstar_%s_All' % (strglimb, gdat.typepopl)
        gdat.namepoplcomptran = '%sstar_%s_Transiting' % (strglimb, gdat.typepopl)
        
        # names of classes of systems to be simulated
        gdat.listnameclastruetype = list(gdat.dictprobclastruetype.keys())
        
        gdat.numbclastruetype = len(gdat.listnameclastruetype)
        gdat.indxclastruetype = np.arange(gdat.numbclastruetype)

        gdat.probclastruetype = np.empty(gdat.numbclastruetype)
        for k, name in enumerate(gdat.listnameclastruetype):
            gdat.probclastruetype[k] = gdat.dictprobclastruetype[name][0]
        
        # names of classes of systems to be simulated that separately counts geometrical (viewing-angle), e.g., transiting vs not
        gdat.listnameclastrue = []
        gdat.listlablclastrue = []
        for name in gdat.listnameclastruetype:
            gdat.listnameclastrue += ['%s' % name]
            gdat.listlablclastrue += [gdat.dictprobclastruetype[name][1]]
            
            gdat.listnameclastrue += ['%s_Transiting' % name]
            gdat.listlablclastrue += [gdat.dictprobclastruetype[name][1] + ', Transiting']
        
        gdat.listcolrclastrue = np.array(['g', 'b', 'r', 'orange', 'olive', 'yellow'])
        
        if len(gdat.listlablclastrue) > len(gdat.listcolrclastrue):
            raise Exception('')

        # number of simulated classes
        gdat.numbclastrue = len(gdat.listnameclastrue)
        gdat.indxclastrue = np.arange(gdat.numbclastrue)
        
        # probabilities of simulated classes
        gdat.probclastrue = np.empty(gdat.numbclastrue)

        gdat.listcolrclastrue = gdat.listcolrclastrue[gdat.indxclastrue]
        
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

        gdat.indxclastruetypetarg = np.random.choice(gdat.indxclastruetype, size=gdat.numbtarg, p=gdat.probclastruetype)
            
        #gdat.booltypetargtrue = dict()
        
        if gdat.dictpoplsystinpt is None:
            gdat.dictpoplsystinpt = dict()
        if not 'booltrancomp' in gdat.dictpoplsystinpt:
            gdat.dictpoplsystinpt['booltrancomp'] = True
        gdat.dictpoplsystinpt['typepoplsyst'] = gdat.typepopl
        gdat.dictpoplsystinpt['minmnumbcompstar'] = 1
        gdat.dictpoplsystinpt['liststrgband'] = gdat.listlablinst[0]
        
        indxoffs = 0
        gdat.dictnumbtarg = dict()
        for nameclastruetype in gdat.listnameclastruetype:
            gdat.dictnumbtarg[nameclastruetype] = int(gdat.numbtarg * gdat.dictprobclastruetype[nameclastruetype][0])
        
            gdat.dicttroy['true'][nameclastruetype] = nicomedia.retr_dictpoplstarcomp(nameclastruetype, \
                                                            numbsyst=gdat.dictnumbtarg[nameclastruetype], **gdat.dictpoplsystinpt)
            
            gdat.indxcompsyst = gdat.dicttroy['true'][nameclastruetype]['dictindx']['comp']['star']

            nameclastruetypetran = nameclastruetype + '_Transiting'
            nameclastruetypentrn = nameclastruetype + '_Nontransiting'
            
            gdat.dictindxtarg[nameclastruetype] = indxoffs + np.arange(gdat.dictnumbtarg[nameclastruetype])
            gdat.dictindxtarg[nameclastruetypetran] = indxoffs + \
                                        np.where(gdat.dicttroy['true'][nameclastruetype]['dictpopl']['star'][gdat.namepoplstartotl]['booltran'][0])[0]
            gdat.dictindxtarg[nameclastruetypentrn] = np.setdiff1d(gdat.dictindxtarg[nameclastruetype], gdat.dictindxtarg[nameclastruetypetran])
            indxoffs += gdat.dictindxtarg[nameclastruetype].size
        
            
            # to be deleted?
            #gdat.booltypetargtrue[nameclastruetype] = gdat.indxtarg == gdat.dictindxtarg[nameclastruetype]
            #gdat.nameclastruetypetarg[nameclastruetype] = gdat.listnameclastruetype[gdat.indxclastruetypetarg]
            
        gdat.numbtargclastrue = dict()
        for name in gdat.listnameclastrue:
            gdat.numbtargclastrue[name] = gdat.dictindxtarg[name].size

        if gdat.booldiag:
            if gdat.boolsimusome:
                for b in gdat.indxdatatser:
                    for strginst in gdat.listlablinst[b]:
                        for strgpopl in gdat.dicttroy['true'][gdat.typesyst]['dictpopl']['star']:
                            if not 'magtsyst' + strginst in gdat.dicttroy['true'][gdat.typesyst]['dictpopl']['star'][strgpopl]:
                                print('')
                                print('')
                                print('')
                                print('strginst')
                                print(strginst)
                                print('gdat.listlablinst[b]')
                                print(gdat.listlablinst[b])
                                print('gdat.dicttroy[true][gdat.typesyst][dictpopl][star][strgpopl].keys()')
                                print(gdat.dicttroy['true'][gdat.typesyst]['dictpopl']['star'][strgpopl].keys())
                                raise Exception('not magtsyst + strginst in gdat.dicttroy[true][gdat.typesyst][dictpopl][star][strgpopl]')
        
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
        gdat.dictindxtarg['rele'] = [[] for v in gdat.indxtypeclastrue]
        gdat.dictindxtarg['irre'] = [[] for v in gdat.indxtypeclastrue]
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
                    print('gdat.indxtypeclastrue')
                    print(gdat.indxtypeclastrue)
                    print('gdat.dictindxtarg[rele]')
                    print(gdat.dictindxtarg['rele'])
                    raise Exception('len(gdat.dictindxtarg[rele]) != 2')
            
            # relevants are Planetary Systems
            gdat.dictindxtarg['rele'][0] = gdat.dictindxtarg['PlanetarySystem_Transiting']
            gdat.dictindxtarg['irre'][0] = gdat.dictindxtarg['PlanetarySystem_Nontransiting']
        
        elif gdat.typesyst == 'StarFlaring':
            #indx = np.where(np.isfinite(gdat.dicttroy['true']['StarFlaring']['dictpopl']['flar'][gdat.namepoplcomptran]['duratrantotl'][gdat.indxssyscosc]))
            gdat.dictindxtarg['rele'][0] = gdat.dictindxtarg['StarFlaring']
        else:
            raise Exception('')
        gdat.numbtargrele = np.empty(gdat.numbtyperele, dtype=int)
        
        gdat.dictindxtarg['irre'] = [[] for v in gdat.indxtypeclastrue]
        for v in gdat.indxtypeclastrue:
            gdat.dictindxtarg['irre'][v] = np.setdiff1d(gdat.indxtarg, gdat.dictindxtarg['rele'][v])
            
            print('gdat.dictindxtarg[irre][v]')
            print(gdat.dictindxtarg['irre'][v])

            # in case it's empty
            gdat.dictindxtarg['irre'][v] = np.array(gdat.dictindxtarg['irre'][v])

            print('gdat.dictindxtarg[irre][v]')
            print(gdat.dictindxtarg['irre'][v])

            gdat.numbtargrele[v] = gdat.dictindxtarg['rele'][v].size
        
        gdat.indxssysrele = [[] for v in gdat.indxtypeclastrue]
        for v in gdat.indxtypeclastrue:
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
        #for namepoplcomm in listnameclastrue:
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
        for u in gdat.indxtypeclasdisp:
            for v in gdat.indxtypeclastrue:
                gdat.boolposirele[u][v] = np.array(gdat.boolposirele[u][v], dtype=bool)
                gdat.boolreleposi[u][v] = np.array(gdat.boolreleposi[u][v], dtype=bool)
    
    gdat.dictindxtarg['posi'] = [[] for u in gdat.indxtypeclasdisp]
    gdat.dictindxtarg['nega'] = [[] for u in gdat.indxtypeclasdisp]
    for u in gdat.indxtypeclasdisp:
        gdat.dictindxtarg['posi'][u] = np.where(gdat.boolpositarg[u])[0]
        gdat.dictindxtarg['nega'][u] = np.setdiff1d(gdat.indxtarg, gdat.dictindxtarg['posi'][u])
    
    # for each positive and relevant type, estimate the recall and precision

    for u in gdat.indxtypeclasdisp:
        for v in gdat.indxtypeclastrue:
            
            if gdat.booldiag:
                if v >= len(gdat.listlablrele):
                    print('')
                    print('')
                    print('')
                    print('gdat.indxtypeclastrueiter')
                    print(gdat.indxtypeclastrueiter)
                    print('v')
                    print(v)
                    print('gdat.listlablrele')
                    print(gdat.listlablrele)
                    raise Exception('v >= len(gdat.listlablrele)')

            # for positive type u and relevant type v
            strguuvv = 'u%dv%d' % (u, v)
            labluuvv = '(u = %d, v = %d)' % (u, v)

            gdat.dictindxtargtemp = dict()
            gdat.dicttarg = dict()

            gdat.dictindxtargtemp[strguuvv + 're'] = gdat.dictindxtarg['rele'][v]
            gdat.dictindxtargtemp[strguuvv + 'ir'] = gdat.dictindxtarg['irre'][v]
            
            gdat.dictindxtargtemp[strguuvv + 'ne'] = gdat.dictindxtarg['nega'][u]
            gdat.dictindxtargtemp[strguuvv + 'po'] = gdat.dictindxtarg['posi'][u]
            
            gdat.dictindxtargtemp[strguuvv + 'trpo'] = np.intersect1d(gdat.dictindxtarg['posi'][u], gdat.dictindxtarg['rele'][v])
            gdat.dictindxtargtemp[strguuvv + 'trne'] = np.intersect1d(gdat.dictindxtarg['nega'][u], gdat.dictindxtarg['irre'][v])
            gdat.dictindxtargtemp[strguuvv + 'flpo'] = np.intersect1d(gdat.dictindxtarg['posi'][u], gdat.dictindxtarg['irre'][v])
            gdat.dictindxtargtemp[strguuvv + 'flne'] = np.intersect1d(gdat.dictindxtarg['nega'][u], gdat.dictindxtarg['rele'][v])
            
            # determine positive population and negative populations for classification of targets based on disposition properties
            if u == 0:
                if gdat.typesyst == 'PlanetarySystem':
                    namepoplclasdispposi = 'HighBLSpower'
                    listnamepoplclasdispnega = ['LowBLSpower']

            # determine relevant population and irrelevant populations for classification of targets based on true properties
            if v == 0:
                if gdat.typesyst == 'PlanetarySystem':
                    dict_keys(['PlanetarySystem_SyntheticPopulation_All', 'PlanetarySystem_SyntheticPopulation_Transiting'])
                    namepoplclastruerele = 'PlanetarySystem_SyntheticPopulation_Transiting'
                    listnamepoplclastrueirre = ['PlanetarySystem_SyntheticPopulation_Nontransiting']
            
            for strgkeyy in gdat.dictindxtargtemp:
                if len(gdat.dictindxtargtemp[strgkeyy]) > 0:
                    
                    gdat.dicttarg[strgkeyy] = dict()
                    
                    # disposition features
                    ## of the positive population
                    for namefeat in gdat.listnamefeatstat:
                        tdpy.setp_dict(gdat.dicttarg[strgkeyy], namefeat, gdat.dictstat[namepoplclasdispposi][namefeat][0][gdat.dictindxtargtemp[strgkeyy]])
                    ## of the negatives population
                    for namepopl in listnamepoplclasdispnega:
                        for namefeat in gdat.listnamefeatstat:
                            if gdat.booldiag:
                                if not namepopl in gdat.dictstat:
                                    print('')
                                    print('')
                                    print('')
                                    print('gdat.dictstat.keys()')
                                    print(gdat.dictstat.keys())
                                    print('namepopl')
                                    print(namepopl)
                                    raise Exception('not namepopl in gdat.dictstat')

                            tdpy.setp_dict(gdat.dicttarg[strgkeyy], namefeat, gdat.dictstat[namepopl][namefeat][0][gdat.dictindxtargtemp[strgkeyy]])

                    # true features
                    ## of the relevant population
                    print('gdat.dictpopltrue')
                    print(gdat.dictpopltrue.keys())
                    for namefeat in gdat.dictpopltrue[namepoplclastruerele].keys():
                        tdpy.setp_dict(gdat.dicttarg[strgkeyy], namefeat, gdat.dictpopltrue[namepoplclastruerele][namefeat][0][gdat.dictindxtargtemp[strgkeyy]])
                    ## of the irrelevant populations
                    for namepopl in listnamepoplclastrueirre:
                        for namefeat in gdat.dictpopltrue[namepopl].keys():
                            tdpy.setp_dict(gdat.dicttarg[strgkeyy], namefeat, gdat.dictpopltrue[namepopl][namefeat][0][gdat.dictindxtargtemp[strgkeyy]])
            
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
            print('gdat.listlablclasdisp')
            print(gdat.listlablclasdisp)
            print('listdictlablcolrpopl')
            print(listdictlablcolrpopl)
            print('gdat.indxtypeclastrue')
            print(gdat.indxtypeclastrue)
            
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
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablclasdisp[u], 'violet']
                if strguuvv + 'ne' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablclasdisp[u], 'brown']
            
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
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablrele[v] + ', ' + gdat.listlablclasdisp[u], 'green']
                if strguuvv + 'trne' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablirre[v] + ', ' + gdat.listlablclasdisp[u], 'blue']
                if strguuvv + 'flpo' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablirre[v] + ', ' + gdat.listlablclasdisp[u], 'red']
                if strguuvv + 'flne' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablrele[v] + ', ' + gdat.listlablclasdisp[u], 'orange']
            
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
                    listtemp.append(gdat.dictstat[gdat.listnameclasdisp[u]][namefeat][0][gdat.dictindxtarg['posi'][u]])
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




