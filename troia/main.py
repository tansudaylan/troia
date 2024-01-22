import os, sys, datetime, fnmatch, copy

from tqdm import tqdm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import astroquery

import numpy as np
import scipy.interpolate

import json

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

        if gdat.booldataobsv and gdat.typepopl == 'list':
            #listarrytser = None
            rasctarg = None
            decltarg = None
            strgmast = liststrgmast[n]
            labltarg = None
            strgtarg = None

        else:
            #listarrytser = dict()
            #listarrytser['raww'] = gdat.listarrytser['data'][n]
            
            rasctarg = None
            decltarg = None
            strgmast = None
            labltarg = gdat.labltarg[n]
            strgtarg = gdat.strgtarg[n]
            if len(strgtarg) == 0:
                raise Exception('')
        
        gdat.dictmileinpttarg = copy.deepcopy(gdat.dictmileinptglob)

        if n < gdat.maxmnumbtargplot:
            gdat.dictmileinpttarg['boolplot'] = gdat.boolplotmile
        else:
            gdat.dictmileinpttarg['boolplot'] = False
        gdat.dictmileinpttarg['rasctarg'] = rasctarg
        gdat.dictmileinpttarg['decltarg'] = decltarg
        gdat.dictmileinpttarg['strgtarg'] = strgtarg
        gdat.dictmileinpttarg['labltarg'] = labltarg
        gdat.dictmileinpttarg['strgmast'] = strgmast
        #gdat.dictmileinpttarg['listarrytser'] = listarrytser
        
        dicttrue = dict()
        dicttrue['numbyearlsst'] = 5
        dicttrue['typemodl'] = 'PlanetarySystem'
        typelevl = 'limb'
        if typelevl == 'body':
            for namepara in gdat.dicttroy['true']['PlanetarySystem']['listnamefeatbody']:
                dicttrue[namepara] = gdat.dicttroy['true']['PlanetarySystem']['dictpopl']['star'][gdat.namepoplstartotl][namepara][n]
        if typelevl == 'limb':
            for namepara in gdat.dicttroy['true']['PlanetarySystem']['listnamefeatlimbonly']:
                dicttrue[namepara] = gdat.dicttroy['true']['PlanetarySystem']['dictpopl']['comp'][gdat.namepoplcomptotl][namepara][gdat.indxcompsyst[n]]
        gdat.dictmileinpttarg['dicttrue'] = dicttrue
        
        gdat.boolskipmile = True

        if gdat.boolskipmile:
            dictmileoutp = dict()
            dictmileoutp['perilspempow'] = 0
            dictmileoutp['powrlspempow'] = 0
            dictmileoutp['dictpboxoutp'] = dict()
            dictmileoutp['dictpboxoutp']['sdeecomp'] = [0]
            dictmileoutp['dictpboxoutp']['pericomp'] = [0]
            
            dictmileoutp['boolposianls'] = []
            for u in gdat.indxtypeposi:
                dictmileoutp['boolposianls'].append(True)
        else:
            # call miletos to analyze data
            dictmileoutp = miletos.init( \
                                        **gdat.dictmileinpttarg, \
                                       )
        
        gdat.dictstat['perilspeprim'][n] = dictmileoutp['perilspempow']
        gdat.dictstat['powrlspeprim'][n] = dictmileoutp['powrlspempow']
        
        gdat.dictstat['sdeepboxprim'][n] = dictmileoutp['dictpboxoutp']['sdeecomp'][0]
        gdat.dictstat['peripboxprim'][n] = dictmileoutp['dictpboxoutp']['pericomp'][0]
        
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

        # type of population of target sources
        typepopl=None, \

        # list of target TIC IDs
        listticitarg=None, \
        
        # type of experiment
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
   
    if gdat.boolplotinit is None:
        gdat.boolplotinit = gdat.boolplot
    
    if gdat.boolplotmile is None:
        gdat.boolplotmile = gdat.boolplot
    
    # check inputs
    if not gdat.boolplot and (gdat.boolplotinit or gdat.boolplotmile):
        raise Exception('')

    print('troia initialized at %s...' % gdat.strgtimestmp)
    
    if gdat.liststrgmast is not None and gdat.listticitarg is not None:
        raise Exception('liststrgmast and listticitarg cannot be defined simultaneously.')

    gdat.booltargusertici = gdat.listticitarg is not None
    gdat.booltargusermast = gdat.liststrgmast is not None
    gdat.booltargusergaid = gdat.listgaid is not None
    gdat.booltarguser = gdat.booltargusertici or gdat.booltargusermast or gdat.booltargusergaid
    
    miletos.setup_miletos(gdat)

    miletos.setup1_miletos(gdat)
    
    if gdat.booltargsynt and gdat.booltarguser or not gdat.booltargsynt and not gdat.booltarguser and gdat.typepopl is None:
        print('gdat.booltarguser')
        print(gdat.booltarguser)
        raise Exception('')

    if (liststrgmast is not None or listticitarg is not None) and gdat.typepopl is None:
        raise Exception('The type of population, typepopl, must be defined by the user when the target list is provided by the user')
    
    if gdat.typepopl is None:
        gdat.typepopl = 'Synthetic'
    
    #        gdat.typepopl = 'CTL_prms_2min'
    #        gdat.typepopl = 'CTL_prms_2min'
    
    print('gdat.typepopl')
    print(gdat.typepopl)

    if gdat.booldiag:
        if gdat.booltargsynt and gdat.typepopl != 'Synthetic':
            print('')
            print('')
            print('')
            print('gdat.booltargsynt')
            print(gdat.booltargsynt)
            print('gdat.typepopl')
            print(gdat.typepopl)
            raise Exception('')
    
    # paths
    ## read environment variable
    gdat.pathbase = os.environ['TROIA_DATA_PATH'] + '/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathdatatess = os.environ['TESS_DATA_PATH'] + '/'
    gdat.pathdatalcur = gdat.pathdata + 'lcur/'
    gdat.pathvisu = gdat.pathbase + 'visuals/'
    gdat.pathtsec = gdat.pathdata + 'logg/tsec/'
    
    gdat.listlablinst = miletos.retr_strginst(gdat, gdat.listlablinst)

    gdat.strginstconc = ''
    k = 0
    for b in range(2):
        for p in range(len(gdat.listlablinst[b])):
            if k > 0:
                gdat.strginstconc += '_'
            gdat.strginstconc += '%s' % gdat.listlablinst[b][p]
            k += 1
    
    gdat.strgtypedataconc = ''
    if gdat.booltargsynt:
        gdat.strgtypedataconc = 'simutargsynt'
    gdat.strgextn = '%s_%s_%s' % (gdat.typepopl, gdat.strgtypedataconc, gdat.strginstconc)
    
    gdat.pathpopl = gdat.pathbase + gdat.strgextn + '/'
    gdat.pathvisupopl = gdat.pathpopl + 'visuals/'
    gdat.pathdatapopl = gdat.pathpopl + 'data/'

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
        if gdat.booltargusermast:
            gdat.numbtarg = len(gdat.liststrgmast)
        if gdat.booltargusergaid:
            gdat.numbtarg = len(gdat.listgaidtarg)
    else:
        if gdat.boolsimusome:
            gdat.numbtarg = 30000
        else:
            gdat.numbtarg = dicttic8['TICID'].size
            
        gdat.numbtarg = 30
    
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
    # make initial plots
    if gdat.boolplot and gdat.boolplotinit:
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            path = gdat.pathvisu + 'radieinsmass.%s' % (gdat.typefileplot) 
            if not os.path.exists(path):
                # plot Einstein radius vs lens mass
                figr, axis = plt.subplots(figsize=(6, 4))
                listsmax = [0.1, 1., 10.] # [AU]
                dictfact = tdpy.retr_factconv()
                peri = 10.#np.logspace(-1., 2., 100)
                masslens = np.logspace(np.log10(0.1), np.log10(100.), 100)
                radilenswdrf = 0.007 * masslens**(-1. / 3.)
                #smax = ephesos.retr_smaxkepl(peri, masstotl) # AU
                for smax in listsmax:
                    radieins = chalcedon.retr_radieinssbin(masslens, smax)
                    axis.plot(masslens, radieins)
                    axis.plot(masslens, radilenswdrf)
                axis.set_xlabel('$M$ [$M_\odot$]')
                axis.set_ylabel('$R$ [$R_\odot$]')
                axis.set_xscale('log')
                axis.set_yscale('log')
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
            path = gdat.pathvisu + 'radieinssmax.%s' % (gdat.typefileplot) 
            if not os.path.exists(path):
                # plot Einstein radius vs lens mass
                figr, axis = plt.subplots(figsize=(6, 4))
                listmasslens = [0.1, 1.0, 10., 100.] # [AU]
                dictfact = tdpy.retr_factconv()
                peri = 10.#np.logspace(-1., 2., 100)
                smax = np.logspace(np.log10(0.01), np.log10(10.), 100)
                listcolr = ['b', 'g', 'r', 'orange']
                #radilenswdrf = 0.007 * masslens**(-1. / 3.)
                #smax = ephesos.retr_smaxkepl(peri, masstotl) # AU
                for k, masslens in enumerate(listmasslens):
                    radieins = chalcedon.retr_radieinssbin(masslens, smax)
                    if masslens < 1.5:
                        masswdrf = masslens
                        radiwdrf = 0.007 * masswdrf**(-1. / 3.)
                        axis.axhline(radiwdrf, ls='--', color=listcolr[k])
                    axis.plot(smax, radieins, color=listcolr[k])
                axis.set_xlabel('$a$ [AU]')
                axis.set_ylabel('$R$ [$R_\odot$]')
                axis.set_xscale('log')
                axis.set_yscale('log')
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
            # plot amplitude vs. orbital period for three components of the light curve of a COSC
            path = gdat.pathvisu + 'amplslen.%s' % gdat.typefileplot
            if not os.path.exists(path):
                radistar = 1.
                massstar = 1.
                densstar = 1.41 # [g/cm^3]
                
                listcolr = ['g', 'b', 'r']
                listlsty = ['-', '--', ':']
                figr, axis = plt.subplots(figsize=(6, 4.5))
                arryperi = np.linspace(0.3, 30., 100)
                listmasscomp = [5., 30., 180.]
                numbmasscomp = len(listmasscomp)
                indxmasscomp = np.arange(numbmasscomp)
                for k in indxmasscomp:

                    amplbeam = nicomedia.retr_deptbeam(arryperi, massstar, listmasscomp[k])
                    amplelli = nicomedia.retr_deptelli(arryperi, densstar, massstar, listmasscomp[k])
                    amplslen = chalcedon.retr_amplslen(arryperi, radistar, listmasscomp[k], massstar)
                    
                    lablmass= 'M=%.3g $M_\odot$' % listmasscomp[k]
                    labl = 'DB, %s' % lablmass
                    axis.plot(arryperi, amplbeam, ls=listlsty[k], color=listcolr[0], label=labl)
                    labl = 'EV, %s' % lablmass
                    axis.plot(arryperi, amplelli, ls=listlsty[k], color=listcolr[1], label=labl)
                    labl = 'SL, %s' % lablmass
                    axis.plot(arryperi, amplslen, ls=listlsty[k], color=listcolr[2], label=labl)
                axis.set_xlabel('Orbital Period [days]')
                axis.set_ylabel('Amplitude [ppt]')
                axis.set_xscale('log')
                axis.set_yscale('log')
                axis.legend()
                axis.set_xlim([0.3, 30])
                axis.set_ylim([5e-3, 5e2])
                plt.tight_layout()
                plt.savefig(path)
                plt.close()
            

            # type of lens model
            gdat.typemodllens = 'gaus'

            # plot model light curves for COSCs with different orbital periods
            time = np.arange(0., 20., 2. / 24. / 60.)
            listperi = [3., 6., 9.]
            numbperi = len(listperi)
            indxperi = np.arange(numbperi)
            para = np.empty(6)
            for k in indxperi:
                path = gdat.pathvisu + 'fig%d.%s' % (k + 1, gdat.typefileplot)
                if not os.path.exists(path):
                    figr, axis = plt.subplots(figsize=(10, 4.5))
                    
                    dictoutp = ephesos.eval_modl(time, pericomp=[listperi[k]], epocmtracomp=[0.], radistar=1., massstar=1., \
                                                                                     masscomp=[10.], inclcomp=[90.], typesyst='CompactObjectStellarCompanion', typemodllens=gdat.typemodllens)
                    rflxmodl = dictoutp['rflx']
                    axis.plot(time, rflxmodl, color='k', lw=2, label='Total')
                    axis.plot(time, dictoutp['rflxelli'][0], color='b', ls='--', label='Ellipsoidal variation')
                    axis.plot(time, dictoutp['rflxbeam'][0], color='g', ls='--', label='Beaming')
                    axis.plot(time, dictoutp['rflxslen'][0], color='r', ls='--', label='Self-lensing')
                    axis.set_title('Orbital period = %.3g days' % listperi[k])
                    axis.legend()
                    axis.set_ylabel('Relative flux')
                    axis.set_xlabel('Time [days]')
                    plt.savefig(path)
                    plt.close()
    
            # plot occurence rate
            occufiel = np.array([ \
                                [1.7e-7, 2.2e-7, 2.2e-7, 2.2e-7, 2.5e-7], \
                                [5.6e-7, 7.7e-7, 8.4e-7, 8.6e-7, 1.0e-6], \
                                [2.2e-6, 3.7e-6, 4.0e-6, 4.3e-6, 5.3e-6], \
                                [4.3e-6, 9.8e-6, 1.0e-5, 1.1e-5, 1.4e-5], \
                                [8.9e-6, 2.3e-5, 2.8e-5, 3.0e-5, 3.6e-5], \
                                ])
            occucoen = np.array([ \
                                [1.3e-6, 1.8e-6, 1.9e-6, 6.7e-7, 8.7e-9], \
                                [2.3e-6, 3.1e-6, 3.1e-6, 7.9e-7, 3.9e-10], \
                                [4.7e-6, 7.9e-6, 1.9e-6, 1.5e-9, 2.2e-10], \
                                [8.4e-6, 1.8e-5, 1.2e-5, 2.7e-6, 2.0e-9], \
                                [8.2e-6, 1.5e-5, 8.9e-6, 1.3e-8, 2.5e-9], \
                                ])
            
            peri = np.array([0.3, 0.8, 1.9, 4.8, 11.9, 30.])
            masscomp = np.array([5.0, 9.1, 16.6, 30.2, 54.9, 100.])
            peri = np.exp((np.log(peri)[1:] + np.log(peri)[:-1]) / 2.)
            masscomp = np.exp((np.log(masscomp)[1:] + np.log(masscomp)[:-1]) / 2.)
            occufielintp = scipy.interpolate.interp2d(peri, masscomp, occufiel)#, kind='linear')
            occucoenintp = scipy.interpolate.interp2d(peri, masscomp, occucoen)#, kind='linear')
            peri = np.linspace(1., 25., 100)
            masscomp = np.linspace(5., 80., 100)
            for k in range(2):
                if k == 0:
                    strg = 'fiel'
                if k == 1:
                    strg = 'coen'
                path = gdat.pathvisu + 'occ_%s.%s' % (strg, gdat.typefileplot)
                if not os.path.exists(path):
                    figr, axis = plt.subplots(figsize=(6, 4.5))
                    if k == 0:
                        data = occufielintp
                    if k == 1:
                        data = occucoenintp
                    c = plt.pcolor(peri, masscomp, data(peri, masscomp), norm=matplotlib.colors.LogNorm(), cmap ='Greens')#, vmin = z_min, vmax = z_max)
                    plt.colorbar(c)
                    axis.set_xlabel('Orbital Period [days]')
                    axis.set_xlabel('CO mass')
                    plt.savefig(path)
                    plt.close()
    
            ## plot TESS photometric precision
            path = gdat.pathvisu + 'sigmtmag.%s' % (gdat.typefileplot) 
            if not os.path.exists(path):
                dictpoplticim110 = nicomedia.retr_dictpopltic8(typepopl='TIC_m110')
       
                ## interpolate TESS photometric precision
                dictpoplticim110['nois'] = nicomedia.retr_noistess(dictpoplticim110['tmag'])

                figr, axis = plt.subplots(figsize=(6, 4))
                print('dictpoplticim110[nois]')
                summgene(dictpoplticim110['nois'])
                print('dictpoplticim110[tmag]')
                summgene(dictpoplticim110['tmag'])
                axis.scatter(dictpoplticim110['tmag'], dictpoplticim110['nois'], rasterized=True)
                axis.set_xlabel('Tmag')
                axis.set_ylabel(r'$s$')
                axis.set_yscale('log')
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
            
            # plot SNR
            path = gdat.pathvisu + 'sigm.%s' % (gdat.typefileplot) 
            if not os.path.exists(path):
                figr, axis = plt.subplots(figsize=(5, 3))
                peri = np.logspace(-1, 2, 100)
                listtmag = [10., 13., 16.]
                listmasscomp = [1., 10., 100.]
                massstar = 1.
                radistar = 1.
                for masscomp in listmasscomp:
                    amplslentmag = chalcedon.retr_amplslen(peri, radistar, masscomp, massstar)
                    axis.plot(peri, amplslentmag, label=r'M = %.3g M$_\odot$' % masscomp)
                for tmag in listtmag:
                    noistess = nicomedia.retr_noistess(tmag)
                    if tmag == 16:
                        axis.text(0.1, noistess * 1.6, ('Tmag = %.3g' % tmag),  color='black')
                    else:
                        axis.text(0.1, noistess / 2, ('Tmag = %.3g' % tmag),  color='black')
                    axis.axhline(noistess, ls='--', color='black')#, label=('Tmag = %.3g' % tmag))
                axis.set_xlabel('Period [day]')
                axis.set_ylabel(r'Self-lensing amplitude')
                axis.set_xscale('log')
                axis.set_yscale('log')
                axis.legend(loc=4, framealpha=1.)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
    
    # target labels and file name extensions
    gdat.strgtarg = [[] for n in gdat.indxtarg]
    if gdat.liststrgmast is None:
        gdat.liststrgmast = [[] for n in gdat.indxtarg]
    gdat.labltarg = [[] for n in gdat.indxtarg]
    for n in gdat.indxtarg:
        if gdat.booltargsynt:
            gdat.strgtarg[n] = 'simugene%04d' % n
            gdat.labltarg[n] = 'Simulated target'
        else:
            if gdat.typepopl[4:12] == 'prms2min':
                gdat.strgtarg[n] = 'TIC%d' % (gdat.listticitarg[n])
                gdat.labltarg[n] = 'TIC %d' % (gdat.listticitarg[n])
                gdat.liststrgmast[n] = gdat.labltarg[n]
            if gdat.booltargusertici:
                gdat.labltarg[n] = 'TIC ' + str(gdat.listticitarg[n])
                gdat.strgtarg[n] = 'TIC' + str(gdat.listticitarg[n])
            if gdat.booltargusermast:
                gdat.labltarg[n] = gdat.liststrgmast[n]
                gdat.strgtarg[n] = ''.join(gdat.liststrgmast[n].split(' '))
            if gdat.booltargusergaid:
                gdat.labltarg[n] = 'GID=' % (dictcatlrvel['rasc'][n], dictcatlrvel['decl'][n])
                gdat.strgtarg[n] = 'R%.4gDEC%.4g' % (dictcatlrvel['rasc'][n], dictcatlrvel['decl'][n])
    
    gdat.listarrytser = dict()
    
    if gdat.boolsimusome:
    
        #gdat.listlablrele = ['Simulated %s' % gdat.typesyst, 'Simulated tr. COSC' % gdat.typesyst]
        #gdat.listlablirre = ['Simulated QS or SB', 'Simulated QS, SB or non-tr. COSC']
        
        # labels of the relevant classes
        gdat.listlablrele = ['Simulated %s' % gdat.typesyst]
        

        # labels of the irrelevant classes
        gdat.listlablirre = ['Simulated no signal']
        gdat.listlablreleirre = [gdat.listlablrele, gdat.listlablirre]
    
        # number of relevant types
        gdat.numbtyperele = len(gdat.listlablrele)
        gdat.indxtyperele = np.arange(gdat.numbtyperele)
        gdat.boolreletarg = [np.empty(gdat.numbtarg, dtype=bool) for v in gdat.indxtyperele]
    
    # number of analyses
    gdat.numbtypeposi = 4
    gdat.indxtypeposi = np.arange(gdat.numbtypeposi)
    gdat.boolpositarg = [np.empty(gdat.numbtarg, dtype=bool) for u in gdat.indxtypeposi]
    
    gdat.dictindxtarg = dict()
    gdat.dicttroy = dict()
    
    if gdat.boolsimusome:
        
        listcolrtypetrue = np.array(['g', 'b', 'orange', 'olive'])
        # types of systems
        #listnametypetrue = ['totl', 'StellarBinary', 'StellarSystem', 'CompactObjectStellarCompanion', 'qstr', 'cosctran']
        #listlabltypetrue = ['All', 'Stellar binary', 'Stellar System', 'Compact object with Stellar Companion', 'QS', 'Tr. COSC']
        #listcolrtypetrue = ['black', 'g', 'b', 'orange', 'yellow', 'olive']
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            listlabltypetrue = ['Stellar binary', 'Eclipsing Binary', 'Compact object with Stellar Companion', 'Transiting Compact object with Stellar Companion']
        elif gdat.typesyst == 'PlanetarySystem':
            listlabltypetrue = ['Planetary System', 'Eclipsing Binary']
        elif gdat.typesyst == 'StarFlaring':
            listlabltypetrue = ['StarFlaring']
        else:
            print('')
            print('')
            print('')
            print('gdat.typesyst')
            print(gdat.typesyst)
            raise Exception('')
        
        numbtypetrue = len(listlabltypetrue)
        indxtypetrue = np.arange(numbtypetrue)
        
        listnametypetrue = []
        for k in indxtypetrue:
            nametypetrue = ''.join(listlabltypetrue[k].split(' '))
            listnametypetrue.append(nametypetrue)
        
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
            listdictlablcolrpopl[-1]['PlanetarySystem_%s_Transiting' % gdat.typepopl] = ['All', 'black']
        elif gdat.typesyst == 'PlanetarySystem':
            listdictlablcolrpopl[-1]['PlanetarySystem_%s_All' % gdat.typepopl] = ['All', 'black']
            listdictlablcolrpopl[-1]['PlanetarySystem_%s_Transiting' % gdat.typepopl] = ['All', 'black']
        
            gdat.dictindxtarg['PlanetarySystem'] = np.arange(gdat.numbtarg)
            gdat.dictindxtarg['StellarSystem'] = np.arange(gdat.numbtarg)
            
        elif gdat.typesyst == 'StarFlaring':
            listdictlablcolrpopl[-1]['StarFlaring_%s_All' % gdat.typepopl] = ['All', 'black']
            listdictlablcolrpopl[-1]['StarFlaring_%s_Mdwarfs' % gdat.typepopl] = ['All', 'red']
        else:
            raise Exception('')

        #for namepoplcomm in listnametypetrue:
        #    gdat.dicttroy['true'][namepoplcomm] = dict()

        if gdat.typesyst == 'CompactObjectStellarCompanion':
            # 0: cosc
            # 1: binary star
            # 2: single star
            gdat.probtypetrue = np.array([0.7, 0.25, 0.05])
        
        gdat.numbtypetrue = gdat.probtypetrue.size
        gdat.indxtypetrue = np.arange(gdat.numbtypetrue)
        gdat.typetruetarg = np.random.choice(gdat.indxtypetrue, size=gdat.numbtarg, p=gdat.probtypetrue)
            
        gdat.indxtypetruetarg = [[] for r in gdat.indxtypetrue]
        for r in gdat.indxtypetrue:
            gdat.indxtypetruetarg[r] = np.where(gdat.typetruetarg == r)[0]
        
        gdat.booltypetargtrue = dict()
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            gdat.booltypetargtrue['CompactObjectStellarCompanion'] = gdat.typetruetarg == 0
            gdat.booltypetargtrue['StellarBinary'] = gdat.typetruetarg == 1
        
        gdat.listnameclastype = list(gdat.dictindxtarg.keys())
        
        #if gdat.typesyst == 'CompactObjectStellarCompanion' or gdat.typesyst == 'PlanetarySystem':
        

        gdat.numbtargtype = dict()
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            gdat.dictindxtarg['CompactObjectStellarCompanion'] = gdat.indxtypetruetarg[0]
            gdat.dictindxtarg['StellarBinary'] = gdat.indxtypetruetarg[1]
            gdat.dictindxtarg['qstr'] = gdat.indxtypetruetarg[2]
            gdat.dictindxtarg['StellarSystem'] = np.concatenate((gdat.indxtypetruetarg[0], gdat.indxtypetruetarg[1]))
            gdat.dictindxtarg['StellarSystem'] = np.sort(gdat.dictindxtarg['StellarSystem'])
            
        print('gdat.numbtargtype')
        for name in gdat.listnameclastype:
            gdat.numbtargtype[name] = gdat.dictindxtarg[name].size
            print(name)
            print(gdat.numbtargtype[name])

        print('gdat.numbtarg')
        print(gdat.numbtarg)
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            print('gdat.boolcosctrue')
            print(gdat.boolcosctrue)
            print('gdat.boolsbintrue')
            print(gdat.boolsbintrue)
            
            gdat.indxssyscosc = np.full(gdat.numbtargcosc, -1, dtype=int)
            gdat.indxssyssbin = np.full(gdat.numbtargsbin, -1, dtype=int)
            gdat.indxcoscssys = np.full(gdat.numbtargssys, -1, dtype=int)
            gdat.indxsbinssys = np.full(gdat.numbtargssys, -1, dtype=int)
            gdat.indxssystarg = np.full(gdat.numbtarg, -1, dtype=int)
            cntrcosc = 0
            cntrsbin = 0
            cntrssys = 0
            for k in gdat.indxtarg:
                if gdat.boolcosctrue[k] or gdat.boolsbintrue[k]:
                    if gdat.boolcosctrue[k]:
                        gdat.indxcoscssys[cntrssys] = cntrcosc
                        gdat.indxssyscosc[cntrcosc] = cntrssys
                        cntrcosc += 1
                    if gdat.boolsbintrue[k]:
                        gdat.indxsbinssys[cntrssys] = cntrsbin
                        gdat.indxssyssbin[cntrsbin] = cntrssys
                        cntrsbin += 1
                    gdat.indxssystarg[k] = cntrssys
                    cntrssys += 1
        
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            gdat.dicttroy['true']['CompactObjectStellarCompanion'] = nicomedia.retr_dictpoplstarcomp('CompactObjectStellarCompanion', gdat.typepopl, minmnumbcompstar=1)
            gdat.indxcompcosc = gdat.dicttroy['true']['CompactObjectStellarCompanion']['dictindx']['comp']['star']
            gdat.dicttroy['true']['StellarBinary'] = nicomedia.retr_dictpoplstarcomp('StellarBinary', gdat.typepopl)
            gdat.indxcompsbin = gdat.dicttroy['true']['StellarBinary']['dictindx']['comp']['star']
        elif gdat.typesyst == 'PlanetarySystem':
            gdat.dicttroy['true']['PlanetarySystem'] = nicomedia.retr_dictpoplstarcomp('PlanetarySystem', gdat.typepopl, minmnumbcompstar=1)
            gdat.indxcompsyst = gdat.dicttroy['true']['PlanetarySystem']['dictindx']['comp']['star']
        elif gdat.typesyst == 'StarFlaring':
            gdat.dicttroy['true']['StarFlaring'] = nicomedia.retr_dictpoplstarcomp('StarFlaring', gdat.typepopl)
        else:
            raise Exception('')

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
            
                # maybe to be deleted
                #gdat.dicttroy['true']['StellarSystem']['dictpopl']['comp'][gdat.namepoplcomp] = dict()
                #for name in listname:
                #    gdat.dicttroy['true']['StellarSystem'][gdat.namepoplcomp][name] = np.concatenate([gdat.dicttroy['true']['CompactObjectStellarCompanion'][gdat.namepoplcomp][name], \
                #                                                                                    gdat.dicttroy['true']['StellarBinary'][gdat.namepoplcomp][name]])
        
            #if gdat.typedata == 'simutargpartinje':
            #    boolsampstar = False
            #    gdat.dicttroy['true']['StellarSystem']['radistar'] = dicttic8['radistar']
            #    gdat.dicttroy['true']['StellarSystem']['massstar'] = dicttic8['massstar']
            #    indx = np.where((~np.isfinite(gdat.dicttroy['true']['StellarSystem']['massstar'])) | (~np.isfinite(gdat.dicttroy['true']['StellarSystem']['radistar'])))[0]
            #    gdat.dicttroy['true']['StellarSystem']['radistar'][indx] = 1.
            #    gdat.dicttroy['true']['StellarSystem']['massstar'][indx] = 1.
            #    gdat.dicttroy['true']['totl']['tmag'] = dicttic8['tmag']
            
            # merge the features of the simulated COSCs and SBs
            gdat.dicttroy['true']['totl'] = dict()
            for namefeat in ['tmag']:
                if gdat.booldiag:
                    if not gdat.namepoplcomptotl in gdat.dicttroy['true']['CompactObjectStellarCompanion']['dictpopl']['comp']:
                        print('')
                        print('')
                        print('')
                        raise Exception('not gdat.namepoplcomptotl in gdat.dicttroy[true][CompactObjectStellarCompanion][dictpopl][comp]')

                gdat.dicttroy['true']['totl']['tmag'] = \
                        np.concatenate([gdat.dicttroy['true']['CompactObjectStellarCompanion']['dictpopl']['comp'][gdat.namepoplcomptotl][namefeat], \
                                                        gdat.dicttroy['true']['StellarBinary']['dictpopl']['comp'][gdat.namepoplcomptotl][namefeat]])

        # grab the photometric noise of TESS as a function of TESS magnitude
        #if gdat.typedata == 'simutargsynt':
        #    gdat.stdvphot = nicomedia.retr_noistess(gdat.dicttroy['true']['totl']['tmag']) * 1e-3 # [dimensionless]
        #    
        #    if not np.isfinite(gdat.stdvphot).all():
        #        raise Exception('')
        
        
        print('Visualizing the features of the simulated population...')

        listboolcompexcl = [False]
        listtitlcomp = ['Binaries']
        
        dictpopltemp = dict()
        if gdat.typesyst == 'CompactObjectStellarCompanion':
            dictpopltemp['CompactObjectStellarCompanion' + gdat.typepopl + 'totl'] = gdat.dicttroy['true']['CompactObjectStellarCompanion']['dictpopl']['comp'][gdat.namepoplcomptotl]
            dictpopltemp['StellarBinary' + gdat.typepopl + 'totl'] = gdat.dicttroy['true']['StellarBinary']['dictpopl']['comp'][gdat.namepoplcomptotl]
            dictpopltemp['CompactObjectStellarCompanion' + gdat.typepopl + 'Transiting'] = gdat.dicttroy['true']['CompactObjectStellarCompanion']['dictpopl']['comp'][gdat.namepoplcomptran]
            dictpopltemp['StellarBinary' + gdat.typepopl + 'Transiting'] = gdat.dicttroy['true']['StellarBinary']['dictpopl']['comp'][gdat.namepoplcomptran]
        elif gdat.typesyst == 'PlanetarySystem':
            dictpopltemp['PlanetarySystem_%s_All' % gdat.typepopl] = gdat.dicttroy['true']['PlanetarySystem']['dictpopl']['comp'][gdat.namepoplcomptotl]
            dictpopltemp['PlanetarySystem_%s_Transiting' % gdat.typepopl] = gdat.dicttroy['true']['PlanetarySystem']['dictpopl']['comp'][gdat.namepoplcomptran]
        elif gdat.typesyst == 'StarFlaring':
            dictpopltemp['StarFlaring_%s_All' % gdat.typepopl] = gdat.dicttroy['true']['StarFlaring']['dictpopl']['flar'][gdat.namepoplcomptotl]
            dictpopltemp['StarFlaring_%s_Mdwarfs' % gdat.typepopl] = gdat.dicttroy['true']['StarFlaring']['dictpopl']['flar'][gdat.namepoplcomptotl]
        else:
            raise Exception('')
        
        typeanls = '%s_%s_%s_%s' % (gdat.typesyst, gdat.strgtypedataconc, gdat.strginstconc, gdat.typepopl)

        pathvisu = gdat.pathvisupopl + 'True_Features/'
        pathdata = gdat.pathdatapopl + 'True_Features/'
        
        lablnumbsamp = 'Number of binaries'

        pergamon.init( \
                      typeanls, \
                      dictpopl=dictpopltemp, \
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
            indx = np.where(np.isfinite(gdat.dicttroy['true']['StellarSystem']['dictpopl']['comp'][gdat.namepoplcomptran]['duratrantotl'][gdat.indxssyscosc]))
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
                if n in gdat.dictindxtarg['StellarSystem']:
                    gdat.indxssysrele[v][cntrrele] = cntrssys
                    cntrssys += 1
                cntrrele += 1

    #if gdat.boolsimusome:
        # move TESS magnitudes from the dictinary of all systems to the dictionaries of each types of system
        #for namepoplcomm in listnametypetrue:
        #    if namepoplcomm != 'totl':
        #        gdat.dicttroy['true'][namepoplcomm]['tmag'] = gdat.dicttroy['true']['totl']['tmag'][gdat.dictindxtarg[namepoplcomm]]
    
    gdat.pathlogg = gdat.pathdata + 'logg/'
    
    # output features of miletos
    gdat.dictstat = dict()

    gdat.listnamefeat = ['peripboxprim', 'sdeepboxprim', 'perilspeprim', 'powrlspeprim']
    for namefeat in gdat.listnamefeat:
        gdat.dictstat[namefeat] = np.empty(gdat.numbtarg)
    
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
    if not 'dictpboxinpt' in gdat.dictmileinptglob:
        gdat.dictmileinptglob['dictpboxinpt'] = dict()
    
    # inputs to the periodic box search pipeline
    gdat.dictmileinptglob['dictpboxinpt']['boolsrchposi'] = True
    gdat.dictmileinptglob['dictpboxinpt']['boolprocmult'] = False
    
    if gdat.boolsimusome:
        gdat.boolreleposi = [[[] for v in gdat.indxtyperele] for u in gdat.indxtypeposi]
        gdat.boolposirele = [[[] for v in gdat.indxtyperele] for u in gdat.indxtypeposi]
        
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
    
    gdat.listlablposi = ['Strong BLS power', 'Strong LS power', 'Strong BLS or LS power', 'Strong BLS and LS power']
    gdat.listlablnega = ['Weak BLS power', 'Weak LS power', 'Weak BLS and LS power', 'Weak BLS or LS power']
            
    # for each positive and relevant type, estimate the recall and precision
    gdat.indxtypeposiiter = np.concatenate((np.array([-1]), gdat.indxtypeposi))
    if gdat.boolsimusome:
        gdat.indxtypereleiter = np.concatenate((np.array([-1]), gdat.indxtyperele))
    else:
        gdat.indxtypereleiter = np.array([-1])
    
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

            if u == -1:
                strguuvv = 'v%d' % (v)
                labluuvv = '(v = %d)' % (v)
            elif v == -1:
                strguuvv = 'u%d' % (u)
                labluuvv = '(u = %d)' % (u)
            else:
                strguuvv = 'u%dv%d' % (u, v)
                labluuvv = '(u = %d, v = %d)' % (u, v)

            gdat.dictindxtargtemp = dict()
            gdat.dicttroy['stat'] = dict()

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
                    gdat.dicttroy['stat']['stat' + strgkeyy] = dict()
                    for namefeat in gdat.listnamefeat:
                        gdat.dicttroy['stat']['stat' + strgkeyy][namefeat] = gdat.dictstat[namefeat][gdat.dictindxtargtemp[strgkeyy]]
            
            listdictlablcolrpopl = []
            listboolcompexcl = []
            listtitlcomp = []
            listnamepoplcomm = list(gdat.dicttroy['stat'].keys())
            strgtemp = 'stat' + strguuvv
            
            print('strguuvv')
            print(strguuvv)
            print('strgtemp')
            print(strgtemp)
            print('listnamepoplcomm')
            print(listnamepoplcomm)
            
            boolgood = False
            for namepoplcomm in listnamepoplcomm:
                if strgtemp + 're' in namepoplcomm:
                    boolgood = True
                if strgtemp + 'ir' in namepoplcomm:
                    boolgood = True
            if boolgood:
                listdictlablcolrpopl.append(dict())
                listboolcompexcl.append(True)
                listtitlcomp.append(None)
            for namepoplcomm in listnamepoplcomm:
                if strgtemp + 're' in namepoplcomm:
                    print('v')
                    print(v)
                    print('gdat.listlablrele')
                    print(gdat.listlablrele)
                    print('listdictlablcolrpopl')
                    print(listdictlablcolrpopl)
                    print('gdat.indxtypereleiter')
                    print(gdat.indxtypereleiter)
                    print('gdat.indxtyperele')
                    print(gdat.indxtyperele)

                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablreleirre[v], 'blue']
                if strgtemp + 'ir' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablreleirre[v], 'orange']
            
            boolgood = False
            for namepoplcomm in listnamepoplcomm:
                if strgtemp + 'po' in namepoplcomm:
                    boolgood = True
                if strgtemp + 'ne' in namepoplcomm:
                    boolgood = True
            if boolgood:
                listdictlablcolrpopl.append(dict())
                listboolcompexcl.append(True)
                listtitlcomp.append(None)
            for namepoplcomm in listnamepoplcomm:
                if strgtemp + 'po' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablposi[u], 'violet']
                if strgtemp + 'ne' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablnega[u], 'brown']
            
            boolgood = False
            for namepoplcomm in listnamepoplcomm:
                if strgtemp + 'trpo' in namepoplcomm:
                    boolgood = True
                if strgtemp + 'trne' in namepoplcomm:
                    boolgood = True
                if strgtemp + 'flpo' in namepoplcomm:
                    boolgood = True
                if strgtemp + 'flne' in namepoplcomm:
                    boolgood = True
            if boolgood:
                listdictlablcolrpopl.append(dict())
                listboolcompexcl.append(True)
                listtitlcomp.append(None)
            for namepoplcomm in listnamepoplcomm:
                if strgtemp + 'trpo' in namepoplcomm:
                    print('u, v')
                    print(u, v)
                    print('gdat.listlablrele')
                    print(gdat.listlablrele)
                    print('gdat.listlablposi')
                    print(gdat.listlablposi)
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablrele[v] + ', ' + gdat.listlablposi[u], 'green']
                if strgtemp + 'trne' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablirre[v] + ', ' + gdat.listlablnega[u], 'blue']
                if strgtemp + 'flpo' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablirre[v] + ', ' + gdat.listlablposi[u], 'red']
                if strgtemp + 'flne' in namepoplcomm:
                    listdictlablcolrpopl[-1][namepoplcomm] = [gdat.listlablrele[v] + ', ' + gdat.listlablnega[u], 'orange']
            
            typeanls = 'cosc_%s_%s_%s' % (gdat.strgextn, gdat.strginstconc, gdat.typepopl)
            print('typeanls')
            print(typeanls)
            print('listdictlablcolrpopl')
            print(listdictlablcolrpopl)
            print('listboolcompexcl')
            print(listboolcompexcl)
            print('listtitlcomp')
            print(listtitlcomp)
            print('gdat.dicttroy[stat]')
            print(gdat.dicttroy['stat'])
            
            for dictlablcolrpopl in listdictlablcolrpopl:
                if len(dictlablcolrpopl) == 0:
                    raise Exception('')

            pergamon.init( \
                          'targ_cosc', \
                          dictpopl=gdat.dicttroy['stat'], \
                          listdictlablcolrpopl=listdictlablcolrpopl, \
                          listboolcompexcl=listboolcompexcl, \
                          listtitlcomp=listtitlcomp, \
                          pathvisu=gdat.pathvisupopl, \
                          pathdata=gdat.pathdatapopl, \
                          lablsampgene='exoplanet', \
                          boolsortpoplsize=False, \
                         )
            

            if gdat.boolplot and gdat.boolsimusome and u != -1 and v != -1:
                listvarbreca = []
                
                listvarbreca.append(gdat.dicttroy['true'][gdat.typesyst]['dictpopl']['comp'][gdat.namepoplcomptotl]['pericomp'][gdat.indxssysrele[v]])
                #listvarbreca.append(gdat.dicttroy['true'][gdat.typesyst]['dictpopl']['comp'][gdat.namepoplcomptotl]['masscomp'][gdat.indxssysrele[v]])
                listvarbreca.append(gdat.dicttroy['true'][gdat.typesyst]['dictpopl']['comp'][gdat.namepoplcomptotl]['tmag'][gdat.dictindxtarg['rele'][v]])
                listvarbreca = np.vstack(listvarbreca).T
                
                liststrgvarbreca = []
                liststrgvarbreca.append('trueperi')
                #liststrgvarbreca.append('truemasscomp')
                liststrgvarbreca.append('truetmag')
                
                #listlablvarbreca = [['$P$', 'day'], ['$M_c$', '$M_\odot$'], ['Tmag', '']]
                
                listlablvarbreca, listscalvarbreca, _, _, _ = tdpy.retr_listlablscalpara(liststrgvarbreca)
                
                listtemp = []
                for namefeat in gdat.listnamefeat:
                    listtemp.append(gdat.dictstat[namefeat][gdat.dictindxtarg['posi'][u]])
                listvarbprec = np.vstack(listtemp).T
                #listvarbprec = np.vstack([gdat.listsdee, gdat.listpowrlspe]).T
                #listlablvarbprec = [['SDE', ''], ['$P_{LS}$', '']]
                liststrgvarbprec = gdat.listnamefeat#['sdeecomp', 'powrlspe']
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
                tdpy.plot_recaprec(gdat.pathvisupopl, strgextn, listvarbreca, listvarbprec, liststrgvarbreca, liststrgvarbprec, \
                                        listlablvarbreca, listlablvarbprec, gdat.boolposirele[u][v], gdat.boolreleposi[u][v])


        #if gdat.typeanls == 'CompactObjectStellarCompanion' or gdat.typeanls == 'psys' or gdat.typeanls == 'plan':
        #    
        #        # calculate photometric precision for the star population
        #        if typeinst.startswith('tess'):
        #            gdat.dictpopl[namepoplcomptran]['nois'] = nicomedia.retr_noistess(gdat.dictpopl[namepoplcomptran]['tmag'])
        #        elif typeinst.startswith('lsst'):
        #            gdat.dictpopl[namepoplcomptran]['nois'] = nicomedia.retr_noislsst(gdat.dictpopl[namepoplcomptran]['rmag'])
        #    
        #        # expected BLS signal detection efficiency
        #        if typeinst.startswith('lsst'):
        #            numbvisi = 1000
        #            gdat.dictpopl[namepoplcomptran]['sdee'] = gdat.dictpopl[namepoplcomptran]['depttrancomp'] / 5. / gdat.dictpopl[namepoplcomptran]['nois'] * \
        #                                                                                                 np.sqrt(gdat.dictpopl[namepoplcomptran]['dcyc'] * numbvisi)
        #        if typeinst.startswith('tess'):
        #            if gdat.typeanls == 'plan':
        #                gdat.dictpopl[namepoplcomptran]['sdee'] = np.sqrt(gdat.dictpopl[namepoplcomptran]['duratrantotl']) * \
        #                                                                    gdat.dictpopl[namepoplcomptran]['depttrancomp'] / gdat.dictpopl[namepoplcomptran]['nois']
        #            if gdat.typeanls == 'CompactObjectStellarCompanion':
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




