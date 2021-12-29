import os, sys, datetime, fnmatch, copy

import matplotlib as mpl
import matplotlib.pyplot as plt

import astroquery

import numpy as np
import scipy.interpolate

import json

from tdpy.util import summgene
import tdpy
import ephesus
import miletos
import pergamon

def retr_angleins(masslens, distlenssour, distlens, distsour):
    '''
    Return Einstein radius.
    '''
    
    angleins = np.sqrt(masslens / 10**(11.09) * distlenssour / distlens / distsour)
    
    return angleins


def retr_dictderi_effe(para, gdat):
    
    radistar = para[0]
    peri = para[1]
    masscomp = para[2]
    massstar = para[3]
    masstotl = massstar + masscomp

    amplslenmodl = ephesus.retr_amplslen(peri, radistar, masscomp, massstar)
    duratrantotlmodl = ephesus.retr_duratrantotl(peri, radistar, masscomp, massstar, incl)
    smax = ephesus.retr_smaxkepl(peri, masstotl) * 215. # [R_S]
    radischw = 4.24e-6 * masscomp # [R_S]

    dictvarbderi = None

    dictparaderi = dict()
    dictparaderi['amplslenmodl'] = np.array([amplslenmodl])
    dictparaderi['duratrantotlmodl'] = np.array([duratrantotlmodl])
    dictparaderi['smaxmodl'] = np.array([smax])
    dictparaderi['radischw'] = np.array([radischw])

    return dictparaderi, dictvarbderi
    

def retr_dflxslensing(time, epocslen, amplslen, duratrantotl):
    
    timediff = time - epocslen
    
    dflxslensing = 1e-3 * amplslen * np.heaviside(duratrantotl / 48. + timediff, 0.5) * np.heaviside(duratrantotl / 48. - timediff, 0.5)
    
    return dflxslensing


def mile_work(gdat, i):
    
    for n in gdat.listindxtarg[i]:
        
        if gdat.typedata == 'toyy' or gdat.typedata == 'mock':
            print('heeey')
            print('n')
            print(n)
            for v in gdat.indxtyperele:
                print('gdat.indxtargrele[v]')
                print(gdat.indxtargrele[v])
                if n in gdat.indxtargrele[v]:
                    gdat.boolreletarg[v][n] = True
                else:
                    gdat.boolreletarg[v][n] = False

        cntr = 0
        for p in gdat.indxinst[0]:
            for y in gdat.indxchun[n][0][p]:
                cntr += gdat.listarrytser['data'][n][0][p][y].shape[0]
        if cntr == 0:
            print('No data on %s! Skipping...' % gdat.labltarg[n])
            raise Exception('')
            continue

        if gdat.typedata == 'obsd' and gdat.typepopl == 'list':
            listarrytser = None
            rasctarg = None
            decltarg = None
            strgmast = liststrgmast[n]
            labltarg = None
            strgtarg = None

        else:
            listarrytser = dict()
            listarrytser['raww'] = gdat.listarrytser['data'][n]
            
            rasctarg = None
            decltarg = None
            strgmast = None
            labltarg = gdat.labltarg[n]
            strgtarg = gdat.strgtarg[n]
            if len(strgtarg) == 0:
                raise Exception('')
        
        gdat.dictmileinpttarg = copy.deepcopy(gdat.dictmileinpt)

        if n < gdat.maxmnumbtargplot:
            gdat.dictmileinpt['boolplot'] = gdat.boolplotmile
        else:
            gdat.dictmileinpt['boolplot'] = False
        gdat.dictmileinpttarg['rasctarg'] = rasctarg
        gdat.dictmileinpttarg['decltarg'] = decltarg
        gdat.dictmileinpttarg['strgtarg'] = strgtarg
        gdat.dictmileinpttarg['labltarg'] = labltarg
        gdat.dictmileinpttarg['strgmast'] = strgmast
        gdat.dictmileinpttarg['listarrytser'] = listarrytser
        
        # call miletos to analyze data
        dictmileoutp = miletos.init( \
                                    **gdat.dictmileinpttarg, \
                                   )
        
        gdat.dictstat['perilspeprim'][n] = dictmileoutp['perilspempow']
        gdat.dictstat['powrlspeprim'][n] = dictmileoutp['powrlspempow']
        
        gdat.dictstat['sdeepboxprim'][n] = dictmileoutp['dictpboxoutp']['sdee'][0]
        gdat.dictstat['peripboxprim'][n] = dictmileoutp['dictpboxoutp']['peri'][0]
        
        # taking the fist element, which belongs to the first TCE
        for u in gdat.indxtypeposi:
            gdat.boolpositarg[u][n] = dictmileoutp['boolposianls'][u]
        
        if gdat.boolplot and n < gdat.maxmnumbtargplot:
            if gdat.boolplotdvrp:
                if gdat.typedata == 'toyy' or gdat.typedata == 'mock':
                    gdat.numbpagedvrp = gdat.numbpagedvrpmock
                else:
                    gdat.numbpagedvrp = 0
        
                gdat.listpathdvrp = [[] for k in range(gdat.numbpagedvrp)]
        
                gdat.numbpagedvrp += len(dictmileoutp['listpathdvrp'])
                gdat.indxpagedvrp = np.arange(gdat.numbpagedvrp)
                
                ## list of dictionaries holding the paths and DV report positions of plots
                gdat.listdictdvrp = [[] for k in gdat.indxpagedvrp]
                
                for pathdvrp in dictmileoutp['listpathdvrp']:
                    gdat.listpathdvrp.append(pathdvrp)

            pathtargimag = dictmileoutp['pathtarg'] + 'imag/'
            if gdat.typedata == 'toyy' or gdat.typedata == 'mock':
                
                sizefigr = [8., 4.]

                if gdat.typeinst == 'TTL5':
                    liststrglimt = ['', '_limt']
                else:
                    liststrglimt = ['']
                    
                # plot mock relevant (i.e., signal-containing) data with known components
                if n in gdat.indxtargssys:
                    nn = gdat.indxssystarg[n]
                    if gdat.boolcosctrue[n]:
                        nnn = gdat.indxcoscssys[nn]
                    else:
                        nnn = gdat.indxsbinssys[nn]
                    
                    ## light curves
                    dictmodl = dict()

                    for a, strglimt in enumerate(liststrglimt):
                        
                        if strglimt == '_limt' and not np.isfinite(gdat.trueduratrantotl[nn]):
                            continue

                        if a == 0:
                            limtxaxi = None
                        else:
                            limtxaxi = [gdat.trueepoc[nn] - 2. * gdat.trueduratrantotl[nn] / 12. - gdat.timeoffs, gdat.trueepoc[nn] + 2. * gdat.trueduratrantotl[nn] / 12. - gdat.timeoffs]
                        
                            if not np.isfinite(limtxaxi).all():
                                print('gdat.trueepoc[nn]')
                                print(gdat.trueepoc[nn])
                                print('gdat.trueduratrantotl[nn]')
                                print(gdat.trueduratrantotl[nn])
                                raise Exception('')

                        maxm = 1e-100
                        minm = 1e100
                        for p in gdat.indxinst[0]:
                            for y in gdat.indxchun[n][0][p]:
                                
                                if gdat.booldiag:
                                    if gdat.boolcosctrue[n]:
                                        if nnn >= len(gdat.truerflxslen):
                                            print('n, nn, nnn')
                                            print(n, nn, nnn)
                                            print('gdat.indxtarg')
                                            print(gdat.indxtarg)
                                            print('gdat.indxtargssys')
                                            print(gdat.indxtargssys)
                                            print('gdat.boolcosctrue')
                                            print(gdat.boolcosctrue)
                                            raise Exception('')
                                        if len(gdat.truerflxslen[nnn][0][p][y]) == 0:
                                            print('n, nn, nnn')
                                            print(n, nn, nnn)
                                            print('gdat.indxtarg')
                                            print(gdat.indxtarg)
                                            print('gdat.indxtargssys')
                                            print(gdat.indxtargssys)
                                            print('gdat.boolcosctrue')
                                            print(gdat.boolcosctrue)
                                            raise Exception('')

                                maxm = max(maxm, np.amax(np.concatenate([gdat.truerflxtotl[nn][0][p][y], gdat.truerflxelli[nn][0][p][y], gdat.truerflxbeam[nn][0][p][y], \
                                                                                                             gdat.listarrytser['data'][n][0][p][y][:, 1]])))
                                
                                minm = min(minm, np.amin(np.concatenate([gdat.truerflxtotl[nn][0][p][y], gdat.truerflxelli[nn][0][p][y], gdat.truerflxbeam[nn][0][p][y], \
                                                                                                             gdat.listarrytser['data'][n][0][p][y][:, 1]])))
                                
                                if gdat.boolcosctrue[n]:
                                    maxm = max(maxm, np.amax(gdat.truerflxslen[nnn][0][p][y]))
                                    minm = min(minm, np.amin(gdat.truerflxslen[nnn][0][p][y]))
                        
                        limtyaxi = [minm, maxm]
                        for p in gdat.indxinst[0]:
                            for y in gdat.indxchun[n][0][p]:
                                dictmodl['modltotl'] = {'lcur': gdat.truerflxtotl[nn][0][p][y], 'time': gdat.listarrytser['data'][n][0][p][y][:, 0], 'labl': 'Model'}
                                dictmodl['modlelli'] = {'lcur': gdat.truerflxelli[nn][0][p][y], 'time': gdat.listarrytser['data'][n][0][p][y][:, 0], 'labl': 'EV'}
                                dictmodl['modlbeam'] = {'lcur': gdat.truerflxbeam[nn][0][p][y], 'time': gdat.listarrytser['data'][n][0][p][y][:, 0], 'labl': 'Beaming'}
                                if gdat.boolcosctrue[n]:
                                    dictmodl['modlslen'] = {'lcur': gdat.truerflxslen[nnn][0][p][y], 'time': gdat.listarrytser['data'][n][0][p][y][:, 0], 'labl': 'SL'}
                                titlraww = '%s, Tmag=%.3g, $R_*$=%.2g $R_\odot$, $M_*$=%.2g $M_\odot$' % ( \
                                                                                                         gdat.labltarg[n], \
                                                                                                         gdat.truetmag[n], \
                                                                                                         gdat.trueradistar[nn], \
                                                                                                         gdat.truemassstar[nn], \
                                                                                                         )
                                
                                # plot data after injection with injected model components highlighted
                                titlinje = titlraww + '\n$M_c$=%.2g $M_\odot$, $P$=%.3f day, $T_0$=%.3f, $i=%.3g^\circ$, Dur=%.2g hr, $q=%.3g$' % ( \
                                                                                                         gdat.truemasscomp[nn], \
                                                                                                         gdat.trueperi[nn], \
                                                                                                         gdat.trueepoc[nn]-gdat.timeoffs, \
                                                                                                         gdat.trueincl[nn], \
                                                                                                         gdat.trueduratrantotl[nn], \
                                                                                                         gdat.truedcyc[nn], \
                                                                                                        )
                                if gdat.boolcosctrue[n]:
                                    titlinje += ', $A_{SL}$=%.2g ppt' % gdat.trueamplslen[nnn]
                                
                                if gdat.typedata == 'mock':
                                    strgextnraww = '%s_%s_%s%s_raww' % (gdat.typedata, gdat.strgtarg[n], gdat.liststrginst[0][p], strglimt)
                                    pathplot = ephesus.plot_lcur(pathtargimag, titl=titlraww, timeoffs=gdat.timeoffs, limtyaxi=limtyaxi, limtxaxi=limtxaxi, \
                                                                                    timedata=gdat.listarrytser['obsd'][n][0][p][y][:, 0], \
                                                                                    lcurdata=gdat.listarrytser['obsd'][n][0][p][y][:, 1], \
                                                                                    typefileplot=gdat.typefileplot, \
                                                                                    strgextn=strgextnraww, sizefigr=sizefigr, \
                                                                                    )
                                    if gdat.boolplotdvrp:
                                        gdat.listdictdvrp[0].append({'path': pathplot, 'limt':[0., 0.5, 1., 0.2]})
                                    
                                    strgextnover = '%s_%s_%s%s_over' % (gdat.typedata, gdat.strgtarg[n], gdat.liststrginst[0][p], strglimt)
                                    pathplot = ephesus.plot_lcur(pathtargimag, dictmodl=dictmodl, titl=titlinje, timeoffs=gdat.timeoffs, limtyaxi=limtyaxi, \
                                                                                    timedata=gdat.listarrytser['obsd'][n][0][p][y][:, 0], \
                                                                                    lcurdata=gdat.listarrytser['obsd'][n][0][p][y][:, 1], \
                                                                                    typefileplot=gdat.typefileplot, \
                                                                                    strgextn=strgextnover, sizefigr=sizefigr, limtxaxi=limtxaxi)
                                    if gdat.boolplotdvrp:
                                        gdat.listdictdvrp[0].append({'path': pathplot, 'limt':[0., 0.3, 1., 0.2]})
                                
                                    strgextninje = '%s_%s_%s%s_inje' % (gdat.typedata, gdat.strgtarg[n], gdat.liststrginst[0][p], strglimt)
                                    pathplot = ephesus.plot_lcur(pathtargimag, titl=titlinje, timeoffs=gdat.timeoffs, limtyaxi=limtyaxi, \
                                                                                    timedata=gdat.listarrytser['data'][n][0][p][y][:, 0], \
                                                                                    lcurdata=gdat.listarrytser['data'][n][0][p][y][:, 1], \
                                                                                    typefileplot=gdat.typefileplot, \
                                                                                    strgextn=strgextninje, sizefigr=sizefigr, limtxaxi=limtxaxi)
                                    if gdat.boolplotdvrp:
                                        gdat.listdictdvrp[0].append({'path': pathplot, 'limt':[0., 0.1, 1., 0.2]})
                                
                                if gdat.typedata == 'toyy':
                                    strgextninje = '%s_%s_%s%s_inje' % (gdat.typedata, gdat.strgtarg[n], gdat.liststrginst[0][p], strglimt)
                                    
                                    pathplot = ephesus.plot_lcur(pathtargimag, \
                                                                                    timedata=gdat.listarrytser['data'][n][0][p][y][:, 0], \
                                                                                    lcurdata=gdat.listarrytser['data'][n][0][p][y][:, 1], \
                                                                                    titl=titlinje, timeoffs=gdat.timeoffs, limtyaxi=limtyaxi, dictmodl=dictmodl, \
                                                                                    typefileplot=gdat.typefileplot, \
                                                                                    strgextn=strgextninje, sizefigr=sizefigr, limtxaxi=limtxaxi)
                                    if gdat.boolplotdvrp:
                                        gdat.listdictdvrp[0].append({'path': pathplot, 'limt':[0., 0.4, 1, 0.2]})
                                    
                if gdat.boolplotdvrp:
                    # make a simulation summary plot
                    for w in gdat.indxpagedvrpmock:
                        # path of DV report
                        gdat.listpathdvrp[w] = pathtargimag + '%s_dvrp_pag%d.png' % (gdat.strgtarg[n], w)
                        if not os.path.exists(gdat.listpathdvrp[w]):
                            figr = plt.figure(figsize=(8.25, 11.75))
                            numbplot = len(gdat.listdictdvrp[w])
                            indxplot = np.arange(numbplot)
                            for dictdvrp in gdat.listdictdvrp[w]:
                                axis = figr.add_axes(dictdvrp['limt'])
                                axis.imshow(plt.imread(dictdvrp['path']))
                            print('Writing to %s...' % gdat.listpathdvrp[w])
                            plt.axis('off')
                            plt.savefig(gdat.listpathdvrp[w], dpi=600)
                            #plt.subplots_adjust(top=1., bottom=0, left=0, right=1)
                            plt.close()
        
            if gdat.boolplotdvrp:
                for w in gdat.indxpagedvrp:
                    os.system('cp %s %s' % (gdat.listpathdvrp[w], gdat.pathimagpopl))
                
        if gdat.typedata == 'mock':
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
        # population type
        typepopl=None, \

        # list of target TIC IDs
        listticitarg=None, \
        
        # type of experiment
        typeinst=None, \

        # list of MAST keywords
        liststrgmast=None, \
        
        # list of GAIA IDs
        listgaid=None, \

        # type of data: 'toyy', 'mock', or 'obsd'
        typedata='obsd', \
        
        # Boolean flag to turn on multiprocessing
        boolprocmult=False, \
        
        # input dictionary to miletos
        dictmileinpt=dict(), \

        # input dictionary to retr_lcurtess()
        dictlcurtessinpt = dict(), \

        # Boolean flag to make plots
        boolplot=True, \
        
        # Boolean flag to make initial plots
        boolplotinit=False, \
        
        # Boolean flag to make DV reports
        boolplotdvrp=None, \
        
        # Boolean flag to make initial plots
        boolplotmile=None, \
        
        # Boolean flag to turn on diagnostic mode
        booldiag=True, \

        # Boolean flag to force rerun and overwrite previous data and plots 
        boolover=True, \
        
        # verbosity type
        typeverb=0, \

        ):
    
    # construct global object
    gdat = tdpy.gdatstrt()
    
    # copy locals (inputs) to the global object
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
   
    if gdat.boolplotdvrp is None:
        gdat.boolplotdvrp = gdat.boolplot
    
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
    
    if gdat.typedata == 'toyy' and gdat.booltarguser or gdat.typedata != 'toyy' and not gdat.booltarguser and gdat.typepopl is None:
        print('gdat.typedata')
        print(gdat.typedata)
        print('gdat.booltarguser')
        print(gdat.booltarguser)
        raise Exception('')

    if (liststrgmast is not None or listticitarg is not None) and gdat.typepopl is None:
        raise Exception('The type of population, typepopl, must be defined by the user when the target list is provided by the user')
    
    # ticim135: all TIC targets brighter than magnitude 13.5
    # ticim110: all TIC targets brighter than magnitude 11
    # xbin: X-ray binaries
    # tsec: a particular TESS Sector
    print('gdat.typedata')
    print(gdat.typedata)
    print('gdat.typepopl')
    print(gdat.typepopl)
    print('gdat.typeinst')
    print(gdat.typeinst)
    
    # paths
    ## read environment variable
    gdat.pathbase = os.environ['TROIA_DATA_PATH'] + '/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathdatatess = os.environ['TESS_DATA_PATH'] + '/'
    gdat.pathdatalcur = gdat.pathdata + 'lcur/'
    gdat.pathimag = gdat.pathbase + 'imag/'
    gdat.pathtsec = gdat.pathdata + 'logg/tsec/'
    gdat.strgextn = '%s_%s_%s' % (gdat.typedata, gdat.typepopl, gdat.typeinst)
    gdat.pathpopl = gdat.pathbase + gdat.strgextn + '/'
    gdat.pathimagpopl = gdat.pathpopl + 'imag/'
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
    
    if not gdat.booltarguser and gdat.typedata != 'toyy':
        dicttic8 = ephesus.retr_dictpopltic8(typepopl=gdat.typepopl)
        
        size = dicttic8['tici'].size
        size = 10

        indx = np.random.choice(np.arange(dicttic8['tici'].size), replace=False, size=size)
        for name in dicttic8.keys():
            dicttic8[name] = dicttic8[name][indx]
    
    # number of time-series data sets
    gdat.numbdatatser = 2
    gdat.indxdatatser = np.arange(gdat.numbdatatser)

    if gdat.typeinst == 'TTL5':
        gdat.liststrginst = [['TESS', 'TEL5'], []]
    else:
        gdat.liststrginst = [['TESS'], []]

    # maximum number of targets to plot
    gdat.maxmnumbtargplot = 10

    gdat.numbinst = np.empty(gdat.numbdatatser, dtype=int)
    gdat.indxinst = [[] for b in gdat.indxdatatser]
    for b in gdat.indxdatatser:
        gdat.numbinst[b] = len(gdat.liststrginst[b])
        gdat.indxinst[b] = np.arange(gdat.numbinst[b])
    
    # data validation (DV) report
    ## number of pages in the DV report
    if gdat.boolplotdvrp and (gdat.typedata == 'toyy' or gdat.typedata == 'mock'):
        gdat.numbpagedvrpmock = 1
        gdat.indxpagedvrpmock = np.arange(gdat.numbpagedvrpmock)
    
    # determine number of targets
    ## number of targets
    if gdat.booltarguser:
        if gdat.booltargusertici:
            gdat.numbtarg = len(gdat.listticitarg)
        if gdat.booltargusermast:
            gdat.numbtarg = len(gdat.liststrgmast)
        if gdat.booltargusergaid:
            gdat.numbtarg = len(gdat.listgaidtarg)
    elif gdat.typedata != 'toyy':
        gdat.numbtarg = dicttic8['tici'].size
    if gdat.typedata == 'toyy':
        gdat.numbtarg = 30

    if gdat.typedata == 'toyy' or gdat.typedata == 'mock':
        gdat.numbtarg = min(100, gdat.numbtarg)
                
    print('Number of targets: %s' % gdat.numbtarg)
    gdat.indxtarg = np.arange(gdat.numbtarg)
    
    gdat.timeexectarg = 120.
    print('Expected execution time: %g seconds (%.3g days, %.3g weeks for 20M targets)' % (gdat.numbtarg * gdat.timeexectarg, gdat.numbtarg * gdat.timeexectarg / 3600. / 24., \
                                                                                                                        20e6 * gdat.timeexectarg / 3600. / 24. / 7.))
    
    if gdat.listticitarg is None:
        gdat.listticitarg = [[] for k in gdat.indxtarg]
    
    if not gdat.booltarguser and gdat.typedata != 'toyy':
        gdat.listticitarg = dicttic8['tici']
    
    # make initial plots
    if gdat.boolplot and boolplotinit:
        path = gdat.pathimag + 'angleins.%s' % (gdat.typefileplot) 
        if not os.path.exists(path):
            # plot Einstein radius vs lens mass
            figr, axis = plt.subplots(figsize=(6, 4))
            distlens = 1e-7 # Gpc
            distsour = 1e-7 # Gpc
            dictfact = ephesus.retr_factconv()
            peri = 10.#np.logspace(-1., 2., 100)
            masstotl = np.logspace(np.log10(5.), np.log10(200.), 100)
            smax = ephesus.retr_smaxkepl(peri, masstotl) # AU
            distlenssour = 1e-9 * smax / dictfact['pcau'] # Gpc
            angleins = retr_angleins(masstotl, distlenssour, distlens, distsour)
            axis.plot(masstotl, angleins)
            axis.set_xlabel('$M$ [$M_\odot$]')
            axis.set_ylabel(r'$\theta_E$ [arcsec]')
            axis.set_xscale('log')
            axis.set_yscale('log')
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
        # plot amplitude vs. orbital period for three components of the light curve of a COSC
        path = gdat.pathimag + 'amplslen.%s' % gdat.typefileplot
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

                amplbeam = ephesus.retr_deptbeam(arryperi, massstar, listmasscomp[k])
                amplelli = ephesus.retr_deptelli(arryperi, densstar, massstar, listmasscomp[k])
                amplslen = ephesus.retr_amplslen(arryperi, radistar, listmasscomp[k], massstar)
                
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
        

        # plot model light curves for COSCs with different orbital periods
        time = np.arange(0., 20., 2. / 24. / 60.)
        listperi = [3., 6., 9.]
        numbperi = len(listperi)
        indxperi = np.arange(numbperi)
        para = np.empty(6)
        for k in indxperi:
            path = gdat.pathimag + 'fig%d.%s' % (k + 1, gdat.typefileplot)
            if not os.path.exists(path):
                figr, axis = plt.subplots(figsize=(10, 4.5))
                
                dictoutp = ephesus.retr_rflxtranmodl(time, pericomp=[listperi[k]], epoccomp=[0.], radistar=1., massstar=1., masscomp=[10.], inclcomp=[90.], typesyst='cosc')
                rflxmodl = dictoutp['rflx']
                axis.plot(time, rflxmodl, color='k', lw=2, label='Total')
                axis.plot(time, dictoutp['rflxelli'], color='b', ls='--', label='Ellipsoidal variation')
                axis.plot(time, dictoutp['rflxbeam'], color='g', ls='--', label='Beaming')
                axis.plot(time, dictoutp['rflxslen'], color='r', ls='--', label='Self-lensing')
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
            path = gdat.pathimag + 'occ_%s.%s' % (strg, gdat.typefileplot)
            if not os.path.exists(path):
                figr, axis = plt.subplots(figsize=(6, 4.5))
                if k == 0:
                    data = occufielintp
                if k == 1:
                    data = occucoenintp
                c = plt.pcolor(peri, masscomp, data(peri, masscomp), norm=mpl.colors.LogNorm(), cmap ='Greens')#, vmin = z_min, vmax = z_max)
                plt.colorbar(c)
                axis.set_xlabel('Orbital Period [days]')
                axis.set_xlabel('CO mass')
                plt.savefig(path)
                plt.close()
    
        ## plot TESS photometric precision
        path = gdat.pathimag + 'sigmtmag.%s' % (gdat.typefileplot) 
        if not os.path.exists(path):
            dictpoplticim110 = ephesus.retr_dictpopltic8(typepopl='ticim110')
       
            ## interpolate TESS photometric precision
            dictpoplticim110['nois'] = ephesus.retr_noistess(dictpoplticim110['tmag'])

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
        path = gdat.pathimag + 'sigm.%s' % (gdat.typefileplot) 
        if not os.path.exists(path):
            figr, axis = plt.subplots(figsize=(5, 3))
            peri = np.logspace(-1, 2, 100)
            listtmag = [10., 13., 16.]
            listmasscomp = [1., 10., 100.]
            massstar = 1.
            radistar = 1.
            for masscomp in listmasscomp:
                amplslentmag = ephesus.retr_amplslen(peri, radistar, masscomp, massstar)
                axis.plot(peri, amplslentmag, label=r'M = %.3g M$_\odot$' % masscomp)
            for tmag in listtmag:
                noistess = ephesus.retr_noistess(tmag)
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
        if gdat.typedata == 'toyy':
            gdat.strgtarg[n] = 'mock%04d' % n
            gdat.labltarg[n] = 'Mock target %08d' % n
        else:
            if gdat.typepopl[4:12] == 'nomi2min':
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
    gdat.indxchun = [[[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for k in gdat.indxtarg]
    gdat.numbchun = [[np.empty(gdat.numbinst[b], dtype=int) for b in gdat.indxdatatser] for k in gdat.indxtarg]
    if gdat.typedata == 'toyy':
        for n in gdat.indxtarg:
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    gdat.numbchun[n][b][p] = 1
                    gdat.indxchun[n][b][p] = np.arange(gdat.numbchun[n][b][p], dtype=int)
        
        if gdat.typedata == 'mock' or gdat.typedata == 'obsd':
            gdat.listarrytser['obsd'] = [[[[[] for y in gdat.indxchun[k][b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for k in gdat.indxtarg]
    if gdat.typedata == 'toyy':
        gdat.listarrytser['data'] = [[[[[] for y in gdat.indxchun[k][b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for k in gdat.indxtarg]
    
    if gdat.typedata == 'toyy' or gdat.typedata == 'mock':
        # number of relevant types
        gdat.numbtyperele = 2
        gdat.indxtyperele = np.arange(gdat.numbtyperele)
        
    # number of analyses
    gdat.numbtypeposi = 4
    gdat.indxtypeposi = np.arange(gdat.numbtypeposi)
    gdat.boolreletarg = [np.empty(gdat.numbtarg, dtype=bool) for v in gdat.indxtyperele]
    gdat.boolpositarg = [np.empty(gdat.numbtarg, dtype=bool) for u in gdat.indxtypeposi]
    
    if gdat.typedata == 'toyy':
        
        print('Making toy simulated data...')
        
        # mock data setup 
        ## cadence
        gdat.cade = 10. / 60. / 24. # days
        ## minimum time
        gdat.minmtime = 0.
        ## maximum time
        gdat.maxmtime = 30.
        ## time axis for each target
    
        time = np.concatenate((np.arange(0., 27.3 / 2. - 1., gdat.cade), np.arange(27.3 / 2. + 1., 27.3, gdat.cade)))
        numbtime = time.size
        for k in gdat.indxtarg: 
            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    for y in gdat.indxchun[k][0][0]:
                        gdat.listarrytser['data'][k][b][p][y] = np.empty((numbtime, 3))
                        if p == 0:
                            gdat.listarrytser['data'][k][b][p][y][:, 0] = gdat.timeoffs + time
                        if p == 1:
                            gdat.listarrytser['data'][k][b][p][y][:, 0] = gdat.timeoffs + time + 8. / 60. / 24.
    else:
        gdat.listarrytser['obsd'] = [[[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for k in gdat.indxtarg]
            
        for k in gdat.indxtarg:
            
            # get TIC ID
            if gdat.booltarguser:
                if gdat.booltargusertici:
                    dictlcurtessinpt['strgmast'] = None
                    dictlcurtessinpt['ticitarg'] = gdat.listticitarg[k]
                elif gdat.booltargusermast:
                    dictlcurtessinpt['strgmast'] = gdat.liststrgmast[k]
                    dictlcurtessinpt['ticitarg'] = None
            else:
                dictlcurtessinpt['strgmast'] = None
                dictlcurtessinpt['ticitarg'] = gdat.listticitarg[k]
            
            arrylcurtess, gdat.arrytsersapp, gdat.arrytserpdcc, listarrylcurtess, gdat.listarrytsersapp, gdat.listarrytserpdcc, \
                                  gdat.listtsec, gdat.listtcam, gdat.listtccd, listpathdownspoclcur = \
                                  ephesus.retr_lcurtess( \
                                                        **gdat.dictlcurtessinpt, \
                                                       )

            # load data
            if len(arrylcurtess) > 0:
                gdat.listarrytser['obsd'][k][0][0] = listarrylcurtess
            else:
                raise Exception('')

            for b in gdat.indxdatatser:
                for p in gdat.indxinst[b]:
                    gdat.numbchun[k][b][p] = len(listarrylcurtess)
                    gdat.indxchun[k][b][p] = np.arange(gdat.numbchun[k][b][p])
        
    # generate mock data
    if gdat.typedata == 'toyy' or gdat.typedata == 'mock':
        
        # 0: cosc
        # 1: binary star
        # 2: single star
        gdat.probtypetrue = np.array([0.7, 0.25, 0.05])
        gdat.numbtypetrue = gdat.probtypetrue.size
        gdat.indxtypetrue = np.arange(gdat.numbtypetrue)
        gdat.typetruetarg = np.random.choice(gdat.indxtypetrue, size=gdat.numbtarg, p=gdat.probtypetrue)
        
        gdat.indxtypetruetarg = [[] for y in gdat.indxtypetrue]
        for y in gdat.indxtypetrue:
            gdat.indxtypetruetarg[y] = np.where(gdat.typetruetarg == y)[0]
            
        gdat.boolcosctrue = gdat.typetruetarg == 0
        gdat.boolsbintrue = gdat.typetruetarg == 1
        gdat.indxtargcosc = gdat.indxtypetruetarg[0]
        gdat.numbtargcosc = gdat.indxtargcosc.size
        gdat.indxtargsbin = gdat.indxtypetruetarg[1]
        gdat.numbtargsbin = gdat.indxtargsbin.size
        gdat.indxtargssys = np.concatenate((gdat.indxtypetruetarg[0], gdat.indxtypetruetarg[1]))
        gdat.indxtargssys = np.sort(gdat.indxtargssys)
        gdat.numbtargssys = gdat.indxtargssys.size
        print('gdat.numbtarg')
        print(gdat.numbtarg)
        print('gdat.numbtargcosc')
        print(gdat.numbtargcosc)
        print('gdat.numbtargsbin')
        print(gdat.numbtargsbin)
        print('gdat.numbtargssys')
        print(gdat.numbtargssys)
        print('gdat.boolcosctrue')
        print(gdat.boolcosctrue)
        print('gdat.boolsbintrue')
        print(gdat.boolsbintrue)
        
        print('gdat.indxtypetrue')
        print(gdat.indxtypetrue)
        for y in gdat.indxtypetrue:
            print('y')
            print(y)
            print('gdat.indxtypetruetarg[y]')
            print(gdat.indxtypetruetarg[y])
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


        gdat.trueminmincl = 89.
        gdat.truemaaxincl = 90.
        gdat.trueincl = tdpy.icdf_self(np.random.random(gdat.numbtargssys), gdat.trueminmincl, gdat.truemaaxincl)
        
        gdat.trueminmperi = 2.
        gdat.truemaxmperi = 100.
        gdat.trueperi = tdpy.icdf_powr(np.random.random(gdat.numbtargssys), gdat.trueminmperi, 100., 2.)
        
        gdat.trueepoc = np.random.rand(gdat.numbtargssys) * 27.3 + gdat.timeoffs
        
        gdat.trueradicompsbin = tdpy.icdf_powr(np.random.random(gdat.numbtargsbin), 0.1, 100., 2.)

        gdat.truemasscompsbin = tdpy.icdf_powr(np.random.random(gdat.numbtargsbin), 0.1, 100., 2.)
        gdat.truemasscompcosc = tdpy.icdf_powr(np.random.random(gdat.numbtargcosc), 5., 100., 2.)
        gdat.truemasscomp = np.empty(gdat.numbtargssys)
        gdat.truemasscomp[gdat.indxssyscosc] = gdat.truemasscompcosc
        gdat.truemasscomp[gdat.indxssyssbin] = gdat.truemasscompsbin

        gdat.trueduratrantotl = np.empty(gdat.numbtargssys)
        gdat.truedcyc = np.empty(gdat.numbtargssys)
        gdat.trueamplslen = np.empty(gdat.numbtargcosc)
        
    if gdat.typedata == 'mock':
        gdat.trueradistar = dicttic8['radistar']
        gdat.truemassstar = dicttic8['massstar']
        indx = np.where((~np.isfinite(gdat.truemassstar)) | (~np.isfinite(gdat.trueradistar)))[0]
        gdat.trueradistar[indx] = 1.
        gdat.truemassstar[indx] = 1.
        gdat.truetmag = dicttic8['tmag']
    
    if gdat.typedata == 'toyy':
        gdat.trueminmradistar = 0.7
        gdat.truemaxmradistar = 2.
        gdat.trueradistar = np.random.random(gdat.numbtargssys) * (gdat.truemaxmradistar - gdat.trueminmradistar) + gdat.trueminmradistar
        
        gdat.trueminmmassstar = 1.
        gdat.truemaxmmassstar = 2.
        gdat.truemassstar = np.random.random(gdat.numbtargssys) * (gdat.truemaxmmassstar - gdat.trueminmmassstar) + gdat.trueminmmassstar
        
        gdat.trueminmtmag = 8.
        gdat.truemaxmtmag = 14.
        print('temp make this exponential')
        gdat.truetmag = tdpy.icdf_self(np.random.random(gdat.numbtarg), 13., 14.)
    
    if gdat.typedata == 'toyy' or gdat.typedata == 'mock':
        gdat.truerflxtotl = [[[[[] for y in gdat.indxchun[k][b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for n in gdat.indxtargssys]
        gdat.truerflxelli = [[[[[] for y in gdat.indxchun[k][b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for n in gdat.indxtargssys]
        gdat.truerflxbeam = [[[[[] for y in gdat.indxchun[k][b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for n in gdat.indxtargssys]
        gdat.truerflxslen = [[[[[] for y in gdat.indxchun[k][b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for n in gdat.indxtargcosc]
        
        if gdat.typedata == 'mock':
            gdat.listarrytser['data'] = [[[[[] for y in gdat.indxchun[k][b][p]] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for k in gdat.indxtarg]
            for n in gdat.indxtarg:
                for p in gdat.indxinst[0]:
                    for y in gdat.indxchun[n][0][p]:
                        gdat.listarrytser['data'][n][0][p][y] = np.copy(gdat.listarrytser['obsd'][n][0][p][y])
        
        if gdat.typedata == 'obsd':
            gdat.listarrytser['data'] = gdat.listarrytser['obsd']
        
        if gdat.typedata == 'mock':
            for n in gdat.indxtarg:
                for p in gdat.indxinst[0]:
                    for y in gdat.indxchun[n][0][p]:
                        gdat.listarrytser['data'][n][0][p][y] = np.copy(gdat.listarrytser['obsd'][n][0][p][y])
            
        ## stellar systems
        for nn, n in enumerate(gdat.indxtargssys):
            if gdat.boolcosctrue[n]:
                nnn = gdat.indxcoscssys[nn]
            else:
                nnn = gdat.indxsbinssys[nn]
            for p in gdat.indxinst[0]:
                for y in gdat.indxchun[n][0][p]:
                    
                    ## COSCs
                    if gdat.boolcosctrue[n]:
                        dictoutp = ephesus.retr_rflxtranmodl(gdat.listarrytser['data'][n][0][p][y][:, 0], \
                                                                 epoccomp=gdat.trueepoc[None, nn], pericomp=gdat.trueperi[None, nn], inclcomp=gdat.trueincl[None, nn], \
                                                                 radistar=gdat.trueradistar[nn], massstar=gdat.truemassstar[nn], \
                                                                 masscomp=gdat.truemasscompcosc[None, nnn], \
                                                                 typesyst='cosc')
                    ## stellar binaries
                    if gdat.boolsbintrue[n]:
                        dictoutp = ephesus.retr_rflxtranmodl(gdat.listarrytser['data'][n][0][p][y][:, 0], \
                                                                 epoccomp=gdat.trueepoc[None, nn], pericomp=gdat.trueperi[None, nn], inclcomp=gdat.trueincl[None, nn], \
                                                                 radistar=gdat.trueradistar[nn], massstar=gdat.truemassstar[nn], \
                                                                 radicomp=gdat.trueradicompsbin[None, nnn], masscomp=gdat.truemasscompsbin[None, nnn], \
                                                                 typesyst='sbin')
                    
                    gdat.truerflxtotl[nn][0][p][y] = dictoutp['rflx']
                    gdat.truerflxelli[nn][0][p][y] = dictoutp['rflxelli']
                    gdat.truerflxbeam[nn][0][p][y] = dictoutp['rflxbeam']
                    
                    if gdat.boolcosctrue[n]:
                        gdat.truerflxslen[nnn][0][p][y] = dictoutp['rflxslen']
                    
                        if len(gdat.truerflxslen[nnn][0][p][y]) == 0:
                            raise Exception('')

                    if p == 0 and y == 0:
                        gdat.trueduratrantotl[nn] = dictoutp['duratrantotl']
                        gdat.truedcyc[nn] = dictoutp['duratrantotl'] / gdat.trueperi[nn] / 12.
                        if gdat.boolcosctrue[n]:
                            gdat.trueamplslen[nnn] = dictoutp['amplslen']
                    
                    if gdat.typedata == 'toyy':
                        gdat.listarrytser['data'][n][0][p][y][:, 1] = np.copy(gdat.truerflxtotl[nn][0][p][y])
                    if gdat.typedata == 'mock':
                        gdat.listarrytser['data'][n][0][p][y][:, 1] += (gdat.truerflxtotl[nn][0][p][y] - 1.)
        
        if gdat.booldiag:
            for nn, n in enumerate(gdat.indxtargssys):
                for p in gdat.indxinst[0]:
                    for y in gdat.indxchun[n][0][p]:
                        if gdat.listarrytser['data'][n][0][p][y].shape[0] == 1:
                            raise Exception('')
                        if not np.isfinite(gdat.truerflxtotl[nn][0][p][y]).all():
                            raise Exception('')
        
        if gdat.typedata == 'toyy':
            ## single star targets with flat light curves
            for nn, n in enumerate(gdat.indxtypetruetarg[2]):
                for p in gdat.indxinst[0]:
                    for y in gdat.indxchun[n][0][p]:
                        gdat.listarrytser['data'][n][0][p][y][:, 1] = np.ones_like(gdat.listarrytser['data'][n][0][p][y][:, 0])
        
        # relevant targets
        gdat.indxtargcosctran = gdat.indxtargcosc[np.where(np.isfinite(gdat.trueduratrantotl[gdat.indxssyscosc]))]
        gdat.indxtargrele = [[] for v in gdat.indxtyperele]
        # relevants are all COSCs
        gdat.indxtargrele[0] = gdat.indxtargcosc
        # relevants are those transiting COSCs
        gdat.indxtargrele[1] = gdat.indxtargcosctran
        gdat.numbtargrele = np.empty(gdat.numbtyperele, dtype=int)
        
        gdat.indxtargirre = [[] for v in gdat.indxtyperele]
        for v in gdat.indxtyperele:
            gdat.indxtargirre[v] = np.setdiff1d(gdat.indxtarg, gdat.indxtargrele[v])
            gdat.numbtargrele[v] = gdat.indxtargrele[v].size
        
        for v in gdat.indxtyperele:
            print('v')
            print(v)
            print('gdat.indxtargrele[v]')
            print(gdat.indxtargrele[v])
    
    if gdat.typedata == 'toyy':
        # add noise
        stdv = ephesus.retr_noistess(gdat.truetmag) * 1e-3 # [dimensionless]
        if not np.isfinite(stdv).all():
            raise Exception('')

        for n in gdat.indxtarg:
            for p in gdat.indxinst[0]:
                for y in gdat.indxchun[n][0][p]:
                    gdat.listarrytser['data'][n][0][p][y][:, 2] = np.ones(gdat.listarrytser['data'][n][0][p][y].shape[0]) * stdv[n]
                    gdat.listarrytser['data'][n][0][p][y][:, 1] += stdv[n] * np.random.randn(gdat.listarrytser['data'][n][0][p][y].shape[0])
    
    if gdat.typedata == 'obsd':
        gdat.listarrytser['data'] = gdat.listarrytser['obsd']
    
    for k in gdat.indxtarg:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                for y in gdat.indxchun[k][b][p]:
                    for a in range(3):
                        if not np.isfinite(gdat.listarrytser['data'][k][b][p][y][:, a]).all():
                            
                            print('')
                            print('gdat.typedata')
                            print(gdat.typedata)
                            print('kbpya')
                            print(k, b, p, y, a)
                            print('gdat.listarrytser[data][k][b][p][y][:, a]')
                            print(gdat.listarrytser['data'][k][b][p][y][:, a])
                            summgene(gdat.listarrytser['data'][k][b][p][y][:, a])
                            raise Exception('')

    # merge data across sectors
    gdat.arrytser = dict()
    gdat.arrytser['data'] = [[[[] for p in gdat.indxinst[b]] for b in gdat.indxdatatser] for n in gdat.indxtarg]
    for n in gdat.indxtarg:
        for b in gdat.indxdatatser:
            for p in gdat.indxinst[b]:
                gdat.arrytser['data'][n][b][p] = np.concatenate(gdat.listarrytser['data'][n][b][p], axis=0)
                if gdat.arrytser['data'][n][b][p].ndim != 2:
                    print('gdat.arrytser[data][n][b][p]')
                    summgene(gdat.arrytser['data'][n][b][p])
                    raise Exception('')

    gdat.pathlogg = gdat.pathdata + 'logg/'
    
    # output features of miletos
    gdat.dictstat = dict()

    gdat.listnamefeat = ['peripboxprim', 'sdeepboxprim', 'perilspeprim', 'powrlspeprim']
    for namefeat in gdat.listnamefeat:
        gdat.dictstat[namefeat] = np.empty(gdat.numbtarg)
    
    ## fill miletos input dictionary
    ### path to put target data and images
    gdat.dictmileinpt['typeverb'] = gdat.typeverb
    gdat.dictmileinpt['pathbasetarg'] = gdat.pathpopl
    ### Boolean flag to use PDC data
    gdat.dictmileinpt['listtimescalbdtrspln'] = [0.5]
    gdat.dictmileinpt['typefileplot'] = gdat.typefileplot
    gdat.dictmileinpt['boolplotpopl'] = False
    gdat.dictmileinpt['boolover'] = gdat.boolover
    gdat.dictmileinpt['liststrginst'] = gdat.liststrginst
    gdat.dictmileinpt['listtypemodl'] = ['cosc']
    gdat.dictmileinpt['maxmfreqlspe'] = 1. / 0.1 # minimum period is 0.1 day
    #gdat.dictmileinpt['boolsrchsingpuls'] = True
    #### define SDE threshold for periodic box search
    if not 'dictpboxinpt' in gdat.dictmileinpt:
        gdat.dictmileinpt['dictpboxinpt'] = dict()
    
    # inputs to the periodic box search pipeline
    gdat.dictmileinpt['dictpboxinpt']['boolsrchposi'] = True
    gdat.dictmileinpt['dictpboxinpt']['boolprocmult'] = False
    
    if gdat.typedata == 'mock':
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

    if gdat.typedata == 'mock':
        for u in gdat.indxtypeposi:
            for v in gdat.indxtyperele:
                gdat.boolposirele[u][v] = np.array(gdat.boolposirele[u][v], dtype=bool)
                gdat.boolreleposi[u][v] = np.array(gdat.boolreleposi[u][v], dtype=bool)
    gdat.indxtargposi = [[] for u in gdat.indxtypeposi]
    gdat.indxtargnega = [[] for u in gdat.indxtypeposi]
    
    for v in gdat.indxtyperele:
        print('v')
        print(v)
        print('gdat.boolreletarg[v]')
        print(gdat.boolreletarg[v])
    
    # plot distributions
    for u in gdat.indxtypeposi:
        gdat.indxtargposi[u] = np.where(gdat.boolpositarg[u])[0]
        gdat.indxtargnega[u] = np.setdiff1d(gdat.indxtarg, gdat.indxtargposi[u])
        print('gdat.indxtargposi[u]')
        summgene(gdat.indxtargposi[u])
            
        print('gdat.boolpositarg[u]')
        print(gdat.boolpositarg[u])
        listlablpopl = []
        
        #gdat.dictindxtarg['posi'] = gdat.indxtargposi[u]
        
        print('gdat.boolpositarg[u]')
        print(gdat.boolpositarg[u])
        
        for v in gdat.indxtyperele:
            
            #gdat.dictindxtarg['rele'] = gdat.indxtargrele[v]
            
            strguuvv = 'u%dv%d' % (u, v)
            labluuvv = '(u = %d, v = %d)' % (u, v)
            
            gdat.dictindxtarg = dict()
            gdat.dictfeat = dict()

            gdat.dictindxtarg['trpo' + strguuvv] = np.intersect1d(gdat.indxtargposi[u], gdat.indxtargrele[v])
            gdat.dictindxtarg['trne' + strguuvv] = np.intersect1d(gdat.indxtargnega[u], gdat.indxtargirre[v])
            gdat.dictindxtarg['flpo' + strguuvv] = np.intersect1d(gdat.indxtargposi[u], gdat.indxtargirre[v])
            gdat.dictindxtarg['flne' + strguuvv] = np.intersect1d(gdat.indxtargnega[u], gdat.indxtargrele[v])
            
            for strg in ['trpo', 'trne', 'flpo', 'flne']:
                strgkeyy = strg + strguuvv
                if gdat.dictindxtarg[strgkeyy].size > 0:
                    gdat.dictfeat['stat' + strgkeyy] = dict()
                    for namefeat in gdat.listnamefeat:
                        gdat.dictfeat['stat' + strgkeyy][namefeat] = gdat.dictstat[namefeat][gdat.dictindxtarg[strgkeyy]]
            
            listnamepoplcomm = [list(gdat.dictfeat.keys())]
            listlablpoplcomm = [[]]
            for namepoplcomm in listnamepoplcomm[0]:
                if 'trpo' in namepoplcomm:
                    listlablpoplcomm[0].append('TP')
                if 'trne' in namepoplcomm:
                    listlablpoplcomm[0].append('TN')
                if 'flpo' in namepoplcomm:
                    listlablpoplcomm[0].append('FP')
                if 'flne' in namepoplcomm:
                    listlablpoplcomm[0].append('FN')
                if namepoplcomm == 'stattrpo' or namepoplcomm == 'stattrne' or namepoplcomm == 'statflpo' or namepoplcomm == 'statflne':
                    listlablpoplcomm[0][-1] += ' ' + labluuvv
            
            print('listlablpoplcomm')
            print(listlablpoplcomm)
            print('listnamepoplcomm')
            print(listnamepoplcomm)
            
            typeanls = 'bhol_%s' % gdat.strgextn
            pergamon.init( \
                          typeanls=typeanls, \
                          dictpopl=gdat.dictfeat, \
                          listnamepoplcomm=listnamepoplcomm, \
                          listlablpoplcomm=listlablpoplcomm, \
                          listlablpopl=listlablpopl, \
                          pathimag=gdat.pathimagpopl, \
                          pathdata=gdat.pathdatapopl, \
                          boolsortpoplsize=False, \
                         )
                
            if gdat.boolplot and typedata == 'mock':
                listvarbreca = np.vstack([gdat.trueperi, gdat.truemasscomp, gdat.truetmag[gdat.indxtruerele[v]]]).T
                print('listvarbreca')
                summgene(listvarbreca)
                listlablvarbreca = [['$P$', 'day'], ['$M_c$', '$M_\odot$'], ['Tmag', '']]
                liststrgvarbreca = ['trueperi', 'truemasscomp', 'truetmag']
                
                listtemp = []
                for namefeat in gdat.listnamefeat:
                    listtemp.append(gdat.dictstat[namefeat])
                listvarbprec = np.vstack(listtemp).T
                #listvarbprec = np.vstack([gdat.listsdee, gdat.listpowrlspe]).T
                #listlablvarbprec = [['SDE', ''], ['$P_{LS}$', '']]
                liststrgvarbprec = gdat.listnamefeat#['sdee', 'powrlspe']
                listlablvarbprec = tdpy.retr_listlablscalpara(gdat.listnamefeat)
                print('listvarbreca')
                print(listvarbreca)
                print('listvarbprec')
                print(listvarbprec)
                print('gdat.boolreletarg[v]')
                print(gdat.boolreletarg[v])
                print('gdat.boolposirele[u][v]')
                print(gdat.boolposirele[u][v])
                print('gdat.boolreleposi[u][v]')
                print(gdat.boolreleposi[u][v])

                strgextn = '%s' % (gdat.typepopl)
                tdpy.plot_recaprec(gdat.pathimagpopl, strgextn, listvarbreca, listvarbprec, liststrgvarbreca, liststrgvarbprec, \
                                        listlablvarbreca, listlablvarbprec, gdat.boolposirele[u][v], gdat.boolreleposi[u][v])





