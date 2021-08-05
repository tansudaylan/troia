import os, sys, datetime, fnmatch

import matplotlib as mpl
import matplotlib.pyplot as plt

import astroquery

import numpy as np
import pandas as pd
import scipy.interpolate

import json

from tdpy.util import summgene
import tdpy
import ephesus
import miletos

def retr_angleins(masslens, distlens, distsour, distlenssour):
    
    angleins = np.sqrt(masslens / 10**(11.09) * distlenssour / distlens / distsour)
    
    return angleins


def retr_dictderi_effe(para, gdat):
    
    radistar = para[0]
    peri = para[1]
    masscomp = para[2]
    massstar = para[3]
    masstotl = massstar + masscomp

    amplslenmodl = ephesus.retr_amplslen(peri, radistar, masscomp, massstar)
    duratranmodl = ephesus.retr_duratran(peri, radistar, masscomp, massstar, incl)
    smax = retr_smaxkepl(peri, masstotl) * 215. # [R_S]
    radischw = 4.24e-6 * masscomp # [R_S]

    dictvarbderi = None

    dictparaderi = dict()
    dictparaderi['amplslenmodl'] = np.array([amplslenmodl])
    dictparaderi['duratranmodl'] = np.array([duratranmodl])
    dictparaderi['smaxmodl'] = np.array([smax])
    dictparaderi['radischw'] = np.array([radischw])

    return dictparaderi, dictvarbderi
    

def retr_dflxslensing(time, epocslen, amplslen, duratran):
    
    timediff = time - epocslen
    
    dflxslensing = 1e-3 * amplslen * np.heaviside(duratran / 48. + timediff, 0.5) * np.heaviside(duratran / 48. - timediff, 0.5)
    
    return dflxslensing


def retr_rflxmodlbhol( \
                      # time axis
                      time, \
                      # parameter vector (see below for details)
                      para, \
                     ):
    
    # parse parameters 
    ## epoch of the orbit
    epoc = para[0]
    ## orbital period in days
    peri = para[1]
    ## radius of the star in Solar radius
    radistar = para[2]
    ## mass of the companion in Solar mass
    masscomp = para[3]
    ## mass of the star in Solar mass
    massstar = para[4]
    ## inclination of the orbit in degrees
    incl = para[5]
    
    # phase
    phas = ((time - epoc) / peri) % 1.
    
    # conversion constants
    dictfact = ephesus.retr_factconv()
    
    ## total mass
    masstotl = masscomp + massstar
    ## semi-major axis
    smax = ephesus.retr_smaxkepl(peri, masstotl)
    ## radius of the star divided by the semi-major axis
    rsma = radistar / smax / dictfact['aurs']
    ## cosine of the inclination angle
    cosi = np.cos(incl / 180. * np.pi)
    
    ## self-lensing
    ### duration
    duratran = ephesus.retr_duratran(peri, rsma, cosi)
    ### amplitude
    amplslen = ephesus.retr_amplslen(peri, radistar, masscomp, massstar)
    ### signal
    dflxslen = np.zeros_like(time)
    if np.isfinite(duratran):
        indxtimetran = ephesus.retr_indxtimetran(time, epoc, peri, duratran)
        dflxslen[indxtimetran] += 1e-3 * amplslen
    
    ## ellipsoidal variation
    ### density of the star in g/cm3
    densstar = 1.41 * massstar / radistar**3
    ### amplitude
    amplelli = ephesus.retr_amplelli(peri, densstar, massstar, masscomp)
    ### signal
    dflxelli = -1e-3 * amplelli * np.cos(4. * np.pi * phas) 
    
    ## beaming
    amplbeam = ephesus.retr_amplbeam(peri, massstar, masscomp)
    ### signal
    dflxbeam = 1e-3 * amplbeam * np.sin(2. * np.pi * phas)
    
    ## total relative flux
    rflxtotl = 1. + dflxslen + dflxelli + dflxbeam
    
    dictoutpbhol = dict()
    dictoutpbhol['amplslen'] = amplslen
    dictoutpbhol['duratran'] = duratran
    dictoutpbhol['rflxelli'] = dflxelli + 1.
    dictoutpbhol['rflxbeam'] = dflxbeam + 1.
    dictoutpbhol['rflxslen'] = dflxslen + 1.
    
    return rflxtotl, dictoutpbhol


def mile_work(gdat, p):
    
    for n in gdat.listindxtarg[p]:
        
        if len(gdat.time[n]) == 0:
            print('No data on %s! Skipping...' % gdat.labltarg[n])
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
            listarrytser['raww'] = [[[[]]], []]
            arrytemp = np.empty((gdat.time[n].size, 3))
            arrytemp[:, 0] = gdat.time[n]
            arrytemp[:, 1] = gdat.rflx[n]
            arrytemp[:, 2] = gdat.stdvrflx[n]
            listarrytser['raww'][0][0][0] = arrytemp
            
            rasctarg = None
            decltarg = None
            strgmast = None
            labltarg = gdat.labltarg[n]
            strgtarg = gdat.strgtarg[n]
        
        print('Calling miletos.init() to analyze and model the data for the target...')
        # call miletos to analyze data
        dictmileoutp = miletos.init( \
                                  rasctarg=rasctarg, \
                                  decltarg=decltarg, \
                                  strgtarg=strgtarg, \
                                  labltarg=labltarg, \
                                  strgmast=strgmast, \
                                  
                                  listarrytser=listarrytser, \
                                  typemodl='bhol', \
                                  **gdat.dictmileinpt, \
                                 )
        
        if len(dictmileoutp['dictsrchpboxoutp']['sdee']) > 0:
            # taking the fist element, which belongs to the first TCE
            gdat.listpowrlspe[n] = dictmileoutp['powrlspe']
            gdat.listsdee[n] = dictmileoutp['dictsrchpboxoutp']['sdee'][0]
            gdat.booltrig[n] = gdat.listsdee[n] >= gdat.dictmileinpt['dictsrchpboxoutp']['thrssdee']
        
        if gdat.typedata == 'mock':
            # plot mock relevant (i.e., signal-containing) data with known components
            if n in gdat.indxtruerele:
                nn = gdat.indxreletrue[n]
                ## light curves
                dictmodl = dict()
                maxm = np.amax(np.concatenate([gdat.truerflxtotl[nn], gdat.truerflxelli[nn], gdat.truerflxbeam[nn], gdat.truerflxslen[nn], gdat.rflxobsd[n], gdat.rflx[n]]))
                minm = np.amin(np.concatenate([gdat.truerflxtotl[nn], gdat.truerflxelli[nn], gdat.truerflxbeam[nn], gdat.truerflxslen[nn], gdat.rflxobsd[n], gdat.rflx[n]]))
                limtyaxi = [minm, maxm]
                dictmodl['modltotl'] = {'lcur': gdat.truerflxtotl[nn], 'time': gdat.time[n], 'labl': 'Model'}
                dictmodl['modlelli'] = {'lcur': gdat.truerflxelli[nn], 'time': gdat.time[n], 'labl': 'EV'}
                dictmodl['modlbeam'] = {'lcur': gdat.truerflxbeam[nn], 'time': gdat.time[n], 'labl': 'Beaming'}
                dictmodl['modlslen'] = {'lcur': gdat.truerflxslen[nn], 'time': gdat.time[n], 'labl': 'SL'}
                titlraww = '%s, Tmag=%.3g, $R_*$=%.2g $R_\odot$, $M_*$=%.2g $M_\odot$' % ( \
                                                                                         gdat.labltarg[n], \
                                                                                         gdat.truetmag[n], \
                                                                                         gdat.trueradistar[n], \
                                                                                         gdat.truemassstar[n], \
                                                                                         )
                
                # plot data after injection with injected model components highlighted
                titlinje = titlraww + '\n$M_c$=%.2g $M_\odot$, P=%.3g day, $i=%.3g^\circ$, $A_{SL}$=%.2g ppt, Dur=%.2g hr' % ( \
                                                                                         gdat.truemasscomp[nn], \
                                                                                         gdat.trueperi[nn], \
                                                                                         gdat.trueincl[nn], \
                                                                                         gdat.trueamplslen[nn], \
                                                                                         gdat.trueduratran[nn], \
                                                                                        )
                
                strgextn = '%s_%s_raww' % (gdat.typedata, gdat.strgtarg[n])
                pathplot = ephesus.plot_lcur(gdat.pathimagpopl, timedata=gdat.time[n], titl=titlraww, timeoffs=gdat.timeoffs, limtyaxi=limtyaxi, \
                                                                            lcurdata=gdat.rflxobsd[n], boolwritover=gdat.boolwritplotover, strgextn=strgextn)
                os.system('cp %s %s' % (pathplot, dictmileoutp['pathtarg'] + 'imag/'))
                
                strgextn = '%s_%s_over' % (gdat.typedata, gdat.strgtarg[n])
                pathplot = ephesus.plot_lcur(gdat.pathimagpopl, dictmodl=dictmodl, timedata=gdat.time[n], titl=titlinje, timeoffs=gdat.timeoffs, limtyaxi=limtyaxi, \
                                                                            lcurdata=gdat.rflxobsd[n], boolwritover=gdat.boolwritplotover, strgextn=strgextn)
                os.system('cp %s %s' % (pathplot, dictmileoutp['pathtarg'] + 'imag/'))
                
                strgextn = '%s_%s_inje' % (gdat.typedata, gdat.strgtarg[n])
                pathplot = ephesus.plot_lcur(gdat.pathimagpopl, timedata=gdat.time[n], titl=titlinje, timeoffs=gdat.timeoffs, limtyaxi=limtyaxi, \
                                                                            lcurdata=gdat.rflx[n], boolwritover=gdat.boolwritplotover, strgextn=strgextn)
                os.system('cp %s %s' % (pathplot, dictmileoutp['pathtarg'] + 'imag/'))
                
        print('gdat.booltrig[n]')
        print(gdat.booltrig[n])
        if gdat.booltrig[n]:
            
            gdat.boolposi[n, 0] = True
            if gdat.typedata == 'mock':
                
                if gdat.indxreletrue[n] != -1:
                    gdat.boolposirele[gdat.indxreletrue[n]] = True
                
                if gdat.boolreletrue[n]:
                    gdat.boolreleposi.append(True)
                else:
                    gdat.boolreleposi.append(False)
            
        else:
            gdat.boolposi[n, 0] = False
    
    return gdat


def init( \
        # population type
        typepopl=None, \

        # list of target TIC IDs
        listticitarg=None, \

        # list of MAST keywords
        liststrgmast=None, \
        
        # list of GAIA IDs
        listgaid=None, \

        # type of data: 'mock', 'toyy', or 'obds'
        typedata='obsd', \
        
        # Boolean flag to turn on multiprocessing
        #boolmultproc=True, \
        boolmultproc=False, \

        # verbosity level
        verbtype=1, \
        
        # Boolean flag to make initial plots
        boolplotinit=False, \

        **args
        ):
    
    # construct global object
    gdat = tdpy.gdatstrt()
    
    # copy unnamed inputs to the global object
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # copy named arguments to the global object
    for strg, valu in args.items():
        setattr(gdat, strg, valu)

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
   
    print('troia initialized at %s...' % gdat.strgtimestmp)
    
    if gdat.liststrgmast is not None and gdat.listticitarg is not None:
        raise Exception('liststrgmast and listticitarg cannot be defined simultaneously.')
    
    gdat.booltargusermast = gdat.liststrgmast is not None
    gdat.booltargusertici = gdat.listticitarg is not None
    gdat.booltargusergaid = gdat.listgaid is not None
    gdat.booltarguser = gdat.booltargusertici or gdat.booltargusermast or gdat.booltargusergaid
    
    if (liststrgmast is not None or listticitarg is not None) and gdat.typepopl is None:
        raise Exception('The type of population, typepopl, must be defined by the user when the target list is provided by the user')
    
    # m135: brighter than magnitude 13.5
    # xbin: X-ray binaries
    # tsec: a particular TESS Sector
    print('gdat.typepopl')
    print(gdat.typepopl)
    
    # paths
    ## read environment variable
    gdat.pathbase = os.environ['TROIA_DATA_PATH'] + '/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathdatatess = os.environ['TESS_DATA_PATH'] + '/'
    gdat.pathdatalcur = gdat.pathdata + 'lcur/'
    gdat.pathimag = gdat.pathbase + 'imag/'
    gdat.pathtsec = gdat.pathdata + 'logg/tsec/'
    gdat.pathpopl = gdat.pathbase + gdat.typepopl + '_' + gdat.typedata + '/'
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
    gdat.typefileplot = 'pdf'
    gdat.timeoffs = 2457000
    gdat.boolanimtmpt = False
    gdat.boolplottmpt = True
    
    if not gdat.booltarguser:
        dictpopl = miletos.retr_dictcatltic8(typepopl=gdat.typepopl)
        
        indx = np.random.choice(np.arange(dictpopl['tici'].size), replace=False, size=dictpopl['tici'].size)
        for name in dictpopl.keys():
            dictpopl[name] = dictpopl[name][indx]
    
    # mock data
    if gdat.typedata == 'mock':
        gdat.cade = 10. / 60. / 24. # days
        gdat.numbtarg = 100
    else:
        ## number of targets
        if gdat.booltarguser:
            if gdat.booltargusertici:
                gdat.numbtarg = len(gdat.listticitarg)
            if gdat.booltargusermast:
                gdat.numbtarg = len(gdat.liststrgmast)
            if gdat.booltargusergaid:
                gdat.numbtarg = len(gdat.listgaidtarg)
        else:
            gdat.numbtarg = dicpopl['ID'].size
                
    print('Number of targets: %s' % gdat.numbtarg)
    gdat.indxtarg = np.arange(gdat.numbtarg)
    
    if gdat.booltarguser:
        if gdat.booltargusermast:
            for k in gdat.indxtarg:
                listdictcatl = astroquery.mast.Catalogs.query_object(gdat.liststrgmast[k], catalog='TIC', radius='40s')
                if listdictcatl[0]['dstArcSec'] < 0.1:
                    gdat.listticitarg = listdictcatl[0]['ID']
                else:
                    print('Warning! No match to the provided MAST keyword: ' % gdat.liststrgmast[k])
        #if gdat.booltargusertici:
        #    dictpopl = miletos.xmat_tici(gdat.listticitarg)
        #    print('dictpopl')
        #    print(dictpopl)
    else:
        gdat.listticitarg = dictpopl['tici']


    if boolplotinit:
        # plot Einstein radius vs lens mass
        figr, axis = plt.subplots(figsize=(6, 4))
        distlens = 1e-7 # Gpc
        distsour = 1e-7 # Gpc
        dictfact = ephesus.retr_factconv()
        peri = 10.#np.logspace(-1., 2., 100)
        masstotl = np.logspace(np.log10(5.), np.log10(200.), 100)
        smax = ephesus.retr_smaxkepl(peri, masstotl) # AU
        distlenssour = 1e-9 * smax / dictfact['pcau'] # Gpc
        angleins = retr_angleins(masstotl, distlens, distsour, distlenssour)
        print('angleins')
        print(angleins)
        axis.plot(masstotl, angleins)
        axis.set_xlabel('$M$ [$M_\odot$]')
        axis.set_ylabel(r'$\theta_E$ [arcsec]')
        axis.set_xscale('log')
        axis.set_yscale('log')
        path = gdat.pathimag + 'angleins.%s' % (gdat.typefileplot) 
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        # plot amplitude vs. orbital period for three components of the light curve of a COSC
        path = gdat.pathimag + 'amplslen.pdf'
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

            amplbeam = ephesus.retr_amplbeam(arryperi, massstar, listmasscomp[k])
            amplelli = ephesus.retr_amplelli(arryperi, densstar, massstar, listmasscomp[k])
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
            path = gdat.pathimag + 'fig%d.pdf' % (k + 1)
            figr, axis = plt.subplots(figsize=(10, 4.5))
            
            para[0] = 3.
            para[1] = listperi[k]
            para[2] = 1.
            para[3] = 10.
            para[4] = 1.
            para[5] = 90.
            rflxmodl, dictoutpbhol = retr_rflxmodlbhol(time, para)
            
            axis.plot(time, rflxmodl, color='k', lw=2, label='Total')
            axis.plot(time, dictoutpbhol['rflxelli'], color='b', ls='--', label='Ellipsoidal variation')
            axis.plot(time, dictoutpbhol['rflxbeam'], color='g', ls='--', label='Beaming')
            axis.plot(time, dictoutpbhol['rflxslen'], color='r', ls='--', label='Self-lensing')
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
            path = gdat.pathimag + 'occ_%s.pdf' % strg
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
    
        #dictpoplm135 = miletos.retr_dictcatltic8(typepopl='m135nomi')
       
        ## interpolate TESS photometric precision
        #dictpoplm135['nois'] = ephesus.retr_noistess(dictpoplm135['tmag'])

        ## plot TESS photometric precision
        #figr, axis = plt.subplots(figsize=(6, 4))
        #axis.plot(dictpoplm135['tmag'], dictpoplm135['nois'])
        #axis.set_xlabel('Tmag')
        #axis.set_ylabel(r'$s$')
        #axis.set_yscale('log')
        #path = gdat.pathimag + 'sigmtmag.%s' % (gdat.typefileplot) 
        #print('Writing to %s...' % path)
        #plt.savefig(path)
        #plt.close()
        
        # plot SNR
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
        path = gdat.pathimag + 'sigm.%s' % (gdat.typefileplot) 
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
    
    ## input dictionary to miletos
    gdat.dictmileinpt = dict()
    
    gdat.dictmileinpt['pathbasetarg'] = gdat.pathpopl

    #### Boolean flag to use PDC data
    gdat.dictmileinpt['timescalbdtrspln'] = 0.5
    gdat.dictmileinpt['boolplotprop'] = False
    gdat.dictmileinpt['boolplotprio'] = False
    
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
            if gdat.typepopl == 'listmast':
                gdat.labltarg[n] = gdat.liststrgmast[n]
                gdat.strgtarg[n] = ''.join(gdat.liststrgmast[n].split(' '))
            if gdat.booltargusergaid:
                gdat.labltarg[n] = 'GID=' % (dictcatlrvel['rasc'][n], dictcatlrvel['decl'][n])
                gdat.strgtarg[n] = 'R%.4gDEC%.4g' % (dictcatlrvel['rasc'][n], dictcatlrvel['decl'][n])
    
    # real data
    gdat.timeobsd = [[] for n in gdat.indxtarg]
    gdat.rflxobsd = [[] for n in gdat.indxtarg]
    gdat.stdvrflxobsd = [[] for n in gdat.indxtarg]
            
    # data to be fed into the analysis
    gdat.time = [[] for n in gdat.indxtarg]
    gdat.rflx = [[] for n in gdat.indxtarg]
    gdat.stdvrflx = [[] for n in gdat.indxtarg]
            
    gdat.boolwritplotover = True

    gdat.numbclasposi = 2
    gdat.boolposi = np.empty((gdat.numbtarg, gdat.numbclasposi), dtype=bool)
    
    if gdat.typedata == 'toyy':
        
        print('Making mock data...')

        # mock data setup 
        gdat.minmtime = 0.
        gdat.maxmtime = 27.3
        for nn, n in enumerate(gdat.indxtarg):
            gdat.time[n] = np.concatenate((np.arange(0., 27.3 / 2. - 1., gdat.cade), np.arange(27.3 / 2. + 1., 27.3, gdat.cade)))
        gdat.indxtimelink = np.where(abs(gdat.time[n] - 13.7) < 2.)[0]
    
    else:
        for nn, n in enumerate(gdat.indxtarg):
            
            # get TIC ID
            if gdat.booltargusertici:
                tici = gdat.listticitarg[n]
            else:
                tici = int(gdat.liststrgmast[n][4:])
            
            # get the list of TESS sectors and file paths
            listtsec, listpath = ephesus.retr_tsectici(tici)
            # get data
            listarrylcur = [[] for k in range(len(listpath))]
            listtsec = np.empty(len(listpath))
            listtcam = np.empty(len(listpath))
            listtccd = np.empty(len(listpath))
            for k in range(len(listpath)):
                listarrylcur[k], indxtimequalgood, indxtimenanngood, listtsec[k], listtcam[k], listtccd[k] = ephesus.read_tesskplr_file(listpath[k], typeinst='tess', \
                                                                                                                                                            strgtype='PDCSAP_FLUX')
            # load data
            if len(listarrylcur) > 0:
                arrylcurtess = np.concatenate(listarrylcur)
                gdat.timeobsd[n] = arrylcurtess[:, 0]
                gdat.rflxobsd[n] = arrylcurtess[:, 1]
                gdat.stdvrflxobsd[n] = arrylcurtess[:, 2]
                
    if gdat.typedata == 'toyy':
        gdat.numbtime = gdat.time[n].size
        
    if gdat.typedata == 'mock':
        
        gdat.numbclastrue = 2
        gdat.indxclastrue = np.arange(gdat.numbclastrue)
        gdat.clastrue = np.random.choice(gdat.indxclastrue, size=gdat.numbtarg, p=[0, 1])
        gdat.indxtrueflat = np.where(gdat.clastrue == 0)[0]
        gdat.boolreletrue = gdat.clastrue == 1
        gdat.indxtruerele = np.where(gdat.boolreletrue)[0]
        gdat.numbtrueslen = gdat.indxtruerele.size
        
        gdat.indxreletrue = np.zeros(gdat.numbtarg, dtype=int) - 1
        cntr = 0
        for k in range(gdat.numbtarg):
            if gdat.boolreletrue[k]:
                gdat.indxreletrue[k] = cntr
                cntr += 1

        # generate mock data
        gdat.numbparatrueslen = 6
        gdat.paratrueslen = np.empty((gdat.numbtrueslen, gdat.numbparatrueslen))
        
        gdat.trueminmperi = 2.
        gdat.truemaxmperi = 100.
        gdat.trueperi = tdpy.icdf_powr(np.random.random(gdat.numbtrueslen), gdat.trueminmperi, 100., 2.)
        
        gdat.trueepoc = np.random.rand(gdat.numbtrueslen) * gdat.truemaxmperi
        
        gdat.trueduratran = np.empty(gdat.numbtrueslen)
        gdat.trueamplslen = np.empty(gdat.numbtrueslen)
        
        gdat.trueminmradistar = 0.5
        gdat.truemaxmradistar = 1.
        gdat.trueradistar = np.random.random(gdat.numbtrueslen) * (gdat.truemaxmradistar - gdat.trueminmradistar) + gdat.trueminmradistar
        
        gdat.truemasscomp = tdpy.icdf_powr(np.random.random(gdat.numbtrueslen), 5., 100., 2.)
        
        gdat.trueminmmassstar = 1.
        gdat.truemaxmmassstar = 2.
        gdat.truemassstar = np.random.random(gdat.numbtrueslen) * (gdat.truemaxmmassstar - gdat.trueminmmassstar) + gdat.trueminmmassstar
        
        gdat.trueminmincl = 89.
        gdat.truemaaxincl = 90.
        gdat.trueincl = tdpy.icdf_self(np.random.random(gdat.numbtrueslen), gdat.trueminmincl, gdat.truemaaxincl)
        
        gdat.truerflxtotl = [[] for n in gdat.indxtruerele]
        gdat.truerflxelli = [[] for n in gdat.indxtruerele]
        gdat.truerflxbeam = [[] for n in gdat.indxtruerele]
        gdat.truerflxslen = [[] for n in gdat.indxtruerele]
        
        gdat.paratrueslen[:, 0] = gdat.trueepoc
        gdat.paratrueslen[:, 1] = gdat.trueperi
        gdat.paratrueslen[:, 2] = gdat.trueradistar
        gdat.paratrueslen[:, 3] = gdat.truemasscomp
        gdat.paratrueslen[:, 4] = gdat.truemassstar
        gdat.paratrueslen[:, 5] = gdat.trueincl
                
        gdat.trueminmtmag = 8.
        gdat.truemaxmtmag = 16.
        gdat.truetmag = np.random.random(gdat.numbtarg) * (gdat.truemaxmtmag - gdat.trueminmtmag) + gdat.trueminmtmag

    if gdat.typedata == 'toyy' or gdat.typedata == 'mock':
        if gdat.typedata == 'toyy':
            ## flat target
            for nn, n in enumerate(gdat.indxtrueflat):
                gdat.rflx[n] = np.ones_like(gdat.time[n])
            
        ## self-lensing targets
        for nn, n in enumerate(gdat.indxtruerele):
            gdat.truerflxtotl[nn], dictoutpbhol = retr_rflxmodlbhol(gdat.timeobsd[n], gdat.paratrueslen[nn, :])
            gdat.truerflxelli[nn] = dictoutpbhol['rflxelli']
            gdat.truerflxbeam[nn] = dictoutpbhol['rflxbeam']
            gdat.truerflxslen[nn] = dictoutpbhol['rflxslen']
            gdat.trueduratran[nn] = dictoutpbhol['duratran']
            gdat.trueamplslen[nn] = dictoutpbhol['amplslen']
            if gdat.typedata == 'mock':
                gdat.rflx[n] = gdat.rflxobsd[n] + gdat.truerflxtotl[nn] - 1.
            if gdat.typedata == 'toyy':
                gdat.rflx[n] = np.copy(gdat.truerflxtotl[nn])
                stdvrflxscal = ephesus.retr_noistess(gdat.truetmag) * 1e-6 # [dimensionless]
                for n in gdat.indxtarg:
                    # add noise
                    gdat.rflx[n] += stdvrflxscal * np.random.randn(gdat.rflx[n].size)
    
    if gdat.typedata == 'obsd':
        gdat.rflx = gdat.rflxobsd
    if gdat.typedata == 'mock' or gdat.typedata == 'obsd':
        gdat.time = gdat.timeobsd
        gdat.stdvrflx = gdat.stdvrflxobsd

    gdat.pathlogg = gdat.pathdata + 'logg/'
    
    gdat.listsdee = np.empty(gdat.numbtarg)
    gdat.listpowrlspe = np.empty(gdat.numbtarg)
    gdat.indxtargposi = []
    
    gdat.booltrig = np.zeros(gdat.numbtarg, dtype=bool)
    
    ## fill miletos input dictionary
    gdat.dictmileinpt['boollspe'] = True
    gdat.dictmileinpt['listtypeobjt'] = ['bhol']
    gdat.dictmileinpt['maxmfreqlspe'] = 1. / 0.1 # minimum period is 0.1 day
    #### SDE threshold for the pipeline to search for periodic boxes
    gdat.dictmileinpt['dictsrchpboxoutp'] = {'thrssdee': 0.}
    #gdat.dictmileinpt['verbtype'] = 0
    #gdat.dictmileinpt['boolsrchsingpuls'] = True
    
    if gdat.typedata == 'mock':
        gdat.boolreleposi = []
        gdat.boolposirele = np.zeros(gdat.numbtrueslen, dtype=bool)
    gdat.strgextn = '%s_%s' % (gdat.typedata, gdat.typepopl)
    
    
    if boolmultproc:
        import multiprocessing
        from functools import partial
        #multiprocessing.set_start_method('spawn')

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

    print('gdat.listsdee')
    summgene(gdat.listsdee)
    
    if gdat.typedata == 'mock':
        gdat.boolposirele = np.array(gdat.boolposirele)
    gdat.indxtargposi = np.where(gdat.boolposi[:, 0])[0]

    # plot distributions
    if typedata == 'mock':
        listvarbreca = np.vstack([gdat.trueperi, gdat.truemasscomp, gdat.truetmag[gdat.indxtruerele]]).T
        print('listvarbreca')
        summgene(listvarbreca)
        listlablvarbreca = [['$P$', 'day'], ['$M_c$', '$M_\odot$'], ['Tmag', '']]
        liststrgvarbreca = ['trueperi', 'truemasscomp', 'truetmag']
    
        listvarbprec = np.vstack([gdat.listsdee, gdat.listpowrlspe]).T
        listlablvarbprec = [['SDE', ''], ['$P_{LS}$', '']]
        liststrgvarbprec = ['sdee', 'powrlspe']
    
        print('listvarbreca')
        print(listvarbreca)
        print('listvarbprec')
        print(listvarbprec)
        print('gdat.boolposirele')
        print(gdat.boolposirele)
        print('gdat.boolreleposi')
        print(gdat.boolreleposi)

        strgextn = '%s' % (gdat.typepopl)
        tdpy.plot_recaprec(gdat.pathimagpopl, strgextn, listvarbreca, listvarbprec, liststrgvarbreca, liststrgvarbprec, \
                                listlablvarbreca, listlablvarbprec, gdat.boolposirele, gdat.boolreleposi)


