import os, sys, datetime, fnmatch

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.interpolate

import json

from tdpy.util import summgene
import tdpy
import ephesus
import miletos


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
    
    dflxslensing = amplslen * np.heaviside(duratran / 2. + timediff, 0.5) * np.heaviside(duratran / 2. - timediff, 0.5)
    
    return dflxslensing


def retr_rflxmodlbhol(time, para):
    
    # parse parameters 
    epoc = para[0]
    peri = para[1]
    radistar = para[2]
    masscomp = para[3]
    massstar = para[4]
    incl = para[5]
    
    # phase
    phas = ((time - epoc) / peri) % 1.
    
    factrsrj, factrjre, factrsre, factmsmj, factmjme, factmsme, factaurs = ephesus.retr_factconv()
    
    ## self-lensing
    ### total mass
    masstotl = masscomp + massstar
    ### semi-major axis
    smax = ephesus.retr_smaxkepl(peri, masstotl)
    ### radius of the star divided by the semi-major axis
    rsma = radistar / smax / factaurs
    ### cosine of the inclination angle
    cosi = np.cos(incl / 180. * np.pi)
    
    ### duration
    duratran = ephesus.retr_duratran(peri, rsma, cosi)
    ### amplitude
    amplslen = ephesus.retr_amplslen(peri, radistar, masscomp, massstar)
    
    #print('radistar')
    #print(radistar)
    #print('masscomp')
    #print(masscomp)
    #print('massstar')
    #print(massstar)
    #print('peri')
    #print(peri)
    #print('smax')
    #print(smax)
    #print('rsma')
    #print(rsma)
    #print('cosi')
    #print(cosi)
    #print('duratran')
    #print(duratran)
    #print('amplslen')
    #print(amplslen)
    
    dflxslen = np.zeros_like(time)
    if np.isfinite(duratran):
        indxtimetran = ephesus.retr_indxtimetran(time, epoc, peri, duratran)
        dflxslen[indxtimetran] += amplslen
    
    ## ellipsoidal variation
    ### density of the star
    print('temp: stellar density calculation is wrong.') 
    # this should change 1.89e-2
    densstar = massstar / radistar**3

    amplelli = ephesus.retr_amplelli(peri, densstar, massstar, masscomp)
    
    ## beaming
    amplbeam = ephesus.retr_amplbeam(peri, massstar, masscomp)
    
    dflxelli = -amplelli * np.cos(4. * np.pi * phas) 
    dflxbeam = amplbeam * np.sin(2. * np.pi * phas)
    
    ## total relative flux
    rflxtotl = 1. + dflxslen + dflxelli + dflxbeam
    
    return rflxtotl, dflxelli + 1., dflxbeam + 1., dflxslen + 1.


def plot_lspe(gdat, n, perisamp, psdn, psdnelli=None, psdnbeam=None, psdnslen=None):
    
    figr, axis = plt.subplots(figsize=(6, 3))
    axis.plot(perisamp, psdn**2, ls='', marker='o', markersize=1, label='Tot', color='black')
    if psdnelli is not None:
        axis.plot(perisamp, psdnelli**2, ls='', marker='o', markersize=1, label='EV')
        axis.plot(perisamp, psdnbeam**2, ls='', marker='o', markersize=1, label='DP')
        axis.plot(perisamp, psdnslen**2, ls='', marker='o', markersize=1, label='SL')
        axis.legend()
    axis.set_xlabel('Power')
    axis.set_xlabel('Period [day]')
    axis.set_xscale('log')
    axis.set_yscale('log')
    plt.tight_layout()
    path = gdat.pathimagpopl + 'psdn_%s.%s' % (gdat.strgextnthis, gdat.plotfiletype)
    plt.savefig(path)
    plt.close()
    

def plot_srch(gdat, n):
    
    gdat.pathplotsrch = gdat.pathtargimag[n] + 'srch_%s_%s.%s' % (gdat.typedata, gdat.strgtarg[n], gdat.plotfiletype)
    
    figr, axis = plt.subplots(figsize=(10, 4.5))
    
    if gdat.typedata == 'obsd':
        axis.text(0.5, 1.25, '%s' % (gdat.labltarg[n]), color='k', transform=axis.transAxes, ha='center')
        
    axis.scatter(gdat.time[n], gdat.rflx[n], color='black', s=1, rasterized=True)
    #axis.set_xlabel('Time [days]')
    axis.set_xticklabels([])
    axis.set_ylabel('Relative Flux')
    
    
    figr, axis = plt.subplots(figsize=(10, 4.5))
    
    if gdat.booltrig[n]:
        axis.plot(gdat.timethis, gdat.postrflxmodl[n, :], color='b', alpha=0.1)
    axis.set_xlabel('Time [days]')
    

    figr, axis = plt.subplots(figsize=(10, 4.5))

    gdat.numbblsq = len(gdat.dictsrchpboxoutp['peri']) 
    gdat.indxblsq = np.arange(gdat.numbblsq)

    for k in gdat.indxblsq:
        axis.axvline(gdat.dictsrchpboxoutp['peri'][k], alpha=0.5, color='b')
        for n in range(2, 10):
            axis.axvline(n*gdat.dictsrchpboxoutp['peri'][k], alpha=0.5, lw=1, linestyle='dashed', color='b')
            axis.axvline(gdat.dictsrchpboxoutp['peri'][k] / n, alpha=0.5, lw=1, linestyle='dashed', color='b')
    axis.set_ylabel(r'SDE')
    axis.set_xlabel('Period [days]')
    axis.plot(gdat.dictsrchpboxoutp['listperi'], gdat.dictsrchpboxoutp['powr'], color='black', lw=0.5)
    #axis.set_xlim([np.amin(gdat.peri), np.amax(gdat.peri)])
        
    
    axis.plot(gdat.phasmodl, gdat.pcurmodl, color='violet')
    
    gdat.fittrflxmodl, gdat.fittdflxelli, gdat.fittdflxbeam, gdat.fittamplslen = retr_rflxmodlbhol(gdat.timethis, gdat.medipara)
    
    mediphas = (gdat.time / gdat.medipara[0] - gdat.medipara[1]) % 1.
    axis.plot(mediphas, gdat.fittrflxmodl, color='b')
    axis.plot(mediphas, gdat.fittdflxelli, color='b', ls='--')
    axis.plot(mediphas, gdat.fittdflxbeam, color='b', ls=':')
    axis.plot(mediphas, gdat.fittamplslen, color='b', ls='-.')
    axis.scatter(gdat.phasdata, gdat.pcurdata, color='black', s=10, alpha=0.5, zorder=2)
    axis.set_xlabel('Phase')
    axis.set_ylabel('Relative flux');
    

    plt.subplots_adjust(hspace=0.35)
    print('Writing to %s...' % gdat.pathplotsrch)
    plt.savefig(gdat.pathplotsrch)
    plt.close()


def init_wrap( \
         listtsec=None, \
         typedata='obsd', \
         **args \
        ):
    
    
    if gdat.boolmultproc:
        listobjtproc = []
    for gdat.tsec in listtsec:
        for gdat.tcam in listtcam:
            for gdat.tccd in listtccd:
                print('Reading files...')
                paththis = '/pdo/qlp-data/sector-%d/ffi/cam%d/ccd%d/LC/' % (tsec, tcam, tccd)
                print('paththis')
                print(paththis)
                liststrgfile = fnmatch.filter(os.listdir(paththis), '*.h5')
                numbrflx = len(liststrgfile)
                print('Number of light curves: %s' % numbrflx)
                liststrgfile = np.array(liststrgfile)
                n = np.arange(numbrflx)
                if boolmultproc:
                    p = multiprocessing.Process(target=work, args=[dictruns])
                    p.start()
                    listobjtproc.append(p)
                else:
                    work(dictruns)
    if boolmultproc:
        for objtproc in listobjtproc:
            objtproc.join() 


def init( \
        # type of data: 'mock' or 'obds'
        typedata='obsd', \
        
        # population type
        typepopl='2minsc17', \

        # Boolean flag to turn on multiprocessing
        boolmultproc=False, \

        # verbosity level
        verbtype=1, \
        
        # Boolean flag to make initial plota
        boolplotinit=True, \

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

    if boolmultproc:
        import multiprocessing
    
    # rvel: High RV targets from Gaia DR2
    # m135: brighter than magnitude 13.5
    # xbin: X-ray binaries
    # tsec: a particular TESS Sector
    print('gdat.typepopl')
    print(gdat.typepopl)
    
    # paths
    ## read PCAT path environment variable
    gdat.pathbase = os.environ['TROIA_DATA_PATH'] + '/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathimag = gdat.pathbase + 'imag/'
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
    gdat.plotfiletype = 'pdf'
    
    gdat.boolanimtmpt = False
    gdat.boolplottmpt = True

    dictpopl = miletos.retr_dictcatltic8(typepopl=gdat.typepopl)
    
    # interpolate TESS photometric precision
    dictpopl['nois'] = ephesus.retr_noistess(dictpopl['tmag'])

    if boolplotinit:
       
        # plot amplitude vs. orbital period for three components of the light curve of a COSC
        path = gdat.pathimag + 'fig0.pdf'
        if not os.path.exists(path):
            
            radistar = 1.
            massstar = 1.
            densstar = 1.41 # [g/cm^3]
            
            listcolr = ['b', 'orange', 'g']
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
                
                axis.plot(arryperi, amplbeam, ls=listlsty[k], color=listcolr[0])
                axis.plot(arryperi, amplelli, ls=listlsty[k], color=listcolr[1])
                axis.plot(arryperi, amplslen, ls=listlsty[k], color=listcolr[2])
            axis.set_xlabel('Orbital Period [days]')
            axis.set_ylabel('Amplitude')
            axis.set_xscale('log')
            axis.set_yscale('log')
            axis.set_xlim([0.3, 30])
            axis.set_ylim([5e-6, 0.5])
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        

        # plot model light curves for COSCs with different orbital periods
        time = np.arange(0., 17.3, 2. / 24. / 60.)
        listperi = [1., 3., 10.]
        numbperi = len(listperi)
        indxperi = np.arange(numbperi)
        para = np.empty(5)
        for k in indxperi:
            path = gdat.pathimag + 'fig%d.pdf' % (k + 1)
            if not os.path.exists(path):
                figr, axis = plt.subplots(figsize=(6, 4.5))
                
                para[0] = 3.
                para[1] = listperi[k]
                para[2] = 1.
                para[3] = 10.
                para[4] = 1.
                rflxmodl, rflxelli, rflxbeam, rflxslen = retr_rflxmodlbhol(time, para)
                
                axis.plot(time, rflxmodl, color='k')
                axis.plot(time, rflxelli, color='b', ls='--')
                axis.plot(time, rflxbeam, color='g', ls='--')
                axis.plot(time, rflxslen, color='r', ls='--')

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
    
        # plot TESS photometric precision
        figr, axis = plt.subplots(figsize=(6, 4))
        axis.plot(dictpopl['tmag'], dictpopl['nois'])
        axis.set_xlabel('Tmag')
        axis.set_ylabel(r'$s$')
        axis.set_yscale('log')
        path = gdat.pathimag + 'sigmtmag.%s' % (gdat.plotfiletype) 
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
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
        path = gdat.pathimag + 'sigm.%s' % (gdat.plotfiletype) 
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
    
    ## input dictionary to miletos
    dictmileinpt = dict()
    
    dictmileinpt['pathbasetarg'] = gdat.pathpopl

    #### Boolean flag to use PDC data
    dictmileinpt['typedataspoc'] = 'PDC'
    dictmileinpt['boolplotprop'] = False
    
    # mock data
    if gdat.typedata == 'mock':
        gdat.cade = 10. / 60. / 24. # days
        gdat.numbtarg = 3
    else:
        ## number of targets
        gdat.numbtarg = dictpopl['rasc'].size
    print('Number of targets: %s' % gdat.numbtarg)
    gdat.indxtarg = np.arange(gdat.numbtarg)
    
    gdat.strgtarg = [[] for n in gdat.indxtarg]
    gdat.labltarg = [[] for n in gdat.indxtarg]
    for n in gdat.indxtarg:
        if gdat.typedata == 'obsd':
            if gdat.typepopl == '2minsc17':
                gdat.strgtarg[n] = 'sc%02d_%12d' % (gdat.tsec, gdat.listticitarg[n])
                gdat.labltarg[n] = 'Sector %02d, TIC %d' % (gdat.tsec, gdat.listticitarg[n])
            if gdat.typepopl == 'xbin':
                gdat.labltarg[n] = gdat.liststrgmast[n]
                gdat.strgtarg[n] = ''.join(gdat.liststrgmast[n].split(' '))
            if gdat.typepopl == 'rvel':
                gdat.labltarg[n] = 'RA=%.4g,DEC=%.4g' % (dictcatlrvel['rasc'][n], dictcatlrvel['decl'][n])
                gdat.strgtarg[n] = 'R%.4gDEC%.4g' % (dictcatlrvel['rasc'][n], dictcatlrvel['decl'][n])
        if gdat.typedata == 'mock':
            gdat.strgtarg[n] = 'mock%04d' % n
            gdat.labltarg[n] = 'Mock target %08d' % n
    
    # get data
    gdat.time = [[] for n in gdat.indxtarg]
    gdat.rflx = [[] for n in gdat.indxtarg]
            
    gdat.boolwritplotover = True

    if typepopl == '2minsc17':
        gdat.tsec = 17
    
    if gdat.tsec is not None:
        print('Sector: %d' % gdat.tsec)

    gdat.numbclasposi = 2
    gdat.boolposi = np.empty((gdat.numbtarg, gdat.numbclasposi), dtype=bool)
    
    if gdat.typedata == 'mock':
        
        print('Making mock data...')

        # mock data setup 
        gdat.minmtime = 0.
        gdat.maxmtime = 27.3
        for nn, n in enumerate(gdat.indxtarg):
            gdat.time[n] = np.concatenate((np.arange(0., 27.3 / 2. - 1., gdat.cade), np.arange(27.3 / 2. + 1., 27.3, gdat.cade)))
        gdat.indxtimelink = np.where(abs(gdat.time[n] - 13.7) < 2.)[0]
    
        gdat.numbtime = gdat.time[n].size
        
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
        
        gdat.trueepoc = np.random.rand(gdat.numbtrueslen) * gdat.maxmtime
        
        gdat.trueminmperi = 10.
        gdat.truemaxmperi = 15.
        gdat.trueperi = np.random.rand(gdat.numbtrueslen) * (gdat.truemaxmperi - gdat.trueminmperi) + gdat.trueminmperi
        
        gdat.trueminmradistar = 0.5
        gdat.truemaxmradistar = 1.
        gdat.trueradistar = np.random.random(gdat.numbtrueslen) * (gdat.truemaxmradistar - gdat.trueminmradistar) + gdat.trueminmradistar
        
        gdat.trueminmmasscomp = 10.
        gdat.truemaxmmasscomp = 60.
        gdat.truemasscomp = np.random.random(gdat.numbtrueslen) * (gdat.truemaxmmasscomp - gdat.trueminmmasscomp) + gdat.trueminmmasscomp
        
        gdat.trueminmmassstar = 1.
        gdat.truemaxmmassstar = 2.
        gdat.truemassstar = np.random.random(gdat.numbtrueslen) * (gdat.truemaxmmassstar - gdat.trueminmmassstar) + gdat.trueminmmassstar
        
        gdat.trueminmincl = 88.
        gdat.truemaaxincl = 90.
        gdat.trueincl = tdpy.icdf_self(np.random.random(gdat.numbtrueslen), gdat.trueminmincl, gdat.truemaaxincl)
        
        gdat.truerflxtotl = np.empty((gdat.numbtime, gdat.numbtrueslen))
        gdat.truedflxelli = np.empty((gdat.numbtime, gdat.numbtrueslen))
        gdat.truedflxbeam = np.empty((gdat.numbtime, gdat.numbtrueslen))
        gdat.truedflxslen = np.empty((gdat.numbtime, gdat.numbtrueslen))
        
        gdat.paratrueslen[:, 0] = gdat.trueepoc
        gdat.paratrueslen[:, 1] = gdat.trueperi
        gdat.paratrueslen[:, 2] = gdat.trueradistar
        gdat.paratrueslen[:, 3] = gdat.truemasscomp
        gdat.paratrueslen[:, 4] = gdat.truemassstar
        gdat.paratrueslen[:, 5] = gdat.trueincl
                
        gdat.trueminmtmag = 8.
        gdat.truemaxmtmag = 16.
        gdat.truetmag = np.random.random(gdat.numbtarg) * (gdat.truemaxmtmag - gdat.trueminmtmag) + gdat.trueminmtmag

        ## flat target
        for nn, n in enumerate(gdat.indxtrueflat):
            gdat.rflx[n] = np.ones_like(gdat.time[n])
            
        ## self-lensing targets
        for nn, n in enumerate(gdat.indxtruerele):
            gdat.truerflxtotl[:, nn], gdat.truedflxelli[:, nn], gdat.truedflxbeam[:, nn], gdat.truedflxslen[:, nn] = retr_rflxmodlbhol( \
                                                                                                                  gdat.time[n], gdat.paratrueslen[nn, :])
            gdat.rflx[n] = np.copy(gdat.truerflxtotl[:, nn])
        
        gdat.truestdvrflx = ephesus.retr_noistess(gdat.truetmag)
        for n in gdat.indxtarg:
            # add noise
            gdat.rflx[n] += gdat.truestdvrflx[n] * np.random.randn(gdat.numbtime)
            
        # histogram
        path = gdat.pathimag + 'truestdvrflx'
        tdpy.plot_hist(path, gdat.truestdvrflx, r'$\sigma$', strgplotextn=gdat.plotfiletype)
        path = gdat.pathimag + 'truetmag'
        tdpy.plot_hist(path, gdat.truetmag, 'Tmag', strgplotextn=gdat.plotfiletype)
            
    pathlogg = gdat.pathdata + 'logg/'
    
    gdat.listsdee = np.empty(gdat.numbtarg)
    gdat.indxtargposi = []
    
    gdat.booltrig = np.zeros(gdat.numbtarg, dtype=bool)
    
    if gdat.typedata == 'mock':
        nn = 0
    
    ## miletos dictionary
    dictmileinpt['boollspe'] = True
    #### SDE threshold for the pipeline to search for periodic boxes
    dictmileinpt['dictsrchpboxoutp'] = {'thrssdee': 0.}
    #dictmileinpt['verbtype'] = 0
    #dictmileinpt['boolsrchsingpuls'] = True
    
    gdat.boolreleposi = []
    gdat.boolposirele = np.empty(gdat.numbtrueslen, dtype=bool)
    gdat.strgextn = '%s_%s' % (gdat.typedata, gdat.typepopl)
    for n in gdat.indxtarg:
        
        gdat.strgextnthis = '%s_%s' % (gdat.typedata, gdat.strgtarg[n])
        pathtcee = pathlogg + '%s_%s.txt' % (gdat.typedata, gdat.strgtarg[n])
        
        if gdat.typedata == 'mock':
            
            if n in gdat.indxtruerele:
                # plot
                dictmodl = dict()
                dictmodl['modltotl'] = {'lcur': gdat.truerflxtotl[:, nn], 'time': gdat.time[n]}
                dictmodl['modlelli'] = {'lcur': gdat.truedflxelli[:, nn], 'time': gdat.time[n]}
                dictmodl['modlbeam'] = {'lcur': gdat.truedflxbeam[:, nn], 'time': gdat.time[n]}
                dictmodl['modlslen'] = {'lcur': gdat.truedflxslen[:, nn], 'time': gdat.time[n]}
                strgextn = '%s_%s' % (gdat.typedata, gdat.strgtarg[n])
                titl = 'P=%.3g day, M=%.3g M$_\odot$, Tmag=%.3g' % (gdat.trueperi[gdat.indxreletrue[n]], \
                                                                gdat.truemasscomp[gdat.indxreletrue[n]], gdat.truetmag[gdat.indxreletrue[n]])
                ephesus.plot_lcur(gdat.pathimagpopl, dictmodl=dictmodl, timedata=gdat.time[n], titl=titl, \
                                                                                lcurdata=gdat.rflx[n], boolwritover=gdat.boolwritplotover, \
                                                                                                                           strgextn=strgextn)
                nn += 1

            gdat.listarry = dict()
            gdat.listarry['raww'] = [[[[]]], []]
            arrytemp = np.empty((gdat.time[n].size, 3))
            arrytemp[:, 0] = gdat.time[n]
            arrytemp[:, 1] = gdat.rflx[n]
            arrytemp[:, 2] = gdat.truestdvrflx[n]
            gdat.listarry['raww'][0][0][0] = arrytemp
            
            rasctarg = None
            decltarg = None
            listarrytser = gdat.listarry
            labltarg = gdat.labltarg[n]
        else:
            if gdat.typepopl == 'tsec':
                arrytemp = np.empty((gdat.time[n].size, 3))
                arrytemp[:, 0] = gdat.time[n]
                arrytemp[:, 1] = gdat.rflx[n]
                arrytemp[:, 2] = gdat.stdvrflx[n]
                
                rasctarg = None
                decltarg = None
                labltarg = None
                listarrytser = gdat.listarry
        
            if gdat.typepopl == 'rvel':
                rasctarg = dictcatlrvel['rasc'][n]
                decltarg = dictcatlrvel['decl'][n]
                labltarg = None
                listarrytser = None
        
        print('Calling miletos.init() to analyze and model the data for the target...')
        # call miletos to analyze data
        dictmileoutp = miletos.init( \
                                  rasctarg=rasctarg, \
                                  decltarg=decltarg, \
                                  labltarg=labltarg, \
                                  
                                  listarrytser=listarrytser, \
                                  typemodl='bhol', \
                                  boolclip=False, \
                                  **dictmileinpt, \
                                 )
        
        if len(dictmileoutp['dictsrchpboxoutp']['sdee']) > 0:
            # taking the fist element, which belongs to the first TCE
            gdat.listsdee[n] = dictmileoutp['dictsrchpboxoutp']['sdee'][0]

            gdat.booltrig[n] = gdat.listsdee[n] >= dictmileinpt['dictsrchpboxoutp']['thrssdee']
        
        print('dictmileoutp[dictsrchpboxoutp][sdee]')
        print(dictmileoutp['dictsrchpboxoutp']['sdee'])
        print('gdat.booltrig[n]')
        print(gdat.booltrig[n])
        if gdat.booltrig[n]:
            
            gdat.boolposi[n, 0] = True
            if gdat.typedata == 'mock':
                
                gdat.boolposirele[gdat.indxreletrue[n]] = True
                
                if gdat.boolreletrue[n]:
                    gdat.boolreleposi.append(True)
                else:
                    gdat.boolreleposi.append(False)
            
        else:
            gdat.boolposi[n, 0] = False
    
    print('gdat.listsdee')
    summgene(gdat.listsdee)

    gdat.boolposirele = np.array(gdat.boolposirele)
    gdat.indxtargposi = np.where(gdat.boolposi[:, 0])[0]

    # plot distributions
    if typedata == 'mock':
        listvarbreca = [gdat.trueperi, gdat.truemasscomp, gdat.truetmag[gdat.indxtruerele]]
        listlablvarbreca = [['$P$', 'day'], ['$M_c$', '$M_\odot$'], ['Tmag', '']]
        liststrgvarbreca = ['trueperi', 'truemasscomp', 'truetmag']
    listvarbprec = [gdat.listsdee]
    listlablvarbprec = [['SNR', '']]
    liststrgvarbprec = ['sdee']
    
    if typedata == 'mock':
        strgextn = '%s' % (gdat.typepopl)
        tdpy.plot_recaprec(gdat.pathimag, strgextn, listvarbreca, listvarbprec, liststrgvarbreca, liststrgvarbprec, \
                                listlablvarbreca, listlablvarbprec, gdat.boolposirele, gdat.boolreleposi)



