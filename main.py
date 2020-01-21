import numpy
import batman
import matplotlib.pyplot as plt
import os
from transitleastsquares import transitleastsquares
import emcee

#from lion import main as lionmain

from tdpy.util import summgene
import tdpy.util
import tdpy.mcmc

import numpy as np

import h5py


import tesstarg.util
#import tcat.main

import time as timemodu

#import pickle

import scipy.signal
from scipy import interpolate

import os, sys, datetime, fnmatch

import matplotlib.pyplot as plt
import multiprocessing

#from astroquery.mast import Catalogs
#import astroquery
#import astroquery.mast
#from astroquery.mast import Tesscut
#from astropy.coordinates import SkyCoord
#from astroquery.simbad import Simbad
#
#import astropy
#import astropy.wcs
#from astropy.wcs import WCS
#from astropy import units as u
#from astropy.io import fits
#import astropy.time

# Create test data
#timeinit = 3.14
#timedura = 100
#numbtimedayy = 48
#numbtime = int(timedura * numbtimedayy)
#time = numpy.linspace(timeinit, timeinit + timedura, numbtime)
#
## Use batman to create transits
#objtbatmpara = batman.TransitParams()
#objtbatmpara.t0 = timeinit  # time of inferior conjunction; first transit is X days after start
#objtbatmpara.per = 10.123  # orbital period
#objtbatmpara.rp = 6371. / 696342.  # 6371 planet radius (in units of stellar radii)
#objtbatmpara.a = 19.  # semi-major axis (in units of stellar radii)
#objtbatmpara.inc = 90.  # orbital inclination (in degrees)
#objtbatmpara.ecc = 0.  # eccentricity
#objtbatmpara.w = 90.  # longitude of periastron (in degrees)
#objtbatmpara.u = [0.4, 0.4]  # limb darkening coefficients
#objtbatmpara.limb_dark = "quadratic"  # limb darkening model
#objtbatmtran = batman.TransitModel(objtbatmpara, time)  # initializes model
#lcurmodl = objtbatmtran.light_curve(objtbatmpara)  # calculates light curve

## Create noise and merge with flux
#ppm = 50  # Noise level in parts per million
#noise = numpy.random.normal(0, 10**-6 * ppm, int(numbtime))
#flux = lcurmodl + noise

def retr_lpos(para, gdat):
    
    if (para[gdat.indxparahard] > gdat.limtpara[1, gdat.indxparahard]).any() or (para[gdat.indxparahard] < gdat.limtpara[0, gdat.indxparahard]).any():
        lpos = -np.inf
    else:
        lcurmodl, deptelli, deptbeam, deptslen = retr_modl(gdat, para)
        lpos = np.sum(-0.5 * (gdat.lcurflat - lcurmodl)**2 / gdat.varilcurflatthis)

        #print('para')
        #print(para)
        #print('lcurmodl')
        #summgene(lcurmodl)
        #print('deptelli')
        #summgene(deptelli)
        #print('deptbeam')
        #summgene(deptbeam)
        #print('deptslen')
        #summgene(deptslen)
        #print('lpos')
        #print(lpos)
        #print
    
    return lpos


def plot_datamodl(gdat):
    
    if gdat.boolblss:
        gdat.numbcols = 3
    else:
        gdat.numbcols = 2

    figr, axgr = plt.subplots(gdat.numbcols, 1, figsize=(10, 4.5))
    
    for k, axis in enumerate(axgr):
        
        if k == 0:
            if gdat.datatype == 'obsd':
                axis.text(0.5, 1.25, 'TIC %s' % (gdat.strgtici), color='k', transform=axis.transAxes, ha='center')
            #axis.text(0.5, 1.15, 'S:%.3g, P:%.3g day, D:%.3g, T:%.3g day' % (gdat.listsdee[gdat.indxfilethis], \
            #                                                                        gdat.fittperimaxmthis, gdat.dept, gdat.dura), ha='center', \
            #                                                                                                transform=axis.transAxes, color='b')
            if gdat.datatype == 'mock':
                axis.text(0.5, 1.05, 'P=%.3g day, M=%.3g M$_\odot$, Tmag=%.3g' % (gdat.trueperi[gdat.indxfilethis], \
                                                                        gdat.truemasscomp[gdat.indxfilethis], gdat.truetmag[gdat.indxfilethis]), \
                                                                                                        transform=axis.transAxes, color='g', ha='center')
            if gdat.datatype == 'mock':
                axis.plot(gdat.time, gdat.truelcurtotl[:, gdat.indxfilethis], color='g', ls='-')
                #axis.plot(gdat.time, gdat.truelcurelli[:, gdat.indxfilethis], color='g', ls='--')
                #axis.plot(gdat.time, gdat.truelcurbeam[:, gdat.indxfilethis], color='g', ls=':')
                axis.plot(gdat.time, gdat.truelcurslen[:, gdat.indxfilethis], color='g', ls='-.')
                
            axis.scatter(gdat.time, gdat.lcurthis, color='black', s=1)
            #axis.set_xlabel("Time [days]")
            axis.set_xticklabels([])
            axis.set_ylabel("Relative Flux")
            
        if k == 1:
            axis.scatter(gdat.timeflat, gdat.lcurflat, color='black', s=1)
            axis.set_xlabel("Time [days]")
            axis.set_ylabel("Detrended Relative Flux")
            if gdat.boolblss:
                if not (np.isscalar(gdat.timetran) and not np.isfinite(gdat.timetran)):
                    for timetran in gdat.timetran:
                        axis.axvline(timetran, color='orange', alpha=0.5, ls='--')
            if gdat.boolmcmc:
                for ii, i in enumerate(gdat.indxsampplot):
                    axis.plot(gdat.timeflat, gdat.postlcurmodl[ii, :], color='b', alpha=0.1)
            axis.set_xlabel("Time [days]")
            
            
        if k == 2 and gdat.boolblss:
            axis.axvline(gdat.fittperimaxmthis, alpha=0.5, color='b')
            for n in range(2, 10):
                axis.axvline(n*gdat.fittperimaxmthis, alpha=0.5, lw=1, linestyle="dashed", color='b')
                axis.axvline(gdat.fittperimaxmthis / n, alpha=0.5, lw=1, linestyle="dashed", color='b')
            axis.set_ylabel(r'SDE')
            axis.set_xlabel('Period [days]')
            axis.plot(gdat.peri, gdat.powr, color='black', lw=0.5)
            axis.set_xlim([np.amin(gdat.peri), np.amax(gdat.peri)])
        
        if k == 3:
            axis.plot(gdat.phasmodl, gdat.pcurmodl, color='violet')
            gdat.fittlcurmodl, gdat.fittdeptelli, gdat.fittdeptbeam, gdat.fittdeptslen = retr_modl(gdat, gdat.medipara)
            mediphas = (gdat.time / gdat.medipara[0] - gdat.medipara[1]) % 1.
            axis.plot(mediphas, gdat.fittlcurmodl, color='b')
            axis.plot(mediphas, gdat.fittdeptelli, color='b', ls='--')
            axis.plot(mediphas, gdat.fittdeptbeam, color='b', ls=':')
            axis.plot(mediphas, gdat.fittdeptslen, color='b', ls='-.')
            axis.scatter(gdat.phasdata, gdat.pcurdata, color='black', s=10, alpha=0.5, zorder=2)
            axis.set_xlabel('Phase')
            axis.set_ylabel('Relative flux');
    plt.subplots_adjust(hspace=0.35)
    print('Writing to %s...' % gdat.pathplottotl)
    plt.savefig(gdat.pathplottotl)
    plt.close()


def exec_srch(gdat):
    
    #lcurdetrregi, indxtimeregi, indxtimeregioutt, listobjtspln = \
    #                                                        tesstarg.util.detr_lcur(gdat.time, gdat.lcurthis)
    #gdat.lcurflat = np.concatenate(lcurdetrregi) 
    
    gdat.lcurflat = 1. + gdat.lcurthis - scipy.signal.medfilt(gdat.lcurthis, gdat.numbtimefilt)
    
    # mask out the edges
    timeedge = tesstarg.util.retr_timeedge(gdat.time)
    listindxtimemask = []
    for k in range(timeedge.size):
        if k != 0:
            indxtimemaskfrst = np.where(gdat.time < timeedge[k])[0][::-1][:2*gdat.numbtimefilt]
            listindxtimemask.append(indxtimemaskfrst)
        if k != timeedge.size - 1:
            indxtimemaskseco = np.where(gdat.time > timeedge[k])[0][:2*gdat.numbtimefilt]
            listindxtimemask.append(indxtimemaskseco)
    listindxtimemask = np.concatenate(listindxtimemask)
    
    gdat.listindxtimegoodedge = np.setdiff1d(gdat.indxtime, listindxtimemask)
    print('gdat.listindxtimegoodedge')
    summgene(gdat.listindxtimegoodedge)
    print('gdat.varilcurthis')
    summgene(gdat.varilcurthis)
    gdat.timeflat = gdat.time[gdat.listindxtimegoodedge]
    gdat.lcurflat = gdat.lcurflat[gdat.listindxtimegoodedge]
    gdat.varilcurflatthis = gdat.varilcurthis[gdat.listindxtimegoodedge]
    print('gdat.indxtime')
    summgene(gdat.indxtime)
    
    print('gdat.lcurthis')
    summgene(gdat.lcurthis)
    print('gdat.lcurflat')
    summgene(gdat.lcurflat)
    
    if gdat.boolblss:
        print('Performing TLS on %s...' % gdat.strgtici)
        model = transitleastsquares(gdat.time, 2. - gdat.lcurflat)
        # temp check how to do BLS instead of TLS
        gdat.results = model.power()
        gdat.listsdee[gdat.indxfilethis] = gdat.results.SDE
        gdat.fittperimaxmthis = gdat.results.period
        gdat.fittperimaxm.append(gdat.fittperimaxmthis)
        gdat.peri = gdat.results.periods
        gdat.dept = gdat.results.depth
        gdat.blssamplslen = 1 -  gdat.dept
        print('gdat.blssamplslen')
        print(gdat.blssamplslen)
        gdat.blssmasscomp = retr_masscomp(gdat, gdat.blssamplslen, 8.964)
        print('gdat.blssmasscomp')
        print(gdat.blssmasscomp)
        gdat.dura = gdat.results.duration
        gdat.powr = gdat.results.power
        gdat.timetran = gdat.results.transit_times
        gdat.phasmodl = gdat.results.model_folded_phase
        gdat.pcurmodl = 2. - gdat.results.model_folded_model
        gdat.phasdata = gdat.results.folded_phase
        gdat.pcurdata = 2. - gdat.results.folded_y
    

def retr_amplslen(gdat, peri, masscomp):
    
    amplslen = 7.15e-5 * peri**(2. / 3.) / gdat.radistar**2. * masscomp * (masscomp + gdat.massstar)**(1. / 3.)

    return amplslen


def retr_masscomp(gdat, amplslen, peri):
    
    # temp
    masscomp = amplslen / 7.15e-5 / peri**(2. / 3.) * gdat.radistar**2. / (gdat.massstar)**(1. / 3.)
    
    return masscomp


def retr_modl(gdat, para):
    
    epoc = para[0]
    peri = para[1]
    masscomp = para[2]
    
    if gdat.boolmodlflat:
        time = gdat.timeflat
    else:
        time = gdat.time
    phas = ((time - epoc) / peri) % 1.
    
    dura = 1.8 / 24. * np.pi / 4. * peri**(1. / 3.) * (masscomp + gdat.massstar)**(-1. / 3.) * gdat.radistar
    indxphasslen = np.where(abs(phas - 0.5) < dura / 2. / peri)[0]
    deptslen = np.zeros_like(time)
    deptslen[indxphasslen] += retr_amplslen(gdat, peri, masscomp)
    
    lcurmodl = 1. + deptslen
    amplelli = 1.89e-2* peri**(-2.) / gdat.densstar * (1. / (1. + gdat.massstar / masscomp))
    amplbeam = 2.8e-3 * peri**(-1. / 3.) * (gdat.massstar + masscomp)**(-2. / 3.) * masscomp
    deptelli = -amplelli * np.cos(4. * np.pi * phas) 
    deptbeam = -amplbeam * np.sin(2. * np.pi * phas)
    if not gdat.boolmodlflat:
        lcurmodl += deptelli + deptbeam
    
    return lcurmodl, deptelli + 1., deptbeam + 1., deptslen + 1.


def init_wrap( \
         listisec=None, \
         datatype='obsd', \
         boolmultproc=True, \
         **args \
        ):
 
    if gdat.boolmultproc:
        listobjtproc = []
    for gdat.isec in listisec:
        for gdat.icam in listicam:
            for gdat.iccd in listiccd:
                print('Reading files...')
                paththis = '/pdo/qlp-data/sector-%d/ffi/cam%d/ccd%d/LC/' % (isec, icam, iccd)
                print('paththis')
                print(paththis)
                liststrgfile = fnmatch.filter(os.listdir(paththis), '*.h5')
                numbfile = len(liststrgfile)
                print('Number of light curves: %s' % numbfile)
                liststrgfile = np.array(liststrgfile)
                n = np.arange(numbfile)
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
        indxruns, \
        datatype='obsd', \
        isec=None, \
        icam=None, \
        iccd=None, \
        strgmast=None, \
        boolmultproc=False, \
        verbtype=1, \
        **args
        ):
    
    # construct global object
    gdat = tdpy.util.gdatstrt()
    
    # copy unnamed inputs to the global object
    for attr, valu in locals().items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    # copy named arguments to the global object
    for strg, valu in args.items():
        setattr(gdat, strg, valu)

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print('BHOL initialized at %s...' % gdat.strgtimestmp)

    if indxruns is None:
        numpy.random.seed(0)
    else:
        numpy.random.seed(indxruns)

    # preliminary setup
    # construct the global object 
   
    # paths
    ## read PCAT path environment variable
    gdat.pathbase = os.environ['BHOL_DATA_PATH'] + '/'
    #gdat.pathdataqlop = gdat.pathbase + '/data/qlop/sector-%d/cam%d/ccd%d/' % (gdat.isec, gdat.icam, gdat.iccd)
    #os.system('os mkdir -p %s' % gdat.pathdataqlop)
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathimag = gdat.pathbase + 'imag/'
    ## define paths
    
    if gdat.datatype == 'mock':
        gdat.cade = 10. / 60. / 24. # days

    # settings
    ## plotting
    gdat.strgplotextn = 'pdf'
    gdat.boolblss = True
    gdat.boolmcmc = False

    gdat.strgdata = None
    gdat.boolsapp = False

    gdat.listlablpara = [['T$_0$', 'day'], ['P', 'day'], ['M', r'M$_s$']]
    gdat.numbpara = len(gdat.listlablpara)
    gdat.meanpara = np.empty(gdat.numbpara)
    gdat.stdvpara = np.empty(gdat.numbpara)
    gdat.minmpara = np.empty(gdat.numbpara)
    gdat.maxmpara = np.empty(gdat.numbpara)
    gdat.scalpara = np.empty(gdat.numbpara, dtype='object')
    gdat.fittminmmasscomp = 1.
    gdat.fittmaxmmasscomp = 10.
    gdat.minmpara[0] = -10.
    gdat.maxmpara[0] = 10.
    #gdat.meanpara[1] = 8.964
    #gdat.stdvpara[1] = 0.001
    gdat.minmpara[1] = 1.
    gdat.maxmpara[1] = 20.
    gdat.minmpara[2] = gdat.fittminmmasscomp
    gdat.maxmpara[2] = gdat.fittmaxmmasscomp
    gdat.scalpara[0] = 'self'
    gdat.scalpara[1] = 'self'
    gdat.scalpara[2] = 'self'
    
    gdat.limtpara = tesstarg.util.retr_limtpara(gdat.scalpara, gdat.minmpara, gdat.maxmpara, gdat.meanpara, gdat.stdvpara)
    gdat.indxparahard = np.where(gdat.scalpara == 'self')[0]

    if gdat.datatype == 'obsd':
        # temp
        gdat.liststrgfile = [''.join(gdat.strgmast.split(' '))]
        gdat.numbfile = 1
    if gdat.datatype == 'mock':
        gdat.numbfile = 5

    print('Number of light curves: %s' % gdat.numbfile)
    if gdat.numbfile == 0:
        return
    gdat.indxfile = np.arange(gdat.numbfile)
    
    if gdat.datatype == 'mock':
        gdat.numbtime = int((27.3 - 2.) / gdat.cade)
        if gdat.numbtime % 2 == 1:
            gdat.numbtime += 1
        print('gdat.numbtime') 
        print(gdat.numbtime)
        
    if gdat.datatype == 'obsd':
        arrylcur, arrylcursapp, arrylcurpdcc, listarrylcur, listarrylcursapp, listarrylcurpdcc = \
                                            tesstarg.util.retr_data(gdat.strgdata, gdat.strgmast, gdat.pathdata, gdat.boolsapp)
        gdat.time = arrylcur[:, 0]
        gdat.cade = gdat.time[1] - gdat.time[0]
        gdat.numbtime = gdat.time.size
    gdat.indxtime = np.arange(gdat.numbtime)
    
    gdat.lcur = np.empty((gdat.numbtime, gdat.numbfile))
    gdat.varilcur = np.empty((gdat.numbtime, gdat.numbfile))
    if gdat.datatype == 'obsd':
        gdat.lcur[:, 0] = arrylcur[:, 1]
        gdat.varilcur[:, 0] = arrylcur[:, 2]**2
    
    gdat.numbtimefilt = int(round(5. / 24. / gdat.cade))
    if gdat.numbtimefilt % 2 == 0:
        gdat.numbtimefilt += 1
    print('gdat.numbtimefilt')
    print(gdat.numbtimefilt)

    #numbsect = len(listarrylcur)
    #indxsect = np.arange(numbsect)
        
    
    print('gdat.cade [min]')
    print(gdat.cade * 24. * 60.)

    #gdat.radistar = 11.2
    #gdat.massstar = 18.
    gdat.radistar = 1.
    gdat.massstar = 1.
    gdat.densstar = 1.41

    # classification
    gdat.thrssdee = 6
    
    # emcee
    if gdat.boolmcmc:
        gdat.numbsampwalk = 1000
        gdat.numbsampburnwalk = 100

    # check inputs
    if gdat.isec is not None:
        strgsecc = '%02d%d%d' % (gdat.isec, gdat.icam, gdat.iccd)
        print('Sector: %d' % gdat.isec)
        print('Camera: %d' % gdat.icam)
        print('CCD: %d' % gdat.iccd)
    
    # get data
    # inject signal
    if gdat.datatype == 'mock':
        gdat.boolsigntrue = np.random.choice([0, 1], p=[0.1, 0.9], size=gdat.numbfile)
        gdat.indxtruesign = np.where(gdat.boolsigntrue == 1.)[0]
        gdat.numbsign = gdat.indxtruesign.size
    
    if gdat.datatype == 'true':
        print('gdat.boolsigntrue')
        summgene(gdat.boolsigntrue)
        print('gdat.indxtruesign')
        summgene(gdat.indxtruesign)
    
    # interpolate TESS photometric precision
    stdv = np.array([40., 40., 40., 90.,200.,700., 3e3, 2e4]) * 1e-6
    magt = np.array([ 2.,  4.,  6.,  8., 10., 12., 14., 16.])
    #objtspln = scipy.interpolate.UnivariateSpline(magt, stdv)
    #objtspln = np.poly1d(np.polyfit(magt, stdv, 3))
    objtspln = interpolate.interp1d(magt, stdv)

    # plot TESS photometric precision
    figr, axis = plt.subplots(figsize=(6, 4))
    axis.plot(magt, objtspln(magt))
    axis.set_xlabel('Tmag')
    axis.set_ylabel(r'$s$')
    axis.set_yscale('log')
    path = gdat.pathimag + 'sigmtmag.%s' % (gdat.strgplotextn) 
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
        
    figr, axis = plt.subplots(figsize=(5, 3))
    peri = np.logspace(-1, 2, 100)
    listtmag = [10., 13., 16.]
    listmasscomp = [1., 10., 100.]
    for masscomp in listmasscomp:
        amplslentmag = retr_amplslen(gdat, peri, masscomp)
        axis.plot(peri, amplslentmag, label=r'M = %.3g M$_\odot$' % masscomp)
    for tmag in listtmag:
        if tmag == 16:
            axis.text(0.1, objtspln(tmag) * 1.6, ('Tmag = %.3g' % tmag),  color='black')
        else:
            axis.text(0.1, objtspln(tmag) / 2, ('Tmag = %.3g' % tmag),  color='black')
        axis.axhline(objtspln(tmag), ls='--', color='black')#, label=('Tmag = %.3g' % tmag))
    axis.set_xlabel('Period [day]')
    axis.set_ylabel(r'Self-lensing amplitude')
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.legend(loc=4, framealpha=1.)
    path = gdat.pathimag + 'sigm.%s' % (gdat.strgplotextn) 
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
    
    listindxtimebadd = None

    if gdat.datatype == 'obsd':
        
        if listindxtimebadd is not None:
            listindxtimebadd = np.concatenate(listindxtimebadd)
            listindxtimebadd = np.unique(listindxtimebadd)
            listindxtimebadd = np.concatenate((listindxtimebadd, np.arange(100)))
            listindxtimebadd = np.concatenate((listindxtimebadd, numbtime / 2 + np.arange(100)))
            listindxtimegood = np.setdiff1d(indxtimetemp, listindxtimebadd)
            print('Filtering the data...')
            # filter the data
            time = time[listindxtimegood]
            gdat.lcur = gdat.lcur[listindxtimegood]
    else:
        # generate mock data
        gdat.boolmodlflat = False
        gdat.indxtruenull = np.setdiff1d(gdat.indxfile, gdat.indxtruesign)
        gdat.trueminmtmag = 8.
        gdat.truemaxmtmag = 16.
        gdat.truetmag = np.random.random(gdat.numbfile) * (gdat.truemaxmtmag - gdat.trueminmtmag) + gdat.trueminmtmag
    
        gdat.trueminmmasscomp = 1.
        gdat.truemaxmmasscomp = 10.
        gdat.truemasscomp = np.random.random(gdat.numbsign) * (gdat.truemaxmmasscomp - gdat.trueminmmasscomp) + gdat.trueminmmasscomp
        
        print('gdat.indxtruesign')
        summgene(gdat.indxtruesign)
        gdat.time = np.concatenate((np.arange(0., 27.3 / 2. - 1., gdat.cade), np.arange(27.3 / 2. + 1., 27.3, gdat.cade)))
        print('gdat.time')
        summgene(gdat.time)
        numbpara = 5
        gdat.trueepoc = np.random.rand(gdat.numbsign)
        gdat.trueperi = np.random.rand(gdat.numbsign) * 5. + 5.
       
        gdat.truelcurtotl = np.empty((gdat.numbtime, gdat.numbfile))
        gdat.truelcurelli = np.empty((gdat.numbtime, gdat.numbsign))
        gdat.truelcurbeam = np.empty((gdat.numbtime, gdat.numbsign))
        gdat.truelcurslen = np.empty((gdat.numbtime, gdat.numbsign))
        para = np.empty((gdat.numbsign, numbpara))
        para[:, 0] = gdat.trueepoc
        para[:, 1] = gdat.trueperi
        para[:, 2] = gdat.truemasscomp
        
        gdat.truestdvlcur = objtspln(gdat.truetmag)
        print('gdat.truelcurtotl')
        summgene(gdat.truelcurtotl)
        print('gdat.truestdvlcur')
        summgene(gdat.truestdvlcur)
        gdat.varilcur = gdat.truestdvlcur[None, :]**2 * np.ones_like(gdat.truelcurtotl)
        for nn, n in enumerate(gdat.indxtruesign):
            gdat.truelcurtotl[:, n], gdat.truelcurelli[:, nn], gdat.truelcurbeam[:, nn], gdat.truelcurslen[:, nn] = retr_modl(gdat, para[nn, :])
        for nn, n in enumerate(gdat.indxtruenull):
            gdat.truelcurtotl[:, n] = 1.
        gdat.lcur = np.copy(gdat.truelcurtotl)
        for n in gdat.indxtruesign:
            gdat.lcur[:, n] += gdat.truestdvlcur[n] * np.random.randn(gdat.numbtime)

        # histogram
        path = gdat.pathimag + 'truestdvlcur'
        tdpy.mcmc.plot_hist(path, gdat.truestdvlcur, r'$\sigma$', strgplotextn=gdat.strgplotextn)
        path = gdat.pathimag + 'truetmag'
        tdpy.mcmc.plot_hist(path, gdat.truetmag, 'Tmag', strgplotextn=gdat.strgplotextn)
            
    gdat.boolmodlflat = True
    if (~np.isfinite(gdat.lcur)).any():
        print('gdat.lcur')
        summgene(gdat.lcur)
        raise Exception('')

    pathlogg = gdat.pathdata + 'logg/'
    pathloggsave = pathlogg + 'save/'
    
    gdat.boolpositrue = np.zeros(gdat.numbfile)
    gdat.booltrueposi = []
    gdat.fittmasscomp = []
    gdat.fittperimaxm = []
    gdat.listsdee = np.empty(gdat.numbfile)
    gdat.indxfileposi = []
    
    print('gdat.lcur')
    summgene(gdat.lcur)

    for n in gdat.indxfile:
        
        gdat.indxfilethis = n
        gdat.lcurthis = gdat.lcur[:, n]
        gdat.varilcurthis = gdat.varilcur[:, n]

        if gdat.datatype == 'obsd':
            gdat.strgtici = gdat.liststrgfile[n][:-3]
            print('gdat.strgtici')
            print(gdat.strgtici)
            # temp
            #tici = int(gdat.strgtici)
        else:
            gdat.strgtici = '%08d' % n

        pathtcee = pathlogg + '%s_%s.txt' % (gdat.datatype, gdat.strgtici)
        
        if gdat.datatype == 'mock':
            print('gdat.truetmag[n]')
            print(gdat.truetmag[n])
            print('gdat.truestdvlcur[n]')
            print(gdat.truestdvlcur[n])
        gdat.pathplotlcur = gdat.pathimag + '%s_lcur_%s.%s' % (gdat.datatype, gdat.strgtici, gdat.strgplotextn)
        gdat.pathplotsdee = gdat.pathimag + '%s_sdee_%s.%s' % (gdat.datatype, gdat.strgtici, gdat.strgplotextn)
        gdat.pathplotpcur = gdat.pathimag + '%s_pcur_%s.%s' % (gdat.datatype, gdat.strgtici, gdat.strgplotextn)
        gdat.pathplottotl = gdat.pathimag + '%s_totl_%s.%s' % (gdat.datatype, gdat.strgtici, gdat.strgplotextn)
        boolpathplotdone = os.path.exists(gdat.pathplotlcur) and os.path.exists(gdat.pathplotsdee) and os.path.exists(gdat.pathplotpcur) \
                                                                                                            and os.path.exists(gdat.pathplottotl)

        # check if log file has been created properly before
        gdat.boolloggprev = False
        if os.path.exists(pathtcee):
            cntr = 0
            with open(pathtcee, 'r') as objtfile:
                for k, line in enumerate(objtfile):
                    cntr += 1
            if cntr == 5:
                gdat.boolloggprev = True
        
        if not gdat.boolloggprev:
            print('Reading %s...' % gdat.strgtici)
            # log file
            filelogg = open(pathtcee, 'w+')
            exec_srch(gdat)
            filelogg.write('SDE: %g\n' % gdat.results.SDE)
            filelogg.write('Period: %g\n' % gdat.results.period)
            filelogg.write('Depth: %g\n' % gdat.results.depth)
            filelogg.write('Duration: %g\n' % gdat.results.duration)
            filelogg.write('\n')
            filelogg.close()
        
        else:
            print('BLS has already been done. Reading the log file for %s at %s...' % (gdat.strgtici, pathtcee))
            filelogg = open(pathtcee, 'r')
            for k, line in enumerate(filelogg):
                if k == 4:
                    continue
                valu = float(line.split(' ')[1])
                if k == 0:
                    gdat.listsdee[n] = valu
                if k == 1:
                    gdat.peri = valu
                if k == 2:
                    gdat.dept = valu
                if k == 3:
                    gdat.dura = valu
            filelogg.close()
       
        if gdat.listsdee[n] >= gdat.thrssdee:
            gdat.boolpositrue[n] = 1.
            if gdat.boolsigntrue[n]:
                gdat.booltrueposi.append(1.)
            else:
                gdat.booltrueposi.append(0.)
            gdat.indxfileposi.append(n)
            if not boolpathplotdone:
                exec_srch(gdat)
    
                if gdat.boolmcmc:
                    print('Performing sampling on %s...' % gdat.strgtici)
                    dictllik = [gdat]
                    gdat.objtsamp = emcee.EnsembleSampler(gdat.numbwalk, gdat.numbpara, retr_lpos, args=dictllik, threads=10)
                    gdat.parainitburn, prob, state = gdat.objtsamp.run_mcmc(gdat.parainit, gdat.numbsampwalk)
                    gdat.medipara = np.median(gdat.objtsamp.flatchain[gdat.numbsamp/2:, :], 0)
                    gdat.fittmasscomp.append(gdat.medipara[4])

                print('Making plots...')
            else:
                print('Plots have been made already at %s. Skipping...' % gdat.pathplottotl)
        
        gdat.numbdata = gdat.numbtime

        exec_srch(gdat)
        if gdat.boolmcmc:
            gdat.parapost = tesstarg.util.samp(gdat, gdat.pathimag, gdat.numbsampwalk, gdat.numbsampburnwalk, retr_modl, retr_lpos, \
                                               gdat.listlablpara, gdat.scalpara, gdat.minmpara, gdat.maxmpara, gdat.meanpara, gdat.stdvpara, gdat.numbdata)

            gdat.numbsamp = gdat.parapost.shape[0]
            gdat.indxsamp = np.arange(gdat.numbsamp)
            gdat.indxsampplot = gdat.indxsamp[::100]
            gdat.numbtimeflat = gdat.timeflat.size
            gdat.numbsampplot = gdat.indxsampplot.size
            gdat.postlcurmodl = np.empty((gdat.numbsampplot, gdat.numbtimeflat))
            for ii, i in enumerate(gdat.indxsampplot):
                gdat.postlcurmodl[ii, :], temp, temp, temp = retr_modl(gdat, gdat.parapost[i, :])
        plot_datamodl(gdat)
        print
        


    gdat.indxfileposi = np.array(gdat.indxfileposi)
    gdat.fittperimaxm = np.array(gdat.fittperimaxm)
    gdat.fittmasscomp = np.array(gdat.fittmasscomp)
    gdat.booltrueposi = np.array(gdat.booltrueposi)
    # plot distributions
    
    numbbins = 10
    indxbins = np.arange(numbbins)
    if gdat.datatype == 'mock':
        indx = np.arange(2)
    else:
        indx = np.arange(1)
    for c in indx:
        
        if c == 1:
            listvarb = [gdat.trueperi, gdat.truemasscomp, gdat.truetmag[gdat.indxtruesign]]
            listlablvarb = ['P', 'M_c', 'Tmag']
            liststrgvarb = ['trueperi', 'truemasscomp', 'truetmag']
        else:
            listvarb = [gdat.listsdee]
            listlablvarb = ['SNR']
            liststrgvarb = ['sdee']
            
        for k, varbfrst in enumerate(listvarb):
            
            # histogram
            figr, axis = plt.subplots(figsize=(6, 4))
            axis.hist(varbfrst)
            if liststrgvarb[k] == 'sdee':
                axis.axvline(gdat.thrssdee, ls='--', color='black')
            axis.set_xlabel(listlablvarb[k])
            axis.set_ylabel('N')
            path = gdat.pathimag + 'hist%s.%s' % (liststrgvarb[k], gdat.strgplotextn) 
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            if gdat.datatype == 'mock':
                bins = np.linspace(np.amin(varbfrst), np.amax(varbfrst), numbbins + 1)
                meanvarb = (bins[1:] + bins[:-1]) / 2.
                metr = np.zeros(numbbins) - 1.
                for a in indxbins:
                    indxfilebins = np.where((bins[a] < varbfrst) & (valu < bins[a+1]))[0]
                    numbfilebins = indxfilebins.size
                    if numbfilebins > 0:
                        # completeness
                        if c == 1:
                            metr[a] = float(np.sum(gdat.boolpositrue[indxfilebins])) / numbfilebins
                        else:
                            print('a')
                            print(a)
                            print('metr')
                            summgene(metr)
                            print('gdat.booltrueposi')
                            summgene(gdat.booltrueposi)
                            print('indxfilebins')
                            summgene(indxfilebins)
                            print
                            metr[a] = float(np.sum(gdat.booltrueposi[indxfilebins])) / numbfilebins
                
                if c == 1:
                    strgmetr = 'cmpl'
                    strgyaxi = 'Completeness'
                else:
                    strgmetr = 'fdis'
                    strgyaxi = 'False Discovery Rate'
                
                figr, axis = plt.subplots(figsize=(6, 4))
                axis.plot(meanvarb, metr)
                axis.set_xlabel(listlablvarb[k])
                axis.set_ylabel(strgyaxi)
                path = gdat.pathimag + '%s%s.%s' % (strgmetr, liststrgvarb[k], gdat.strgplotextn) 
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()

            for l, varbseco in enumerate(listvarb):
                
                if k == l:
                    continue

                # plot distributions
                figr, axis = plt.subplots(figsize=(6, 4))
                print('liststrgvarb[k]')
                print(liststrgvarb[k])
                print('liststrgvarb[l]')
                print(liststrgvarb[l])
                print('varbfrst')
                summgene(varbfrst)
                print('varbseco')
                summgene(varbseco)
                print
                axis.scatter(varbfrst, varbseco)
                axis.set_xlabel(listlablvarb[k])
                axis.set_ylabel(listlablvarb[l])
                if liststrgvarb[k] == 'sdee':
                    axis.axvline(gdat.thrssdee, ls='--', color='black')
                if liststrgvarb[l] == 'sdee':
                    axis.axhline(gdat.thrssdee, ls='--', color='black')
                path = gdat.pathimag + 'scat%s%s.%s' % (liststrgvarb[k], liststrgvarb[l], gdat.strgplotextn)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()

    

def cnfg_obsd():
   
    listisec = [9]
    init( \
         None, \
         #boolmultproc=False, \
         #listisec=listisec, \
         strgmast='Vela X-1', \
        )


def cnfg_mock():
   
    listisec = [9]
    init( \
         None, \
         #boolmultproc=False, \
         #listisec=listisec, \
         
         datatype='mock', \
        )


globals().get(sys.argv[1])()


