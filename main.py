import matplotlib.pyplot as plt

from transitleastsquares import transitleastsquares

from tdpy.util import summgene
import tdpy.util
import tdpy.mcmc
import tesstarg.util

import numpy as np

import h5py

import scipy
from scipy import fftpack
from scipy import signal
from scipy import optimize
import scipy.signal
from scipy import interpolate

import time as timemodu

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
    
    if gdat.boolmodlbdtr:
        time = gdat.timethis
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
    if not gdat.boolmodlbdtr:
        lcurmodl += deptelli + deptbeam
    
    return lcurmodl, deptelli + 1., deptbeam + 1., deptslen + 1.


def retr_lcur(peri, amplellp, ampldopp, ampllens, phaslens, stdvlens, boolfull):
    
    peri = para[0]
    amplellp = para[1]
    ampldopp = para[2]
    ampllens = para[3]
    phaslens = para[4]
    stdvlens = para[5]
    
    lcurellp = amplellp * np.sin(2. * np.pi * gdat.time / peri)
    lcurdopp = -ampldopp * np.sin(2. * np.pi * gdat.time / peri)
    lcurlens = np.zeros_like(lcurdopp) 
    for k in range(10):
        lcurlens += ampllens / np.sqrt(2. * np.pi) / stdvlens * np.exp(-0.5 * ((k + phaslens) * peri - gdat.time)**2 / stdvlens**2)
    
    lcur = lcurellp + lcurdopp + lcurlens
    lcur += stdvnois * np.random.randn(gdat.numbtime)
    
    return lcurellp, lcurdopp, lcurlens, lcur
    

def retr_llik(para, gdat):
    
    lcurmodl, deptelli, deptbeam, deptslen = retr_modl(gdat, para)
    lpos = np.sum(-0.5 * (gdat.lcurbdtr - lcurmodl)**2 / gdat.varilcurbdtrthis)

    return lpos


def plot_lcur(path, lcur, lcurellp=None, lcurdopp=None, lcurlens=None, titl=None):
    
    figr, axis = plt.subplots(figsize=(6, 3))
    axis.plot(gdat.time, lcur, ls='', marker='o', markersize=1, label='Tot', color='black')
    if lcurellp is not None:
        axis.plot(gdat.time, lcurellp, ls='', marker='o', markersize=1, label='EV')
        axis.plot(gdat.time, lcurdopp, ls='', marker='o', markersize=1, label='DP')
        axis.plot(gdat.time, lcurlens, ls='', marker='o', markersize=1, label='SL')
        axis.legend()
    axis.set_xlabel('T [BJD]')
    if titl != None:
        axis.set_title(titl)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_psdn(path, psdn, psdnellp=None, psdndopp=None, psdnlens=None, titl=None):
    
    figr, axis = plt.subplots(figsize=(6, 3))
    axis.plot(perisamp, psdn**2, ls='', marker='o', markersize=1, label='Tot', color='black')
    if psdnellp is not None:
        axis.plot(perisamp, psdnellp**2, ls='', marker='o', markersize=1, label='EV')
        axis.plot(perisamp, psdndopp**2, ls='', marker='o', markersize=1, label='DP')
        axis.plot(perisamp, psdnlens**2, ls='', marker='o', markersize=1, label='SL')
        axis.legend()
    axis.axvline(peri, ls='--', alpha=0.3, color='black')
    axis.set_xlabel('P [day]')
    axis.set_xscale('log')
    axis.set_yscale('log')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    

def plot_datamodl(gdat):
    
    if gdat.boolblss:
        gdat.numbcols = 3
    else:
        gdat.numbcols = 2

    figr, axgr = plt.subplots(gdat.numbcols, 1, figsize=(10, 4.5))
    
    for k, axis in enumerate(axgr):
        
        if k == 0:
            if gdat.datatype == 'obsd':
                axis.text(0.5, 1.25, 'TIC %s' % (gdat.strglcur), color='k', transform=axis.transAxes, ha='center')
            #axis.text(0.5, 1.15, 'S:%.3g, P:%.3g day, D:%.3g, T:%.3g day' % (gdat.listsdee[gdat.indxlcurthis], \
            #                                                                        gdat.fittperimaxmthis, gdat.dept, gdat.dura), ha='center', \
            #                                                                                                transform=axis.transAxes, color='b')
            if gdat.datatype == 'mock':
                axis.text(0.5, 1.05, 'P=%.3g day, M=%.3g M$_\odot$, Tmag=%.3g' % (gdat.trueperi[gdat.indxlcurthis], \
                                                                        gdat.truemasscomp[gdat.indxlcurthis], gdat.truetmag[gdat.indxlcurthis]), \
                                                                                                        transform=axis.transAxes, color='g', ha='center')
                axis.plot(gdat.time, gdat.truelcurtotl[:, gdat.indxlcurthis], color='g', ls='-')
                #axis.plot(gdat.time, gdat.truelcurelli[:, gdat.indxlcurthis], color='g', ls='--')
                #axis.plot(gdat.time, gdat.truelcurbeam[:, gdat.indxlcurthis], color='g', ls=':')
                axis.plot(gdat.time, gdat.truelcurslen[:, gdat.indxlcurthis], color='g', ls='-.')
                
            axis.scatter(gdat.time, gdat.lcurthis, color='black', s=1)
            #axis.set_xlabel("Time [days]")
            axis.set_xticklabels([])
            axis.set_ylabel("Relative Flux")
            
        if k == 1:
            axis.scatter(gdat.timethis, gdat.lcurbdtr, color='black', s=1)
            axis.set_xlabel("Time [days]")
            axis.set_ylabel("Detrended Relative Flux")
            if gdat.boolblss:
                for k in range(len(gdat.dicttlss['peri'])):
                    for n in range(-10, 10):
                        axis.axvline(gdat.dicttlss['peri'] * n + gdat.dicttlss['epoc'], color='orange', alpha=0.5, ls='--')
            if gdat.boolmcmc:
                for ii, i in enumerate(gdat.indxsampplot):
                    axis.plot(gdat.timethis, gdat.postlcurmodl[ii, :], color='b', alpha=0.1)
            axis.set_xlabel("Time [days]")
            
        gdat.numbtlss = len(gdat.dicttlss['peri']) 
        gdat.indxtlss = np.arange(gdat.numbtlss)

        if k == 2 and gdat.boolblss:
            for k in gdat.indxtlss:
                axis.axvline(gdat.dicttlss['peri'][k], alpha=0.5, color='b')
                for n in range(2, 10):
                    axis.axvline(n*gdat.dicttlss['peri'][k], alpha=0.5, lw=1, linestyle="dashed", color='b')
                    axis.axvline(gdat.dicttlss['peri'][k] / n, alpha=0.5, lw=1, linestyle="dashed", color='b')
            axis.set_ylabel(r'SDE')
            axis.set_xlabel('Period [days]')
            axis.plot(gdat.dicttlss['listperi'], gdat.dicttlss['powr'], color='black', lw=0.5)
            #axis.set_xlim([np.amin(gdat.peri), np.amax(gdat.peri)])
        
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
    
    # baseline detrend
    lcurbdtrregi, gdat.listindxtimeregi, gdat.indxtimeregioutt, gdat.listobjtspln, timeedge = \
                     tesstarg.util.bdtr_lcur(gdat.timethis, gdat.lcurthis, weigsplnbdtr=gdat.weigsplnbdtr, \
                                                verbtype=gdat.verbtype, durabrek=gdat.durabrek, ordrspln=gdat.ordrspln, bdtrtype=gdat.bdtrtype)
    
    gdat.numbtime = gdat.timethis.size
    gdat.listarrylcurbdtr = np.zeros((gdat.numbtime, 3))
    gdat.listarrylcurbdtr[:, 0] = gdat.timethis
    gdat.listarrylcurbdtr[:, 1] = np.concatenate(lcurbdtrregi)
    gdat.lcurbdtr = gdat.listarrylcurbdtr[:, 1]
    numbsplnregi = len(lcurbdtrregi)
    gdat.indxsplnregi = np.arange(numbsplnregi)
    gdat.indxtime = np.arange(gdat.numbtime)
    
    # mask out the edges
    #durabrek = 0.5
    #timeedge = tesstarg.util.retr_timeedge(gdat.timethis, gdat.lcurbdtr, durabrek)
    #listindxtimemask = []
    #for k in range(timeedge.size):
    #    if k != 0:
    #        indxtimemaskfrst = np.where(gdat.time < timeedge[k])[0][::-1][:2*gdat.numbtimefilt]
    #        print('indxtimemaskfrst')
    #        summgene(indxtimemaskfrst)
    #        listindxtimemask.append(indxtimemaskfrst)
    #    if k != timeedge.size - 1:
    #        indxtimemaskseco = np.where(gdat.time > timeedge[k])[0][:2*gdat.numbtimefilt]
    #        print('indxtimemaskseco')
    #        summgene(indxtimemaskseco)
    #        listindxtimemask.append(indxtimemaskseco)
    #    print('listindxtimemask')
    #    print(listindxtimemask)
    #listindxtimemask = np.concatenate(listindxtimemask)
    #gdat.listindxtimegoodedge = np.setdiff1d(gdat.indxtime, listindxtimemask)
    #gdat.timethis = gdat.time[gdat.listindxtimegoodedge]
    #gdat.lcurbdtr = gdat.lcurbdtr[gdat.listindxtimegoodedge]
    #gdat.varilcurbdtrthis = gdat.varilcurthis[gdat.listindxtimegoodedge]
    
    print('gdat.lcurthis')
    summgene(gdat.lcurthis)
    print('gdat.lcurbdtr')
    summgene(gdat.lcurbdtr)
    
    if gdat.boolblss:
        print('Performing TLS on %s...' % gdat.strglcur)
        arry = np.zeros((gdat.numbtime, 3))
        arry[:, 0] = gdat.timethis
        arry[:, 1] = 2. - gdat.lcurbdtr
        gdat.dicttlss = tesstarg.util.exec_tlss(arry, gdat.pathimag, thrssdee=gdat.thrssdee)#, ticitarg=ticitarg)


def proc_samp(gdat):
    
    freq, gdat.psdn = scipy.signal.periodogram(gdat.lcurdata, fs=fs, window=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)


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
                numblcur = len(liststrgfile)
                print('Number of light curves: %s' % numblcur)
                liststrgfile = np.array(liststrgfile)
                n = np.arange(numblcur)
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
        datatype='obsd', \
        
        # data input
        listticitarg=None, \
        liststrgmast=None, \
        isec=None, \
        
        # method, mfil or tlss
        strgmeth='tlss', \
        
        # baseline detrending
        weigsplnbdtr=None, \
        ordrspln=None, \
        durabrek=None, \
        bdtrtype=None, \
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
   
    # check input
    if gdat.liststrgmast is None and gdat.listticitarg is None and gdat.isec is None:
        raise Exception('')
    if gdat.listticitarg is not None and (gdat.liststrgmast is not None or gdat.isec is not None):
        raise Exception('')
    if gdat.liststrgmast is not None and (gdat.listticitarg is not None or gdat.isec is not None):
        raise Exception('')
    if gdat.isec is not None and (gdat.liststrgmast is not None or gdat.listticitarg is not None):
        raise Exception('')
    if gdat.liststrgmast is not None:
        gdat.inpttype = 'mast'
    if gdat.isec is not None:
        gdat.inpttype = 'sect'
    if gdat.listticitarg is not None:
        gdat.inpttype = 'tici'
    print('BHOL initialized at %s...' % gdat.strgtimestmp)

    np.random.seed(0)

    # preliminary setup
   
    # paths
    ## read PCAT path environment variable
    gdat.pathbase = os.environ['BHOL_DATA_PATH'] + '/'
    #gdat.pathdataqlop = gdat.pathbase + '/data/qlop/sector-%d/cam%d/ccd%d/' % (gdat.isec, gdat.icam, gdat.iccd)
    #os.system('os mkdir -p %s' % gdat.pathdataqlop)
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathimag = gdat.pathbase + 'imag/'
    ## define paths
    
    # settings
    # mock data
    if gdat.datatype == 'mock':
        gdat.cade = 10. / 60. / 24. # days
    
    # SDE threshold
    gdat.thrssdee = 7.1

    ## plotting
    gdat.strgplotextn = 'pdf'
    gdat.boolblss = True
    gdat.boolmcmc = False

    gdat.strgdata = None
    gdat.boolsapp = False
    
    ### MCMC
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
    
    gdat.limtpara = tdpy.mcmc.retr_limtpara(gdat.scalpara, gdat.minmpara, gdat.maxmpara, gdat.meanpara, gdat.stdvpara)
    gdat.indxparahard = np.where(gdat.scalpara == 'self')[0]

    if gdat.datatype == 'obsd':
        gdat.numblcur = len(gdat.liststrgmast)
    if gdat.datatype == 'mock':
        gdat.numblcur = 5

    print('Number of light curves: %s' % gdat.numblcur)
    if gdat.numblcur == 0:
        return
    gdat.indxlcur = np.arange(gdat.numblcur)
    
    if gdat.datatype == 'mock':
        print('gdat.numbtime') 
        print(gdat.numbtime)
        gdat.indxtimelink = np.where(abs(gdat.time - 13.7) < 2.)[0]
    
    # get data
    gdat.lcur = []
    gdat.varilcur = []
    if gdat.datatype == 'obsd':
        gdat.time = [[] for n in gdat.indxlcur]
        gdat.lcur = [[] for n in gdat.indxlcur]
        gdat.varilcur = [[] for n in gdat.indxlcur]
        for n in gdat.indxlcur:
            datatype, arrylcur, arrylcursapp, arrylcurpdcc, listarrylcur, listarrylcursapp, listarrylcurpdcc, listisec, listicam, listiccd = \
                                                tesstarg.util.retr_data(gdat.strgdata, gdat.liststrgmast[n], gdat.pathdata, gdat.boolsapp)
            gdat.time[n] = arrylcur[:, 0]
            gdat.lcur[n] = arrylcur[:, 1]
            gdat.varilcur[n] = arrylcur[:, 2]**2
            gdat.cade = np.amin(gdat.time[n][1:] - gdat.time[n][:-1])
            
            print('gdat.lcur[n]')
            summgene(gdat.lcur[n])

    # baseline detrending
    gdat.numbtimefilt = int(round(5. / 24. / gdat.cade))
    if gdat.numbtimefilt % 2 == 0:
        gdat.numbtimefilt += 1
    print('gdat.numbtimefilt')
    print(gdat.numbtimefilt)

    #numbchun = len(listarrylcur)
    #indxchun = np.arange(numbchun)
    
    print('gdat.cade [min]')
    print(gdat.cade * 24. * 60.)
    
    # to be done by pexo
    ## target properties
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
        gdat.boolreletrue = np.random.choice([0, 1], p=[0.1, 0.9], size=gdat.numblcur)
        gdat.indxtruerele = np.where(gdat.boolreletrue == 1.)[0]
        gdat.numbrele = gdat.indxtruerele.size
    
        print('gdat.boolreletrue')
        summgene(gdat.boolreletrue)
        print('gdat.indxtruerele')
        summgene(gdat.indxtruerele)
    
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
    
    if gdat.datatype == 'obsd':
        pass
        #listindxtimebadd = np.concatenate(listindxtimebadd)
        #listindxtimebadd = np.unique(listindxtimebadd)
        #listindxtimebadd = np.concatenate((listindxtimebadd, np.arange(100)))
        #listindxtimebadd = np.concatenate((listindxtimebadd, numbtime / 2 + np.arange(100)))
        #listindxtimegood = np.setdiff1d(indxtimetemp, listindxtimebadd)
        #print('Filtering the data...')
        ## filter the data
        #time = time[listindxtimegood]
        #gdat.lcur = gdat.lcur[listindxtimegood]
    else:

        # mock data setup 
    
        peri = 3. # [days]
        stdvnois = 1e-2
        amplellp = 10.
        ampldopp = 1.
        ampllens = 1.
        phaslens = 0.5
        stdvlens = 0.1
        
        gdat.numbpara = 7
        
        argsdict = {'gdat':gdat}
        args = (argsdict)
        paratrue = np.empty(gdat.numbpara)
        paratrue[0] = peri
        paratrue[1] = amplellp
        paratrue[2] = ampldopp
        paratrue[3] = ampllens
        paratrue[4] = phaslens
        paratrue[5] = stdvlens
        paratrue[6] = stdvnois
    
        gdat.boolfull = True
        lcurellp, lcurdopp, lcurlens, lcur = retr_lcur(peri, amplellp, ampldopp, ampllens, phaslens, stdvlens, True)
    
        delttime = 1. / 24. / 2.
        freq, gdat.psdn = scipy.signal.periodogram(lcur, fs=fs, window=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
        perisamp = 1. / freq
        
        pathplot = gdat.pathdata + 'lcur.pdf'
        if not os.path.exists(pathplot):
            plot_lcur(pathplot, lcur, lcurellp, lcurdopp, lcurlens)
        
        parainit = paratrue
    	
        gdat.boolfull = False
        numbsamp = 10
        indxsamp = np.arange(numbsamp)
        boolreletrue = np.ones(numbsamp, dtype=bool)
        boolreletrue[0] = False
        boolreleposi = np.empty_like(boolreletrue)
    
        if boolreletrue[k]:
            truepara = np.empty(gdat.numbpara)
            truepara[0] = np.random.uniform() * 9. + 1.
            truepara[1] = (1. + 1e-1 * np.random.randn()) * 10.
            truepara[2] = (1. + 1e-1 * np.random.randn()) * 1.
            truepara[3] = (1. + 1e-1 * np.random.randn()) * 1.
            truepara[4] = (1. + 1e-1 * np.random.randn()) * 0.5
            truepara[5] = (1. + 1e-1 * np.random.randn()) * 0.1
            truepara[6] = (1. + 1e-1 * np.random.randn()) * 1e-2
            gdat.lcur = retr_lcur(paratrue[0], paratrue[1], paratrue[2], paratrue[3], paratrue[4], paratrue[5], False)
            lcurdata = (1. + 1e-1 * np.random.randn(gdat.numbtime)) * gdat.lcur
            lcurdata[gdat.indxtimelink] = 0.
            #lcurdata[gdat.indxtimelink] = np.nan
            gdat.lcurdata = retr_lcurdata(gdat.lcur)
        else:
            gdat.lcurdata = np.random.randn(gdat.numbtime)

        # generate mock data
        gdat.boolmodlbdtr = False
        gdat.indxtruenull = np.setdiff1d(gdat.indxlcur, gdat.indxtruerele)
        gdat.trueminmtmag = 8.
        gdat.truemaxmtmag = 16.
        gdat.truetmag = np.random.random(gdat.numblcur) * (gdat.truemaxmtmag - gdat.trueminmtmag) + gdat.trueminmtmag
    
        gdat.trueminmmasscomp = 1.
        gdat.truemaxmmasscomp = 10.
        gdat.truemasscomp = np.random.random(gdat.numbrele) * (gdat.truemaxmmasscomp - gdat.trueminmmasscomp) + gdat.trueminmmasscomp
        
        print('gdat.indxtruerele')
        summgene(gdat.indxtruerele)
        gdat.time = np.concatenate((np.arange(0., 27.3 / 2. - 1., gdat.cade), np.arange(27.3 / 2. + 1., 27.3, gdat.cade)))
        print('gdat.time')
        summgene(gdat.time)
        numbpara = 5
        gdat.trueepoc = np.random.rand(gdat.numbrele)
        gdat.trueperi = np.random.rand(gdat.numbrele) * 5. + 5.
       
        gdat.truelcurtotl = np.empty((gdat.numbtime, gdat.numblcur))
        gdat.truelcurelli = np.empty((gdat.numbtime, gdat.numbrele))
        gdat.truelcurbeam = np.empty((gdat.numbtime, gdat.numbrele))
        gdat.truelcurslen = np.empty((gdat.numbtime, gdat.numbrele))
        para = np.empty((gdat.numbrele, numbpara))
        para[:, 0] = gdat.trueepoc
        para[:, 1] = gdat.trueperi
        para[:, 2] = gdat.truemasscomp
        
        gdat.truestdvlcur = objtspln(gdat.truetmag)
        print('gdat.truelcurtotl')
        summgene(gdat.truelcurtotl)
        print('gdat.truestdvlcur')
        summgene(gdat.truestdvlcur)
        gdat.varilcur = gdat.truestdvlcur[None, :]**2 * np.ones_like(gdat.truelcurtotl)
        for nn, n in enumerate(gdat.indxtruerele):
            gdat.truelcurtotl[:, n], gdat.truelcurelli[:, nn], gdat.truelcurbeam[:, nn], gdat.truelcurslen[:, nn] = retr_modl(gdat, para[nn, :])
        for nn, n in enumerate(gdat.indxtruenull):
            gdat.truelcurtotl[:, n] = 1.
        gdat.lcur = np.copy(gdat.truelcurtotl)
        for n in gdat.indxtruerele:
            gdat.lcur[n] += gdat.truestdvlcur[n] * np.random.randn(gdat.numbtime)

        # histogram
        path = gdat.pathimag + 'truestdvlcur'
        tdpy.mcmc.plot_hist(path, gdat.truestdvlcur, r'$\sigma$', strgplotextn=gdat.strgplotextn)
        path = gdat.pathimag + 'truetmag'
        tdpy.mcmc.plot_hist(path, gdat.truetmag, 'Tmag', strgplotextn=gdat.strgplotextn)
            
        # Boolean array of whether the mock light curves with signal have been labeled positively
        gdat.boolpositrue = np.zeros(gdat.numblcur)
        
        # Boolean array of whether the positives have signal in them
        gdat.booltrueposi = []
    
    gdat.boolmodlbdtr = True

    pathlogg = gdat.pathdata + 'logg/'
    pathloggsave = pathlogg + 'save/'
    
    gdat.fittmasscomp = []
    gdat.fittperimaxm = []
    gdat.listsdee = np.empty(gdat.numblcur)
    gdat.indxlcurposi = []
    
    boolposirele = np.empty(gdat.numblcur, dtype=bool)
    print('gdat.lcur')
    summgene(gdat.lcur)

    for n in gdat.indxlcur:
        
        gdat.indxlcurthis = n
        gdat.timethis = gdat.time[n]
        gdat.lcurthis = gdat.lcur[n]
        gdat.varilcurthis = gdat.varilcur[n]
        print('gdat.lcurthis')
        summgene(gdat.lcurthis)
        delttime = np.amin(gdat.timethis[1:] - gdat.timethis[:-1])
        fs = 1. / delttime

        # check data for finiteness
        if (~np.isfinite(gdat.lcurthis)).any():
            print('gdat.lcurthis')
            summgene(gdat.lcurthis)
            raise Exception('')
        
        if gdat.datatype == 'obsd':
            if gdat.inpttype == 'sect':
                gdat.strglcur = 'sc%02d_%12d' % gdat.listticitarg[n]
            if gdat.inpttype == 'mast':
                gdat.strglcur = gdat.liststrgmast[n]
            if gdat.inpttype == 'tici':
                gdat.strglcur = 'targ_%12d' % gdat.listticitarg[n]
            print('gdat.strglcur')
            print(gdat.strglcur)
        else:
            gdat.strglcur = '%08d' % n
        
        print('gdat.inpttype')
        print(gdat.inpttype)
        print('gdat.strglcur')
        print(gdat.strglcur)
        pathtcee = pathlogg + '%s_%s.txt' % (gdat.datatype, gdat.strglcur)
        
        if gdat.datatype == 'mock':
            print('gdat.truetmag[n]')
            print(gdat.truetmag[n])
            print('gdat.truestdvlcur[n]')
            print(gdat.truestdvlcur[n])
        gdat.pathplotlcur = gdat.pathimag + '%s_lcur_%s.%s' % (gdat.datatype, gdat.strglcur, gdat.strgplotextn)
        gdat.pathplotsdee = gdat.pathimag + '%s_sdee_%s.%s' % (gdat.datatype, gdat.strglcur, gdat.strgplotextn)
        gdat.pathplotpcur = gdat.pathimag + '%s_pcur_%s.%s' % (gdat.datatype, gdat.strglcur, gdat.strgplotextn)
        gdat.pathplottotl = gdat.pathimag + '%s_totl_%s.%s' % (gdat.datatype, gdat.strglcur, gdat.strgplotextn)
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
            print('Reading %s...' % gdat.strglcur)
            # log file
            filelogg = open(pathtcee, 'w+')
            exec_srch(gdat)
        
        else:
            print('BLS has already been done. Reading the log file for %s at %s...' % (gdat.strglcur, pathtcee))
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
       
        # perform matched filter
        exec_srch(gdat)

        # calculate PSD
        freq, gdat.psdn = scipy.signal.periodogram(gdat.lcurthis, fs=fs)
        
        if gdat.listsdee[n] >= gdat.thrssdee:
            
            if gdat.datatype == 'mock':
                gdat.boolpositrue[n] = 1.
                if gdat.boolreletrue[n]:
                    gdat.booltrueposi.append(1.)
                else:
                    gdat.booltrueposi.append(0.)
            gdat.indxlcurposi.append(n)
            if not boolpathplotdone:
                exec_srch(gdat)
                proc_samp(gdat)
    
                if gdat.boolmcmc:
                    print('Performing sampling on %s...' % gdat.strglcur)
                    dictllik = [gdat]
                    gdat.objtsamp = emcee.EnsembleSampler(gdat.numbwalk, gdat.numbpara, retr_lpos, args=dictllik, threads=10)
                    gdat.parainitburn, prob, state = gdat.objtsamp.run_mcmc(gdat.parainit, gdat.numbsampwalk)
                    
                    gdat.medipara = np.median(gdat.objtsamp.bdtrchain[gdat.numbsamp/2:, :], 0)
                    gdat.fittmasscomp.append(gdat.medipara[4])

                print('Making plots...')
            else:
                print('Plots have been made already at %s. Skipping...' % gdat.pathplottotl)
        
            boolposirele[k] = True
        else:
            boolposirele[k] = False
            
        # perform forward-modeling
        if gdat.boolmcmc:
            gdat.parapost = tesstarg.util.samp(gdat, gdat.pathimag, gdat.numbsampwalk, gdat.numbsampburnwalk, retr_modl, retr_lpos, \
                                               gdat.listlablpara, gdat.scalpara, gdat.minmpara, gdat.maxmpara, gdat.meanpara, gdat.stdvpara, gdat.numbdata)

            gdat.numbsamp = gdat.parapost.shape[0]
            gdat.indxsamp = np.arange(gdat.numbsamp)
            gdat.indxsampplot = gdat.indxsamp[::100]
            gdat.numbsampplot = gdat.indxsampplot.size
            gdat.postlcurmodl = np.empty((gdat.numbsampplot, gdat.numbtime))
            for ii, i in enumerate(gdat.indxsampplot):
                gdat.postlcurmodl[ii, :], temp, temp, temp = retr_modl(gdat, gdat.parapost[i, :])
        plot_datamodl(gdat)
        
        titl = 'Classified as '
        if boolreleposi[k]:
            titl += 'BHC candidate'
        else:
            titl += 'background'
        
        path = gdat.pathdata + 'lcur%04d.pdf' % k
        plot_lcur(path, gdat.lcurdata, titl=titl)
        
        path = gdat.pathdata + 'psdn%04d.pdf' % k
        plot_psdn(path, gdat.psdn, titl=titl)
        
        print('')
        
    gdat.indxlcurposi = np.array(gdat.indxlcurposi)
    gdat.fittperimaxm = np.array(gdat.fittperimaxm)
    gdat.fittmasscomp = np.array(gdat.fittmasscomp)
    
    if gdat.datatype == 'mock':
        gdat.booltrueposi = np.array(gdat.booltrueposi)
    
    # plot distributions
    numbbins = 10
    indxbins = np.arange(numbbins)
    if datatype == 'mock':
        listvarbtrue = [gdat.trueperi, gdat.truemasscomp, gdat.truetmag[gdat.indxtruerele]]
        listlablvarbtrue = ['P', 'M_c', 'Tmag']
        liststrgvarbtrue = ['trueperi', 'truemasscomp', 'truetmag']
    listvarb = [gdat.listsdee]
    listlablvarb = ['SNR']
    liststrgvarb = ['sdee']
    
    plot_recaprec(gdat.pathimag, gdat.datatype, gdat.thrssdee, gdat.boolpositrue, datatype=gdat.datatype, strgplotextn=gdat.strgplotextn)


def cnfg_HR6819():
   
    init( \
         liststrgmast=['HR6819'], \
        )


def cnfg_obsd():
   
    listisec = [9]
    init( \
         #listisec=listisec, \
         strgmast='Vela X-1', \
        )


def cnfg_mock():
   
    listisec = [9]
    init( \
         #listisec=listisec, \
         
         datatype='mock', \
        )


globals().get(sys.argv[1])()

