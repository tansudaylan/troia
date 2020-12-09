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


def retr_smaxfromperimasstotl(peri, masstotl):
    
    smax = (masstotl * 7.496e-6 * peri*2)**(1. / 3.) # [AU]

    return smax


def retr_amplslen(radistar, peri, masscomp, massstar):
    
    amplslen = 7.15e-5 * radistar**(-2.) * peri**(2. / 3.) * masscomp * (masscomp + massstar)**(1. / 3.)

    return amplslen


def retr_duraslen(radistar, peri, masscomp, massstar):
    
    duraslen = 1.8 * np.pi / 4. * peri**(1. / 3.) * (masscomp + massstar)**(-1. / 3.) * radistar

    return duraslen


def retr_llik_effe(para, gdat):
    
    radistar = para[0]
    peri = para[1]
    masscomp = para[2]
    massstar = para[3]
    
    amplslenmodl = retr_amplslen(radistar, peri, masscomp, massstar)
    duraslenmodl = retr_duraslen(radistar, peri, masscomp, massstar)
    
    modleffe = np.array([amplslenmodl, duraslenmodl, peri])
    
    #print('para')
    #print(para)
    #print('modleffe')
    #print(modleffe)
    #print('gdat.dataeffe')
    #print(gdat.dataeffe)
    #print('gdat.varidataeffe')
    #print(gdat.varidataeffe)
    #print('')

    llik = np.sum(-0.5 * (gdat.dataeffe - modleffe)**2 / gdat.varidataeffe)
    
    return llik


def retr_dictderi_effe(para, gdat):
    
    radistar = para[0]
    peri = para[1]
    masscomp = para[2]
    massstar = para[3]
    masstotl = massstar + masscomp

    amplslenmodl = retr_amplslen(radistar, peri, masscomp, massstar)
    duraslenmodl = retr_duraslen(radistar, peri, masscomp, massstar)
    smax = retr_smaxfromperimasstotl(peri, masstotl) * 215. # [R_S]
    radischw = 4.24e-6 * masscomp # [R_S]

    dictvarbderi = None

    dictparaderi = dict()
    dictparaderi['amplslenmodl'] = np.array([amplslenmodl])
    dictparaderi['duraslenmodl'] = np.array([duraslenmodl])
    dictparaderi['smaxmodl'] = np.array([smax])
    dictparaderi['radischw'] = np.array([radischw])

    return dictparaderi, dictvarbderi
    

def retr_masscomp( amplslen, peri):
    
    # temp
    masscomp = amplslen / 7.15e-5 / peri**(2. / 3.) * gdat.radistar**2. / (gdat.massstar)**(1. / 3.)
    
    return masscomp


def retr_dflxslensing(time, timeslen, amplslen, duraslen):
    
    dflxslensing = amplslen / np.sqrt(2. * np.pi) / duraslen * np.exp(-0.5 * (timeslen - time)**2 / duraslen**2)
    
    return dflxslensing


def retr_rflxcosc(gdat, time, para):
    
    # parse parameters 
    epoc = para[0]
    peri = para[1]
    radistar = para[2]
    masscomp = para[3]
    massstar = para[4]
   
    # temp -- this should change 1.89e-2
    densstar = massstar / radistar**3

    # phase
    phas = ((time - epoc) / peri) % 1.
    
    ## self-lensing
    ### duration
    duraslen = retr_duraslen(radistar, peri, masscomp, massstar) # [hour]
    ### amplitude
    amplslen = retr_amplslen(radistar, peri, masscomp, massstar)
    dflxslen = np.zeros_like(time)
    for k in range(10):
        dflxslen += retr_dflxslensing(time, epoc + peri * k, amplslen, duraslen)
    
    ## ellipsoidal variation
    amplelli = 1.89e-2 * peri**(-2.) / densstar * (1. / (1. + massstar / masscomp))
    dflxelli = -amplelli * np.cos(4. * np.pi * phas) 
    
    ## beaming
    amplbeam = 2.8e-3 * peri**(-1. / 3.) * (massstar + masscomp)**(-2. / 3.) * masscomp
    dflxbeam = -amplbeam * np.sin(2. * np.pi * phas)
    
    ## total relative flux
    rflxtotl = 1. + dflxslen + dflxelli + dflxbeam
    
    return rflxtotl, dflxelli + 1., dflxbeam + 1., dflxslen + 1.


def retr_llik(para, gdat):
    
    rflxtotl, dflxelli, dflxbeam, dflxslen = retr_rflxcosc(gdat, gdat.timethis, para)
    lpos = np.sum(-0.5 * (gdat.rflxbdtr - rflxmodl)**2 / gdat.varirflxbdtrthis)

    return lpos


def plot_psdn(gdat, n, perisamp, psdn, psdnelli=None, psdnbeam=None, psdnslen=None):
    
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
    path = gdat.pathtargimag[n] + 'psdn_%s.%s' % (gdat.strgextnthis, gdat.plotfiletype)
    plt.savefig(path)
    plt.close()
    

def plot_srch(gdat, n):
    
    titl = 'Classified as '
    if gdat.boolposi[n]:
        titl += 'BHC candidate'
    else:
        titl += 'background'
        
    if gdat.booltlsq:
        gdat.numbcols = 3
    else:
        gdat.numbcols = 2

    figr, axgr = plt.subplots(gdat.numbcols, 1, figsize=(10, 4.5))
    
    for k, axis in enumerate(axgr):
        
        if k == 0:
            if gdat.datatype == 'obsd':
                axis.text(0.5, 1.25, '%s' % (gdat.labltarg[n]), color='k', transform=axis.transAxes, ha='center')
            #axis.text(0.5, 1.15, 'S:%.3g, P:%.3g day, D:%.3g, T:%.3g day' % (gdat.listsdee[gdat.indxtargthis], \
            #                                                                        gdat.fittperimaxmthis, gdat.dept, gdat.dura), ha='center', \
            #                                                                                                transform=axis.transAxes, color='b')
            if gdat.datatype == 'mock':
                axis.text(0.5, 1.05, 'P=%.3g day, M=%.3g M$_\odot$, Tmag=%.3g' % (gdat.trueperi[gdat.indxtargthis], \
                                                                        gdat.truemasscomp[gdat.indxtargthis], gdat.truetmag[gdat.indxtargthis]), \
                                                                                                        transform=axis.transAxes, color='g', ha='center')
                axis.plot(gdat.time, gdat.truerflxtotl[:, gdat.indxtargthis], color='g', ls='-')
                #axis.plot(gdat.time, gdat.truedflxelli[:, gdat.indxtargthis], color='g', ls='--')
                #axis.plot(gdat.time, gdat.truedflxbeam[:, gdat.indxtargthis], color='g', ls=':')
                axis.plot(gdat.time, gdat.truedflxslen[:, gdat.indxtargthis], color='g', ls='-.')
                
            axis.scatter(gdat.time[n], gdat.rflxthis, color='black', s=1, rasterized=True)
            #axis.set_xlabel("Time [days]")
            axis.set_xticklabels([])
            axis.set_ylabel("Relative Flux")
            
        if k == 1:
            axis.scatter(gdat.timethis, gdat.rflxbdtr, color='black', s=1, rasterized=True)
            axis.set_xlabel("Time [days]")
            axis.set_ylabel("Detrended Relative Flux")
            if gdat.booltlsq:
                for k in range(len(gdat.dicttlsq['peri'])):
                    for n in range(-10, 10):
                        axis.axvline(gdat.dicttlsq['peri'] * n + gdat.dicttlsq['epoc'], color='orange', alpha=0.5, ls='--')
            if gdat.boolmcmc and gdat.booltrig[n]:
                axis.plot(gdat.timethis, gdat.postrflxmodl[n, :], color='b', alpha=0.1)
            axis.set_xlabel("Time [days]")
        
        if gdat.booltlsq:
            gdat.numbtlsq = len(gdat.dicttlsq['peri']) 
            gdat.indxtlsq = np.arange(gdat.numbtlsq)

        if k == 2 and gdat.booltlsq:
            for k in gdat.indxtlsq:
                axis.axvline(gdat.dicttlsq['peri'][k], alpha=0.5, color='b')
                for n in range(2, 10):
                    axis.axvline(n*gdat.dicttlsq['peri'][k], alpha=0.5, lw=1, linestyle="dashed", color='b')
                    axis.axvline(gdat.dicttlsq['peri'][k] / n, alpha=0.5, lw=1, linestyle="dashed", color='b')
            axis.set_ylabel(r'SDE')
            axis.set_xlabel('Period [days]')
            axis.plot(gdat.dicttlsq['listperi'], gdat.dicttlsq['powr'], color='black', lw=0.5)
            #axis.set_xlim([np.amin(gdat.peri), np.amax(gdat.peri)])
        
        if k == 3:
            axis.plot(gdat.phasmodl, gdat.pcurmodl, color='violet')
            gdat.fittrflxmodl, gdat.fittdflxelli, gdat.fittdflxbeam, gdat.fittamplslen = retr_rflxcosc(gdat, gdat.timethis, gdat.medipara)
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


def exec_srch(gdat, n):
    
    # baseline detrend
    print('Baseline detrending...')
    rflxbdtrregi, gdat.listindxtimeregi, gdat.indxtimeregioutt, gdat.listobjtspln, timeedge = \
                     tesstarg.util.bdtr_lcur(gdat.timethis, gdat.rflxthis, \
                                                verbtype=gdat.verbtype, \
                                                durabrek=gdat.durabrek, ordrspln=gdat.ordrspln, bdtrtype=gdat.bdtrtype)
    
    gdat.numbtime = gdat.timethis.size
    gdat.listarryrflxbdtr = np.zeros((gdat.numbtime, 3))
    gdat.listarryrflxbdtr[:, 0] = gdat.timethis
    gdat.listarryrflxbdtr[:, 1] = np.concatenate(rflxbdtrregi)
    gdat.rflxbdtr = gdat.listarryrflxbdtr[:, 1]
    numbsplnregi = len(rflxbdtrregi)
    gdat.indxsplnregi = np.arange(numbsplnregi)
    gdat.indxtime = np.arange(gdat.numbtime)

    # mask out the edges
    #durabrek = 0.5
    #timeedge = tesstarg.util.retr_timeedge(gdat.timethis, gdat.rflxbdtr, durabrek)
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
    #gdat.rflxbdtr = gdat.rflxbdtr[gdat.listindxtimegoodedge]
    #gdat.varirflxbdtrthis = gdat.varirflxthis[gdat.listindxtimegoodedge]
    
    if gdat.booltlsq:
        print('Performing TLS on %s...' % gdat.labltarg[n])
        arry = np.zeros((gdat.numbtime, 3))
        arry[:, 0] = gdat.timethis
        arry[:, 1] = 2. - gdat.rflxbdtr
        gdat.dicttlsq = tesstarg.util.exec_tlsq(arry, gdat.pathimag, thrssdee=gdat.thrssdee, strgextn=gdat.strgextnthis)#, ticitarg=ticitarg)
    
    if gdat.booltmpt:
        corr, listindxtimeposimaxm, timefull, rflxfull = tesstarg.util.corr_tmpt(gdat.timethis, gdat.rflxthis, gdat.listtimetmpt, gdat.listdflxtmpt, \
                                                                                thrs=gdat.thrstmpt, boolanim=gdat.boolanimtmpt, boolplot=gdat.boolplottmpt, \
                                                                            verbtype=gdat.verbtype, strgextn=gdat.strgextnthis, pathimag=gdat.pathtargimag)
        #find_bump(gdat.timethis, gdat.rflxthis, verbtype=1, strgextn='', numbduraslentmpt=3, minmduraslentmpt=None, maxmduraslentmpt=None, \
        #                                                            pathimag=None, boolplot=True, boolanim=False, gdat.thrstmpt=None)



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


def infe_para():
    
    # construct global object
    gdat = tdpy.util.gdatstrt()
    
    gdat.bfitperi = 4.25 # [days]
    gdat.stdvperi = 1e-2 * gdat.bfitperi # [days]
    gdat.bfitduraslen = 0.45 * 24. # [hours]
    gdat.stdvduraslen = 1e-1 * gdat.bfitduraslen # [hours]
    gdat.bfitamplslen = 0.14 # [relative]
    gdat.stdvamplslen = 1e-1 * gdat.bfitamplslen # [relative]
    
    listlablpara = [['$R_s$', 'R$_{\odot}$'], ['$P$', 'days'], ['$M_c$', 'M$_{\odot}$'], ['$M_s$', 'M$_{\odot}$']]
    listlablparaderi = [['$A$', ''], ['$D$', 'hours'], ['$a$', 'R$_{\odot}$'], ['$R_{Sch}$', 'R$_{\odot}$']]
    listminmpara = np.array([ 0.01, 0.1, 1e3, 1e-5])
    listmaxmpara = np.array([ 1e4, 100., 1e8, 1e3])
    #listlablpara += [['$M$', '$M_E$'], ['$T_{0}$', 'BJD'], ['$P$', 'days']]
    #listminmpara = np.concatenate([listminmpara, np.array([ 10., minmtime,  50.])])
    #listmaxmpara = np.concatenate([listmaxmpara, np.array([1e4, maxmtime, 200.])])
    listmeangauspara = None
    liststdvgauspara = None
    numbpara = len(listlablpara)
    indxpara = np.arange(numbpara)
    listscalpara = ['self' for k in indxpara]
    
    numbsampwalk = 10000
    numbsampburnwalk = 10000
    numbsampburnwalkseco = 5000
   
    gdat.dataeffe = np.array([gdat.bfitamplslen, gdat.bfitduraslen, gdat.bfitperi])
    gdat.varidataeffe = np.array([gdat.stdvamplslen, gdat.stdvduraslen, gdat.stdvperi])**2
    
    ## define paths
    gdat.pathbase = os.environ['BHOL_DATA_PATH'] + '/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathimag = gdat.pathbase + 'imag/'
    gdat.pathimageffe = gdat.pathimag + 'effe/'
    os.system('mkdir -p %s' % gdat.pathimageffe)
    numbdata = gdat.dataeffe.size

    strgextn = 'effe'
    listpara, null = tdpy.mcmc.samp(gdat, gdat.pathimageffe, numbsampwalk, numbsampburnwalk, numbsampburnwalkseco, retr_llik_effe, \
                                    listlablpara, listscalpara, listminmpara, listmaxmpara, listmeangauspara, liststdvgauspara, numbdata, strgextn=strgextn, \
                                    retr_dictderi=retr_dictderi_effe, boolpool=False, listlablparaderi=listlablparaderi)
   
    return


def init( \
        datatype='obsd', \
        
        # data input
        listticitarg=None, \
        liststrgmast=None, \
        isec=None, \
        
        # method, mfil or tlsq
        strgmeth='tlsq', \
        
        # baseline detrending
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
    if gdat.datatype == 'mocksimp':
        if gdat.liststrgmast is not None or gdat.isec is not None:
            raise Exception('')
    else:
        if gdat.liststrgmast is None and gdat.isec is None:
            raise Exception('')
        if gdat.liststrgmast is not None and gdat.isec is not None:
            raise Exception('')
    
    if gdat.datatype == 'mocksimp':
        gdat.targtype = 'mock'
    else:
        if gdat.liststrgmast is not None:
            gdat.targtype = 'mast'
        if gdat.isec is not None:
            gdat.targtype = 'sect'
    print('gdat.targtype')
    print(gdat.targtype)
    
    print('BHOL initialized at %s...' % gdat.strgtimestmp)

    # paths
    ## read PCAT path environment variable
    gdat.pathbase = os.environ['BHOL_DATA_PATH'] + '/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathimag = gdat.pathbase + 'imag/'
    
    # settings
    ## seed
    np.random.seed(0)
    
    ## plotting
    gdat.plotfiletype = 'pdf'
    
    gdat.boolanimtmpt = False
    gdat.boolplottmpt = True

    ## data
    ### observed data
    #### be agnostic about the source of data
    gdat.strgdata = None
    #### Boolean flag to use SAP data (as oppsed to PDC SAP)
    gdat.boolsapp = False
    # mock data
    if gdat.datatype.startswith('mock'):
        gdat.cade = 10. / 60. / 24. # days
   
    ## number of targets
    if gdat.datatype == 'obsd':
        gdat.numbtarg = len(gdat.liststrgmast)
    if gdat.datatype != 'obsd':
        gdat.numbtarg = 5
        gdat.numbtrue = gdat.numbtarg
    print('Number of targets: %s' % gdat.numbtarg)
    if gdat.numbtarg == 0:
        return
    gdat.indxtarg = np.arange(gdat.numbtarg)
    
    if gdat.datatype.startswith('mock'):
        gdat.indxtrue = gdat.indxtarg
    
    gdat.strgtarg = [[] for n in gdat.indxtarg]
    gdat.labltarg = [[] for n in gdat.indxtarg]
    gdat.pathtargdata = [[] for n in gdat.indxtarg]
    gdat.pathtargimagmcmc = [[] for n in gdat.indxtarg]
    gdat.pathtargimag = [[] for n in gdat.indxtarg]
    for n in gdat.indxtarg:
        if gdat.datatype == 'obsd':
            if gdat.targtype == 'sect':
                gdat.strgtarg[n] = 'sc%02d_%12d' % (gdat.isec, gdat.listticitarg[n])
                gdat.labltarg[n] = 'Sector %02d, TIC %d' % (gdat.isec, gdat.listticitarg[n])
            if gdat.targtype == 'mast':
                gdat.labltarg[n] = gdat.liststrgmast[n]
                gdat.strgtarg[n] = ''.join(gdat.liststrgmast[n].split(' '))
        if gdat.datatype != 'obsd':
            gdat.strgtarg[n] = 'mock%04d' % n
            gdat.labltarg[n] = 'Mock target %08d' % n
        gdat.pathtarg = gdat.pathbase + '%s/' % gdat.strgtarg[n]
        gdat.pathtargdata[n] = gdat.pathtarg + 'data/'
        gdat.pathtargimag[n] = gdat.pathtarg + 'imag/'
        gdat.pathtargimagmcmc[n] = gdat.pathtargimag[n] + 'mcmc/'
        os.system('mkdir -p %s' % gdat.pathtargdata[n])
        os.system('mkdir -p %s' % gdat.pathtargimag[n])
        
        print('gdat.strgtarg[n]')
        print(gdat.strgtarg[n])
        print('gdat.labltarg[n]')
        print(gdat.labltarg[n])
        
    # get data
    gdat.time = [[] for n in gdat.indxtarg]
    gdat.rflx = [[] for n in gdat.indxtarg]
    if gdat.datatype == 'obsd':
        gdat.varirflx = [[] for n in gdat.indxtarg]
        for n in gdat.indxtarg:
            datatype, arryrflx, arryrflxsapp, arryrflxpdcc, listarryrflx, listarryrflxsapp, listarryrflxpdcc, listisec, listicam, listiccd = \
                                                tesstarg.util.retr_data(gdat.strgdata, gdat.liststrgmast[n], gdat.pathdata, gdat.boolsapp)
            gdat.time[n] = arryrflx[:, 0]
            gdat.rflx[n] = arryrflx[:, 1]
            gdat.varirflx[n] = arryrflx[:, 2]**2
            gdat.cade = np.amin(gdat.time[n][1:] - gdat.time[n][:-1])
            
    ## analysis
    gdat.booltlsq = False
    gdat.booltmpt = True
    gdat.boolmcmc = True
    ### TLS
    #### SDE threshold
    gdat.thrssdee = 7.1

    ### template matching
    if gdat.booltmpt:
        gdat.thrstmpt = None
        
        minmduraslentmpt = 0.5
        maxmduraslentmpt = 24.
        
        numbduraslentmpt = 3
        indxduraslentmpt = np.arange(numbduraslentmpt)
        listduraslentmpt = np.linspace(minmduraslentmpt, maxmduraslentmpt, numbduraslentmpt)
        
        listcorr = []
        gdat.listdflxtmpt = [[] for k in indxduraslentmpt]
        gdat.listtimetmpt = [[] for k in indxduraslentmpt]
        numbtimekern = np.empty(numbduraslentmpt, dtype=int)
        for k in indxduraslentmpt:
            numbtimekern[k] = listduraslentmpt[k] / gdat.cade
            if numbtimekern[k] == 0:
                print('gdat.cade')
                print(gdat.cade)
                print('listduraslentmpt[k]')
                print(listduraslentmpt[k])
                raise Exception('')
            gdat.listtimetmpt[k] = np.arange(numbtimekern[k]) * gdat.cade
            timeslen = gdat.listtimetmpt[k][int(numbtimekern[k]/2)]
            amplslen = 1.
            gdat.listdflxtmpt[k] = retr_dflxslensing(gdat.listtimetmpt[k], timeslen, amplslen, listduraslentmpt[k])
            if not np.isfinite(gdat.listdflxtmpt[k]).all():
                raise Exception('')
        
    ### MCMC
    if gdat.boolmcmc:
        gdat.listlablpara = [['T$_0$', 'day'], ['P', 'day'], ['M', r'M$_s$']]
        gdat.numbtargwalk = 1000
        gdat.numbtargburnwalk = 100

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

    # baseline detrending
    gdat.numbtimefilt = int(round(5. / 24. / gdat.cade))
    if gdat.numbtimefilt % 2 == 0:
        gdat.numbtimefilt += 1
    print('gdat.numbtimefilt')
    print(gdat.numbtimefilt)

    print('gdat.cade [min]')
    print(gdat.cade * 24. * 60.)
    
    gdat.boolwritplotover = True

    # to be done by pexo
    ## target properties
    #gdat.radistar = 11.2
    #gdat.massstar = 18.
    gdat.radistar = 1.
    gdat.massstar = 1.
    gdat.densstar = 1.41

    if gdat.isec is not None:
        strgsecc = '%02d%d%d' % (gdat.isec, gdat.icam, gdat.iccd)
        print('Sector: %d' % gdat.isec)
        print('Camera: %d' % gdat.icam)
        print('CCD: %d' % gdat.iccd)
    
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
        amplslentmag = retr_amplslen(radistar, peri, masscomp, massstar)
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
    path = gdat.pathimag + 'sigm.%s' % (gdat.plotfiletype) 
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
    
    #gdat.claspred = np.choice(gdat.indxclastrue, size=numbsamp)
    
    gdat.boolposi = np.empty(gdat.numbtarg, dtype=bool)
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
        #gdat.rflx = gdat.rflx[listindxtimegood]
    
    if gdat.datatype != 'obsd':
        
        # mock data setup 
        if gdat.datatype == 'mocksimp':
            gdat.minmtime = 0.
            gdat.maxmtime = 27.3
            for nn, n in enumerate(gdat.indxtrue):
                gdat.time[n] = np.concatenate((np.arange(0., 27.3 / 2. - 1., gdat.cade), np.arange(27.3 / 2. + 1., 27.3, gdat.cade)))
            gdat.indxtimelink = np.where(abs(gdat.time[n] - 13.7) < 2.)[0]
    
        gdat.numbtime = gdat.time[n].size
        
        gdat.numbclastrue = 2
        gdat.indxclastrue = np.arange(gdat.numbclastrue)
        gdat.clastrue = np.random.choice(gdat.indxclastrue, size=gdat.numbtarg)
        gdat.indxtrueflat = np.where(gdat.clastrue == 0)[0]
        gdat.indxtrueslen = np.where(gdat.clastrue == 1)[0]
        gdat.numbtrueslen = gdat.indxtrueslen.size
        
        # generate mock data
        gdat.numbparatrueslen = 5
        gdat.paratrueslen = np.empty((gdat.numbtrueslen, gdat.numbparatrueslen))
        
        gdat.trueepoc = np.random.rand(gdat.numbtrueslen) * gdat.maxmtime
        
        gdat.trueminmperi = 4.5
        gdat.truemaxmperi = 4.5
        gdat.trueperi = np.random.rand(gdat.numbtrueslen) * (gdat.truemaxmperi - gdat.trueminmperi) + gdat.trueminmperi
        
        gdat.trueminmradistar = 660.
        gdat.truemaxmradistar = 660.
        gdat.trueradistar = np.random.random(gdat.numbtrueslen) * (gdat.truemaxmradistar - gdat.trueminmradistar) + gdat.trueminmradistar
        
        gdat.trueminmmasscomp = 2.4e6
        gdat.truemaxmmasscomp = 2.4e6
        gdat.truemasscomp = np.random.random(gdat.numbtrueslen) * (gdat.truemaxmmasscomp - gdat.trueminmmasscomp) + gdat.trueminmmasscomp
        
        gdat.trueminmmassstar = 100.
        gdat.truemaxmmassstar = 100.
        gdat.truemassstar = np.random.random(gdat.numbtrueslen) * (gdat.truemaxmmassstar - gdat.trueminmmassstar) + gdat.trueminmmassstar
        
        gdat.truerflxtotl = np.empty((gdat.numbtime, gdat.numbtrueslen))
        gdat.truedflxelli = np.empty((gdat.numbtime, gdat.numbtrueslen))
        gdat.truedflxbeam = np.empty((gdat.numbtime, gdat.numbtrueslen))
        gdat.truedflxslen = np.empty((gdat.numbtime, gdat.numbtrueslen))
        
        gdat.paratrueslen[:, 0] = gdat.trueepoc
        gdat.paratrueslen[:, 1] = gdat.trueperi
        gdat.paratrueslen[:, 2] = gdat.trueradistar
        gdat.paratrueslen[:, 3] = gdat.truemasscomp
        gdat.paratrueslen[:, 4] = gdat.truemassstar
                
        gdat.trueminmtmag = 8.
        gdat.truemaxmtmag = 16.
        gdat.truetmag = np.random.random(gdat.numbtarg) * (gdat.truemaxmtmag - gdat.trueminmtmag) + gdat.trueminmtmag

        ## flat target
        for nn, n in enumerate(gdat.indxtrueflat):
            gdat.rflx[n] = np.ones_like(gdat.time[n])
            
        ## self-lensing targets
        for nn, n in enumerate(gdat.indxtrueslen):
            gdat.truerflxtotl[:, nn], gdat.truedflxelli[:, nn], gdat.truedflxbeam[:, nn], gdat.truedflxslen[:, nn] = retr_rflxcosc(gdat, \
                                                                                                                  gdat.time[n], gdat.paratrueslen[nn, :])
            gdat.rflx[n] = np.copy(gdat.truerflxtotl[:, nn])
            
        for n in gdat.indxtarg:
            # add noise
            gdat.truestdvrflx = objtspln(gdat.truetmag)
            gdat.rflx[n] += gdat.truestdvrflx[n] * np.random.randn(gdat.numbtime)
            
            # determine data variance
            gdat.varirflx = gdat.truestdvrflx[n]**2 * np.ones_like(gdat.rflx[n])
            
        for nn, n in enumerate(gdat.indxtrueslen):
            # plot
            dictmodl = dict()
            dictmodl['modltotl'] = {'lcur': gdat.truerflxtotl[:, nn], 'time': gdat.time[n]}
            dictmodl['modlelli'] = {'lcur': gdat.truedflxelli[:, nn], 'time': gdat.time[n]}
            dictmodl['modlbeam'] = {'lcur': gdat.truedflxbeam[:, nn], 'time': gdat.time[n]}
            dictmodl['modlslen'] = {'lcur': gdat.truedflxslen[:, nn], 'time': gdat.time[n]}
            strgextn = '%s_%s' % (gdat.datatype, gdat.strgtarg[n])
            titl = ''
            tesstarg.util.plot_lcur(gdat.pathtargimag[n], dictmodl=dictmodl, timedata=gdat.time[n], lcurdata=gdat.rflx[n], boolover=gdat.boolwritplotover, \
                                                                                                                            strgextn=strgextn, titl=titl)
            
        delttime = 1. / 24. / 2.
        fs = 1. / delttime
        
        # histogram
        path = gdat.pathimag + 'truestdvrflx'
        tdpy.mcmc.plot_hist(path, gdat.truestdvrflx, r'$\sigma$', strgplotextn=gdat.plotfiletype)
        path = gdat.pathimag + 'truetmag'
        tdpy.mcmc.plot_hist(path, gdat.truetmag, 'Tmag', strgplotextn=gdat.plotfiletype)
            
    pathlogg = gdat.pathdata + 'logg/'
    
    gdat.listsdee = np.empty(gdat.numbtarg)
    gdat.indxtargposi = []
    
    gdat.booltrig = np.empty(gdat.numbtarg, dtype=bool)
    for n in gdat.indxtarg:
        
        gdat.indxtargthis = n
        gdat.timethis = gdat.time[n]
        gdat.rflxthis = gdat.rflx[n]
        gdat.varirflxthis = gdat.varirflx[n]
        
        delttime = np.amin(gdat.timethis[1:] - gdat.timethis[:-1])
        fs = 1. / delttime

        # check data for finiteness
        if (~np.isfinite(gdat.rflxthis)).any():
            print('gdat.rflxthis')
            summgene(gdat.rflxthis)
            raise Exception('')
        
        gdat.strgextnthis = '%s_%s' % (gdat.datatype, gdat.strgtarg[n])
        pathtcee = pathlogg + '%s_%s.txt' % (gdat.datatype, gdat.strgtarg[n])
        
        if gdat.datatype == 'mock':
            print('gdat.truetmag[n]')
            print(gdat.truetmag[n])
            print('gdat.truestdvrflx[n]')
            print(gdat.truestdvrflx[n])

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
            print('Reading %s...' % gdat.strgtarg[n])
            # log file
            filelogg = open(pathtcee, 'w+')
            exec_srch(gdat, n)
        
        else:
            print('BLS has already been done. Reading the log file for %s at %s...' % (gdat.strgtarg[n], pathtcee))
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
        exec_srch(gdat, n)

        # calculate PSD
        freq, gdat.psdn = scipy.signal.periodogram(gdat.rflxthis, fs=fs)
        perisamp = 1. / freq
        
        gdat.booltrig[n] = gdat.listsdee[n] >= gdat.thrssdee
        if gdat.booltrig[n]:
            
            gdat.indxtargposi.append(n)
            
            if gdat.datatype == 'mock':
                boolreleposi[gdat.indxtrueslen[k]] = True
            
            gdat.boolposi[n] = True
            if gdat.datatype == 'mock':
                boolposirele[gdat.indxtrueslen[k]] = True
            
            # temp
            if (True or not os.path.exists(gdat.pathtargimagmcmc[n])) and gdat.boolmcmc:
                print('Performing sampling on %s...' % gdat.labltarg[n])
                dictllik = [gdat]
                
                # perform forward-modeling
                if gdat.boolmcmc:
                    gdat.parapost = tdpy.mcmc.samp(gdat, gdat.pathimag, gdat.numbtargwalk, gdat.numbtargburnwalk, retr_rflxcosc, retr_lpos, \
                                           gdat.listlablpara, gdat.scalpara, gdat.minmpara, gdat.maxmpara, gdat.meanpara, gdat.stdvpara, gdat.numbdata)

                print('Making plots...')
                os.system('mkdir -p %s' % gdat.pathtargimagmcmc[n])
        
        else:
            gdat.boolposi[n] = False
            if gdat.datatype == 'mock':
                boolposirele[gdat.indxtrueslen[k]] = False
            
        gdat.pathplotsrch = gdat.pathtargimag[n] + 'srch_%s_%s.%s' % (gdat.datatype, gdat.strgtarg[n], gdat.plotfiletype)
        # temp
        if True or not os.path.exists(gdat.pathtargimagmcmc[n]):
            plot_srch(gdat, n)
        
        plot_psdn(gdat, n, perisamp, gdat.psdn)
        
        print('')
        
    gdat.indxtargposi = np.array(gdat.indxtargposi)
    
    # plot distributions
    if datatype == 'mock':
        listvarbreca = [gdat.trueperi, gdat.truemasscomp, gdat.truetmag[gdat.indxtrueslen]]
        listlablvarbreca = ['P', 'M_c', 'Tmag']
        liststrgvarbreca = ['trueperi', 'truemasscomp', 'truetmag']
    listvarbprec = [gdat.listsdee]
    listlablvarbprec = ['SNR']
    liststrgvarbprec = ['sdee']
    
    if datatype == 'mock':
        tdpy.util.plot_recaprec(gdat.pathimag, gdat.strgextn, listvarbreca, listvarbprec, liststrgvarbreca, liststrgvarbprec, \
                                listlablvarbreca, listlablvarbprec, boolposirele, boolreleposi)


def cnfg_HR6819():
   
    init( \
         liststrgmast=['HR6819'], \
         bdtrtype='medi', \
        )


def cnfg_Rafael():
   
    init( \
         liststrgmast=['TIC 356069146'], \
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
         datatype='mocksimp', \
        )


globals().get(sys.argv[1])()

