import math
import numpy as np
import matplotlib.pyplot as plt
import mock_light_curves as mlc
import confusion_matrix as cf

def generate_flat_signal(size, noise):
    return np.array([np.random.normal(0, noise) for _ in range(size)])

def P_rng(low=1, high=27):
    logP = np.random.uniform(np.log(low),np.log(high))
    P = np.e**logP
    return P

def mbh_rng():
    low = 5
    mbh = np.random.normal(7.8, 0.6)
    if mbh < low:
        mbh = low
    return mbh

def i_rng():
    cosi  = np.random.random()*2 - 1
    i = math.acos(cosi) + np.random.choice([0,np.pi])
    return i
        
def P_simulation(num_simulations, noise, num_bins):
    actual = [[] for _ in range(num_bins)]
    predicted = [[] for _ in range(num_bins)]
    
    periods = [[] for _ in range(num_bins)]
    # generate logarithmic bins
    bins = [np.e**(j*(np.log(27)-np.log(1))/num_bins) for j in range(num_bins)]
    # find a set M_BH 
    M_BH = 10
    # set other parameters
    i, M_S, R_S, rho_S = math.pi/2, 1, 1, 1.41
    
    for _ in range(num_simulations):
        # randomly select P
        P = P_rng()
        # fairly choose whether to generate positive/negative signal 
        lc_pos = np.random.choice([True, False])
        #generate light curve
        if lc_pos:
            lc, EV, Beam, SL = mlc.generate_light_curve(P, i, M_BH, M_S, R_S, rho_S, std=noise)
        else:
            lc, EV, Beam, SL = mlc.generate_light_curve(P, i, M_BH, M_S, R_S, rho_S, pos=False, std=noise)
        
        ss_lc = mlc.supersample(lc, P, i, M_BH)
        flat_lc = mlc.flatten(ss_lc, P, i, M_BH, M_S, R_S, rho_S)
        result, corrs, template, threshold = mlc.match_filter(flat_lc, None, P, i, M_BH, M_S, R_S, rho_S, mock=True)
        
        if result != lc_pos and np.random.random() > .9:    # plot all light curves that were labeled incorrectly
            mlc.plot_lc(lc, P, M_BH, i, filename="P" + str((round(P,2))) + "_MLC.pdf", EV=EV, Beam=Beam, SL=SL if lc_pos else None)
            mlc.plot_lc(ss_lc, P, M_BH, i, filename="P" + str((round(P,2))) + "_MLCS.pdf", EV=EV, Beam=Beam, SL=SL if lc_pos else None)
            mlc.plot_lc(flat_lc, P, M_BH, i, filename="P" + str((round(P,2))) + "_MLCF.pdf")
            mlc.plot_corr(corrs, P, M_BH, i, len(template), threshold)
            
        # place result into correct bin
        for k in range(num_bins-1,-1,-1):
            if P >= bins[k]:
                periods[k].append(P)
                actual[k].append(lc_pos)
                predicted[k].append(result)
                break
    
    accs = []
    pres = []
    recs = []
    F1s = []
    for l in range(num_bins):
        cm, acc, pre, rec, F1 = cf.confusion_matrix(actual[l], predicted[l])
        accs.append(acc)
        pres.append(pre)
        recs.append(rec)
        F1s.append(F1)
       
    # plot accuracy
    plt.figure()
    plt.xlabel('Orbital Period (days)')
    plt.ylabel('Accuracy')
    plt.plot([str(round(b, 1)) for b in bins], accs, 'k')
    filename = 'accuracy.pdf'
    plt.savefig(filename)
    plt.close()
    
    #plot precision
    plt.figure()
    plt.xlabel('Orbital Period (days)')
    plt.ylabel('Precision')
    plt.plot([str(round(b, 1)) for b in bins], pres, 'k')
    filename = 'precision.pdf'
    plt.savefig(filename)
    plt.close()
    
    # plot recall
    plt.figure()
    plt.xlabel('Orbital Period (days)')
    plt.ylabel('Recall')
    plt.plot([str(round(b, 1)) for b in bins], recs, 'k')
    filename = 'recall.pdf'
    plt.savefig(filename)
    plt.close()
    
    # plot F1
    plt.figure()
    plt.xlabel('Orbital Period (days)')
    plt.ylabel('F1')
    plt.plot([str(round(b, 1)) for b in bins], F1s, 'k')
    filename = 'F1.pdf'
    plt.savefig(filename)
    plt.close()
        
    return bins, accs, pres, recs, F1s

                    
def MBH_simulation(num_simulations, noise, num_bins):
    actual = [[] for _ in range(num_bins)]
    predicted = [[] for _ in range(num_bins)]
    masses = [[] for _ in range(num_bins)]
    # generate bins
    bins = None
    # find a set P
    P = 10
    # set other parameters
    i, M_S, R_S, rho_S = math.pi/2, 1, 1, 1.41
    
    for _ in range(num_simulations):
        # randomly select M_BH
        M_BH = mbh_rng()
        # fairly choose whether to generate positive/negative signal 
        lc_pos = np.random.choice([True, False])
        #generate light curve
        if lc_pos:
            lc = mlc.generate_light_curve(P, i, M_BH, M_S, R_S, rho_S, std=noise)
        else:
            lc = mlc.generate_light_curve(P, i, M_BH, M_S, R_S, rho_S, pos=False, std=noise)
        
        result, corrs = mlc.match_filter(lc, None, P, i, M_BH, M_S, R_S, rho_S, mock=True)
        
        # place result into correct bin
        for k in range(num_bins-1,-1,-1):
            if P >= bins[k]:
                masses[k].append(P)
                actual[k].append(lc_pos)
                predicted[k].append(result)
                break
                
    
def i_simulation(num_simulations, noise, num_bins):
    actual = [[] for _ in range(num_bins)]
    predicted = [[] for _ in range(num_bins)]
    inclinations = [[] for _ in range(num_bins)]
    # generate bins
    bins = [2*np.pi/num_bins*j for j in range(num_bins)]
    # find a set M_BH and P
    M_BH = 10
    P = 3
    # set other parameters
    M_S, R_S, rho_S = 1, 1, 1.41
        
    for _ in range(num_simulations):
        # randomly select cosi
        i = i_rng()
        # fairly choose whether to generate positive/negative signal 
        lc_pos = np.random.choice([True, False])
        #generate light curve
        if lc_pos:
            lc, EV, Beam, SL = mlc.generate_light_curve(P, i, M_BH, M_S, R_S, rho_S, std=noise)
        else:
            lc, EV, Beam, SL = mlc.generate_light_curve(P, i, M_BH, M_S, R_S, rho_S, pos=False, std=noise)
            
        ss_lc = mlc.supersample(lc)
        flat_lc = mlc.flatten(ss_lc, P, i, M_BH, M_S, R_S, rho_S)
        result, highest_corr, corrs, template, threshold = mlc.match_filter(flat_lc, None, P, i, M_BH, M_S, R_S, rho_S, mock=True)
            
        if result != lc_pos and np.random.random() > .5:    # plot all light curves that were labeled incorrectly
            mlc.plot_lc(lc, P, M_BH, i, filename="i" + str((round(i,2))) + "_MLC.pdf", EV=EV, Beam=Beam, SL=SL if lc_pos else None)
            mlc.plot_lc(ss_lc, P, M_BH, i, filename="i" + str((round(i,2))) + "_MLCS.pdf", EV=EV, Beam=Beam, SL=SL if lc_pos else None)
            mlc.plot_lc(flat_lc, P, M_BH, i, filename="i" + str((round(i,2))) + "_MLCF.pdf")
            mlc.plot_corr(corrs, P, M_BH, i, len(template), threshold, 'i' + str(round(i,2)) + 'CORR.pdf')
                
        # place result into correct bin
        for k in range(num_bins-1,-1,-1):
            if i >= bins[k]:
                inclinations[k].append(i)
                actual[k].append(lc_pos)
                predicted[k].append(result)
                break
        
    accs = []
    pres = []
    recs = []
    F1s = []
    for l in range(num_bins):
        cm, acc, pre, rec, F1 = cf.confusion_matrix(actual[l], predicted[l])
        accs.append(acc)
        pres.append(pre)
        recs.append(rec)
        F1s.append(F1)
           
    # plot accuracy
    plt.figure()
    plt.xlabel('Orbital Inclination (Radians)')
    plt.ylabel('Accuracy')
    plt.plot([str(round(b, 1)) for b in bins], accs, 'k')
    filename = 'accuracy.pdf'
    plt.savefig(filename)
    plt.close()
    
    #plot precision
    plt.figure()
    plt.xlabel('Orbital Inclination (Radians)')
    plt.ylabel('Precision')
    plt.plot([str(round(b, 1)) for b in bins], pres, 'k')
    filename = 'precision.pdf'
    plt.savefig(filename)
    plt.close()
    
    # plot recall
    plt.figure()
    plt.xlabel('Orbital Inclination (Radians)')
    plt.ylabel('Recall')
    plt.plot([str(round(b, 1)) for b in bins], recs, 'k')
    filename = 'recall.pdf'
    plt.savefig(filename)
    plt.close()
    
    # plot F1
    plt.figure()
    plt.xlabel('Orbital Inclination (Radians)')
    plt.ylabel('F1')
    plt.plot([str(round(b, 1)) for b in bins], F1s, 'k')
    filename = 'F1.pdf'
    plt.savefig(filename)
    plt.close()
        
    return bins, accs, pres, recs, F1s

def alpha_simulation(num_simulations, noise, num_bins):
    actual = [[] for _ in range(num_bins)]
    predicted = [[] for _ in range(num_bins)]
    alphas = [[] for _ in range(num_bins)]
    # generate bins
    bins = [j/(2*num_bins) for j in range(num_bins)]
    # find a set M_BH and P
    M_BH = mbh_rng()
    P = P_rng()
    i = i_rng()
    # set other parameters
    M_S, R_S, rho_S = 1, 1, 1.41
        
    for _ in range(num_simulations):
        # randomly select alpha
        alpha = np.random.random()/2
        # fairly choose whether to generate positive/negative signal 
        lc_pos = np.random.choice([True, False])
        #generate light curve
        if lc_pos:
            lc, EV, Beam, SL = mlc.generate_light_curve(P, i, M_BH, M_S, R_S, rho_S, std=noise)
        else:
            lc, EV, Beam, SL = mlc.generate_light_curve(P, i, M_BH, M_S, R_S, rho_S, pos=False, std=noise)
            
        ss_lc = mlc.supersample(lc)
        flat_lc = mlc.flatten(ss_lc, P, i, M_BH, M_S, R_S, rho_S)
        result, highest_corr, corrs, template, threshold = mlc.match_filter(flat_lc, None, P, i, M_BH, M_S, R_S, rho_S, mock=True, alpha=alpha)
            
        if result != lc_pos and np.random.random() > .75:    # plot all light curves that were labeled incorrectly
            mlc.plot_lc(lc, P, M_BH, i, filename="alpha" + str((round(alpha,2))) + "_MLC.pdf", EV=EV, Beam=Beam, SL=SL if lc_pos else None)
            mlc.plot_lc(ss_lc, P, M_BH, i, filename="alpha" + str((round(alpha,2))) + "_MLCS.pdf", EV=EV, Beam=Beam, SL=SL if lc_pos else None)
            mlc.plot_lc(flat_lc, P, M_BH, i, filename="alpha" + str((round(alpha,2))) + "_MLCF.pdf")
            mlc.plot_corr(corrs, P, M_BH, i, len(template), threshold, 'alpha' + str(round(alpha,2)) + 'CORR.pdf')
                
        # place result into correct bin
        for k in range(num_bins-1,-1,-1):
            if alpha >= bins[k]:
                alphas[k].append(i)
                actual[k].append(lc_pos)
                predicted[k].append(result)
                break
        
    accs = []
    pres = []
    recs = []
    F1s = []
    for l in range(num_bins):
        cm, acc, pre, rec, F1 = cf.confusion_matrix(actual[l], predicted[l])
        accs.append(acc)
        pres.append(pre)
        recs.append(rec)
        F1s.append(F1)
           
    # plot accuracy
    plt.figure()
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.plot([str(round(b, 2)) for b in bins], accs, 'k')
    filename = 'accuracy.pdf'
    plt.savefig(filename)
    plt.close()
    
    #plot precision
    plt.figure()
    plt.xlabel('Alpha')
    plt.ylabel('Precision')
    plt.plot([str(round(b, 2)) for b in bins], pres, 'k')
    filename = 'precision.pdf'
    plt.savefig(filename)
    plt.close()
    
    # plot recall
    plt.figure()
    plt.xlabel('Alpha')
    plt.ylabel('Recall')
    plt.plot([str(round(b, 2)) for b in bins], recs, 'k')
    filename = 'recall.pdf'
    plt.savefig(filename)
    plt.close()
    
    # plot F1
    plt.figure()
    plt.xlabel('Alpha')
    plt.ylabel('F1')
    plt.plot([str(round(b, 2)) for b in bins], F1s, 'k')
    filename = 'F1.pdf'
    plt.savefig(filename)
    plt.close()
        
    return bins, accs, pres, recs, F1s