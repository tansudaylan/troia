from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import math
import mock_light_curves as mlc
import confusion_matrix as cf
import os
import scipy.signal
import time
import sqlite3

def mf_pipeline(directory, result_foldername, noise=.00005, mock=False, num_simulations=None, threshold=.05):
    '''
    Pipeline runs a match filter on light curves and finds if the light curve 
    matches a predetermined template.
    
    directory: string, directory where light curve .fits files are located
    result_foldername: string, location of resulting plots, data 
    mock: bool, true if using mock data, False if using real light curve
    mock_params: dict, data used for mock simulation
        - 'num_simulations': int, number of mock light curves to generate
        -
        
    returns: dict, results for each light curve (file)
    '''
    if mock:
# =============================================================================
#         conn = sqlite3.connect('{}.db'.format(result_foldername))  # You can create a new database by changing the name within the quotes
#         c = conn.cursor()
#         c.execute('''CREATE TABLE IF NOT EXISTS Data ([generated_id] INTEGER PRIMARY KEY,[Period] REAL, [M_BH] REAL, [M_S] REAL, [cosi] REAL, [Actual] INTEGER, [Predicted] INTEGER)''')
# =============================================================================
        kernel = 99
        num_bins = 10
        counter = 0
        
        # create folder for resulting plots
        if not os.path.exists(result_foldername):
            os.mkdir(result_foldername)
        
        # results array for completeness analysis of each variable
        i_actual = [[] for _ in range(num_bins)]
        i_predicted = [[] for _ in range(num_bins)]
        inclinations = [[] for _ in range(num_bins)]
        
        P_actual = [[] for _ in range(num_bins)]
        P_predicted = [[] for _ in range(num_bins)]
        periods = [[] for _ in range(num_bins)]
        
        mbh_actual = [[] for _ in range(num_bins)]
        mbh_predicted = [[] for _ in range(num_bins)]
        mbhs = [[] for _ in range(num_bins)]
        
        ms_actual = [[] for _ in range(num_bins)]
        ms_predicted = [[] for _ in range(num_bins)]
        mss = [[] for _ in range(num_bins)]
        
        total_actual = []
        total_predicted = []
        
        # generate bins
        cosi_bins = [j/num_bins * .01  for j in range(num_bins)]
        P_bins = [np.e**(j*(np.log(27)-np.log(1))/num_bins) for j in range(num_bins)]
        mbh_bins = [15*j/num_bins + 5 for j in range(num_bins)]
        ms_bins = [j/num_bins + 0.5 for j in range(num_bins)]
        
        # generate templates varying width from 10mins to 2hrs (5 bins to 60 bins)
        templates = []
        for width in [5*j + 5 for j in range(12)]: 
            templates.append(mlc.offset_and_normalize(mlc.generate_template(1, width)))
        
        start = time.time()
        for z in range(num_simulations):
            
            # randomly generate relevant parameters from known priors
            P = mlc.P_rng()
            M_BH = mlc.mbh_rng()
            M_S = mlc.ms_rng()
            i = mlc.i_rng()
            cosi = math.cos(i)
            
            # generate positive/negative signal at random
            pos_signal = np.random.choice([True, False])
            if pos_signal:
                lcur, EV, Beam, SL = mlc.generate_light_curve(P, i, M_BH, M_S=M_S, std=noise)
            else:
                lcur = mlc.generate_flat_signal(noise)
                
            # subtract median filter from signal and normalize for correlation analysis
            flat_lcur = lcur - scipy.signal.medfilt(lcur, kernel)
            flat_lcur = mlc.offset_and_normalize(flat_lcur)
            
            initial = True
            best_result = None
            best_corr = 0
            best_correlations = None
            best_template = None
            
            # perform cross-correlation for all template widths
            for template in templates:  
                correlations = scipy.signal.correlate(flat_lcur, template)
                highest_corr = max(correlations)
                result  = highest_corr > threshold
                    
                # choose best correlation result so far (break on positive signal detection)
                if initial or highest_corr > best_corr:
                    best_corr = highest_corr
                    best_result = result
                    best_correlations = correlations
                    best_template = template
                    initial = False
                    
                if result:
                    break
            
            # plot some light curves and their correlations at random based on prediction
            if (best_result != pos_signal and np.random.random() > .95) or (pos_signal and np.random.random() > .99):
                lc_folder = "./{}/lc{}".format(result_foldername, counter)
                if not os.path.exists(lc_folder):
                    os.mkdir(lc_folder)
                mlc.plot_lc(lcur, P, M_BH, i, M_S, filename="{}/lcur{}.pdf".format(lc_folder, counter), EV=EV if pos_signal else None, Beam=Beam if pos_signal else None, SL=SL if pos_signal else None)
                mlc.plot_lc(flat_lcur, P, M_BH, i, M_S, filename="{}/flat_lcur{}.pdf".format(lc_folder, counter))
                mlc.plot_corr(best_correlations, P, M_BH, i, M_S, threshold, "{}/corr{}.pdf".format(lc_folder, counter))
                counter += 1
            
            # bin result depending on parameter values
            if pos_signal or best_result:
                i_binned, P_binned, mbh_binned, ms_binned = False, False, False, False
                for k in range(num_bins-1,-1,-1):
                    if not i_binned and cosi >= cosi_bins[k]:
                        inclinations[k].append(i)
                        i_actual[k].append(pos_signal)
                        i_predicted[k].append(result)
                    if not P_binned and P >= P_bins[k]:
                        periods[k].append(i)
                        P_actual[k].append(pos_signal)
                        P_predicted[k].append(result)
                    if not mbh_binned and M_BH >= mbh_bins[k]:
                        mbhs[k].append(i)
                        mbh_actual[k].append(pos_signal)
                        mbh_predicted[k].append(result)
                    if not ms_binned and M_S >= ms_bins[k]:
                        mss[k].append(i)
                        ms_actual[k].append(pos_signal)
                        ms_predicted[k].append(result)
                    if all([i_binned, P_binned, mbh_binned, ms_binned]):
                        break
                    
            total_actual.append(pos_signal)
            total_predicted.append(best_result) 
            
# =============================================================================
#             c.execute(''' INSERT INTO Data(Period, M_BH, M_S, cosi, Actual, Predicted)
#               VALUES(?,?,?,?,?,?) ''', (P, M_BH, M_S, cosi, int(pos_signal), int(best_result)))      
# =============================================================================
            # perform completeness analysis
            if z%1000 == 0 and z != 0:
                #conn.commit()
                print('{} simulations complete'.format(z))
                prefix = "./{}/".format(result_foldername)
                #plot_completeness(r'$\alpha$', alpha_bins, alpha_actual, alpha_predicted, num_bins, prefix + 'alpha')
                plot_completeness('cosi', cosi_bins, i_actual, i_predicted, num_bins, prefix + 'cosi')
                plot_completeness('Period [days]', P_bins, P_actual, P_predicted, num_bins, prefix + 'period')
                plot_completeness(r'$M_{BH} [M_{\odot}]$', mbh_bins, mbh_actual, mbh_predicted, num_bins, prefix + 'mbh')
                plot_completeness(r'$M_{\star} [M_{\odot}]$', ms_bins, ms_actual, ms_predicted, num_bins, prefix + 'ms')
            
        #conn.close()
        end = time.time()
        print("{} minutes".format(round((end - start)/60, 2)))
        
        return total_actual, total_predicted
            
    else:
        alpha = 2
        if not os.path.exists(directory+result_foldername):
            os.mkdir(directory+result_foldername)
        file_results = {}
        for filename in os.listdir(directory):
            if filename.endswith(".fits"):
                fits_file = directory + filename
                
                try:
                    with fits.open(fits_file, mode="readonly", memmap=False) as hdulist:
                        tess_bjds = hdulist[1].data['TIME']
                        #sap_fluxes = hdulist[1].data['SAP_FLUX']
                        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
                        #var = hdulist[1].data['PSF_CENTR1']
                except:
                    continue
                
                #num_days = len(tess_bjds)
                #bin_size = (tess_bjds[-1] - tess_bjds[0]) / num_days
                # Find average flux and rms from valid data points
                total = 0
                total_square = 0
                x = 0
                valid_times = []
                valid_fluxes = []
                for i in range(len(pdcsap_fluxes)):
                    flux = pdcsap_fluxes[i]
                    if not np.isnan(flux):
                        total += flux
                        total_square += flux**2
                        x += 1
                        valid_times.append(tess_bjds[i])
                        valid_fluxes.append(pdcsap_fluxes[i])
                        
                average = total/x
                rms = (total_square/x)**.5
                
                # normalize light curve
                
                norm_fluxes = (valid_fluxes - average)/rms
                        
                #subtract median filter to get high frequency noise
                flat = norm_fluxes - scipy.signal.medfilt(norm_fluxes, 99)
                
                
                std = np.std(flat)
                amp = 4*std
                initial = True
                best_ratio = 0
                best_results = None
                
                for width in [30 + 5*j for j in range(13)]:  # try widths from 10 mins to 2hrs (5 bins to 60 bins)
                    template = mlc.generate_template(amp, width)
                    threshold = amp**2 * width / alpha
                    mf_results = mlc.match_filter(flat, template, threshold=threshold)
                    if initial or mf_results["highest_corr"]/mf_results["threshold"] > best_ratio:
                        best_ratio = mf_results["highest_corr"]/mf_results["threshold"]
                        best_results = mf_results
                        initial = False
                    if mf_results["result"]:
                        break
                
                file_results[filename] = best_results 
                
                if mf_results["result"] or np.random.random() > .9: # get plots for positive results 
                    folder = '{}{}/{}'.format(directory, result_foldername, filename[:-5])
                    if not os.path.exists(folder):
                        os.mkdir(folder)
                    # plot light curve 
                    plt.figure()
                    plt.xlabel('Time [days]')
                    plt.ylabel('PDCSAP Flux')
                    plt.plot(valid_times, valid_fluxes, 'ko', rasterized=True)
                    plt.tight_layout()
                    graph_filename = '{}/light_curve.pdf'.format(folder)
                    plt.savefig(graph_filename)
                    plt.close()
                    
                    # plot flat light curve
                    plt.figure()
                    plt.xlabel('Time [days]')
                    plt.ylabel('Relative Flux')
                    plt.plot(valid_times, flat, 'ko', rasterized=True)
                    plt.tight_layout()
                    graph_filename = '{}/flat_lcur.pdf'.format(folder)
                    plt.savefig(graph_filename)
                    plt.close()
                    
                    # plot correlations
                    window = len(best_results["template"])
                    plt.figure()
                    plt.xlabel('Time [days]')
                    plt.ylabel('Correlation')
                    plt.plot(valid_times[:-window], best_results["correlations"], 'k', rasterized=True)
                    plt.plot(valid_times[:-window], [best_results["threshold"] for _ in range(len(best_results["correlations"]))], '--', color='orange', rasterized=True)
                    plt.tight_layout()
                    graph_filename = '{}/correlations.pdf'.format(folder)
                    plt.savefig(graph_filename)
                    plt.close()
                
        return file_results
    
def plot_completeness(variable, bins, actual, predicted, num_bins, path_prefix, scale=False):
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
    plt.xlabel(variable)
    plt.ylabel('Accuracy')
    if scale:
        plt.plot([b for b in bins], accs, 'k', rasterized=True)
    else:
        plt.plot([str(round(b, 2)) for b in bins], accs, 'k', rasterized=True)
    filename = '{}_accuracy.pdf'.format(path_prefix, variable)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    #plot precision
    plt.figure()
    plt.xlabel(variable)
    plt.ylabel('Precision')
    if scale:
        plt.plot([b for b in bins], pres, 'k', rasterized=True)
    else:
        plt.plot([str(round(b, 2)) for b in bins], pres, 'k', rasterized=True)
    filename = '{}_precision.pdf'.format(path_prefix, variable)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    # plot recall
    plt.figure()
    plt.xlabel(variable)
    plt.ylabel('Recall')
    if scale:
        plt.plot([b for b in bins], recs, 'k', rasterized=True)
    else:
        plt.plot([str(round(b, 2)) for b in bins], recs, 'k', rasterized=True)
    filename = '{}_recall.pdf'.format(path_prefix, variable)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    # plot F1
    plt.figure()
    plt.xlabel(variable)
    plt.ylabel('F1')
    if scale:
        plt.plot([b for b in bins], F1s, 'k', rasterized=True)
    else:
        plt.plot([str(round(b, 2)) for b in bins], F1s, 'k', rasterized=True)
    filename = '{}_F1.pdf'.format(path_prefix, variable)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def noise_test(result_foldername):
    if not os.path.exists(result_foldername):
        os.mkdir(result_foldername)
    noise_values = np.array([0,50,100,150,200,300,400,500,750,1000]) 
    altered_noise = noise_values * 10**(-6) + 10**(-9)
    actuals = []
    predicteds = []
    for noise in altered_noise:
        actual, predicted = mf_pipeline(None, result_foldername, noise=noise, mock=True, num_simulations=1000)
        actuals.append(actual)
        predicteds.append(predicted)
    
    plot_completeness('Noise [ppm]', noise_values, actuals, predicteds, 10, "./{}/noise".format(result_foldername), scale=True)
    return actuals, predicteds

def alpha_test(result_foldername):
    if not os.path.exists(result_foldername):
        os.mkdir(result_foldername)
    alpha_values = np.array([.005*i for i in range(1, 11)]) 
    actuals = []
    predicteds = []
    for alpha in alpha_values:
        actual, predicted = mf_pipeline(None, result_foldername, mock=True, num_simulations=1000, threshold=alpha)
        actuals.append(actual)
        predicteds.append(predicted)
    
    plot_completeness('Noise [ppm]', alpha_values, actuals, predicteds, 10, "./{}/alpha".format(result_foldername), scale=True)
    return actuals, predicteds
    