from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import math
import mock_light_curves as mlc
import confusion_matrix as cf
import os
import scipy.signal
import time

def mf_pipeline(directory, result_foldername, mock=False, num_simulations=None):
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
        noise = .0005
        kernel = 99
        num_bins = 10
        counter = 0
        if not os.path.exists(result_foldername):
            os.mkdir(result_foldername)
        
        alpha_actual = [[] for _ in range(num_bins)]
        alpha_predicted = [[] for _ in range(num_bins)]
        i_actual = [[] for _ in range(num_bins)]
        i_predicted = [[] for _ in range(num_bins)]
        P_actual = [[] for _ in range(num_bins)]
        P_predicted = [[] for _ in range(num_bins)]
        mass_actual = [[] for _ in range(num_bins)]
        mass_predicted = [[] for _ in range(num_bins)]
        alphas = [[] for _ in range(num_bins)]
        inclinations = [[] for _ in range(num_bins)]
        periods = [[] for _ in range(num_bins)]
        masses = [[] for _ in range(num_bins)]
        
        # generate bins
        alpha_bins = [j*2/(num_bins) + 2 for j in range(num_bins)]
        cosi_bins = [2*j/num_bins - 1 for j in range(num_bins)]
        P_bins = [np.e**(j*(np.log(27)-np.log(1))/num_bins) for j in range(num_bins)]
        mass_bins = [5.6*j/num_bins + 5 for j in range(num_bins)]
        
        start = time.time()
        for z in range(num_simulations):
            pos_signal = np.random.choice([True, False])
            P = mlc.P_rng()
            M_BH = mlc.mbh_rng()
            i = mlc.i_rng()
            cosi = math.cos(i)
            alpha = np.random.random() * 2 + 2
            if pos_signal:
                lcur, EV, Beam, SL = mlc.generate_light_curve(P, i, M_BH, std=noise)
            else:
                lcur = mlc.generate_flat_signal(noise)
            flat_lcur = lcur - scipy.signal.medfilt(lcur, kernel)
            
            std = np.std(flat_lcur)
            amp = 4*std
            initial = True
            best_ratio = 0
            best_results = None
            
            for width in [5 + 5*j for j in range(12)]:  # try widths from 10 mins to 2hrs (5 bins to 60 bins)
                template = mlc.generate_template(amp, width)
                threshold = amp**2 * width / alpha
                mf_results = mlc.match_filter(flat_lcur, template, threshold=threshold)
                if initial or mf_results["highest_corr"]/mf_results["threshold"] > best_ratio:
                    best_ratio = mf_results["highest_corr"]/mf_results["threshold"]
                    best_results = mf_results
                    initial = False
                if mf_results["result"]:
                    break
                
            result = best_results["result"]
            
            if (result != pos_signal and np.random.random() > .95) or (result == pos_signal and np.random.random() > .995):    # plot all light curves that were labeled incorrectly
                lc_folder = "./{}/lc{}".format(result_foldername, counter)
                if not os.path.exists(lc_folder):
                    os.mkdir(lc_folder)
                mlc.plot_lc(lcur, P, M_BH, i, filename="{}/lcur{}.pdf".format(lc_folder, counter), EV=EV if pos_signal else None, Beam=Beam if pos_signal else None, SL=SL if pos_signal else None)
                #mlc.plot_lc(ss_lc, P, M_BH, i, filename="{}/MLCS{}.pdf".format(lc_folder, counter), EV=EV, Beam=Beam, SL=SL if lc_pos else None)
                mlc.plot_lc(flat_lcur, P, M_BH, i, filename="{}/MLCF{}.pdf".format(lc_folder, counter))
                mlc.plot_corr(best_results["correlations"], P, M_BH, i, alpha, len(best_results["template"]), threshold, "{}/CORR{}.pdf".format(lc_folder, counter))
                counter += 1
            
            alpha_binned, i_binned, P_binned, mass_binned = False, False, False, False
            
            for k in range(num_bins-1,-1,-1):
                if not alpha_binned and alpha >= alpha_bins[k]:
                    alphas[k].append(i)
                    alpha_actual[k].append(pos_signal)
                    alpha_predicted[k].append(result)
                if not i_binned and cosi >= cosi_bins[k]:
                    inclinations[k].append(i)
                    i_actual[k].append(pos_signal)
                    i_predicted[k].append(result)
                if not P_binned and P >= P_bins[k]:
                    periods[k].append(i)
                    P_actual[k].append(pos_signal)
                    P_predicted[k].append(result)
                if not mass_binned and M_BH >= mass_bins[k]:
                    masses[k].append(i)
                    mass_actual[k].append(pos_signal)
                    mass_predicted[k].append(result)
                if all([alpha_binned, i_binned, P_binned, mass_binned]):
                    break
            
            if z%1000 == 0 and z != 0:
                print('{} simulations complete'.format(z))
                prefix = "./{}/".format(result_foldername)
                plot_completeness(r'$\alpha$', alpha_bins, alpha_actual, alpha_predicted, num_bins, prefix + 'alpha')
                plot_completeness('cosi', cosi_bins, i_actual, i_predicted, num_bins, prefix + 'cosi')
                plot_completeness('Period [days]', P_bins, P_actual, P_predicted, num_bins, prefix + 'period')
                plot_completeness(r'$M_{BH} [M_{\odot}]$', mass_bins, mass_actual, mass_predicted, num_bins, prefix + 'mbh')
            
        end = time.time()
        
        return "{} minutes".format(round((end - start)/60, 2))
            
    else:
        alpha = 2
        if not os.path.exists(directory+result_foldername):
            os.mkdir(directory+result_foldername)
        file_results = {}
        for filename in os.listdir(directory):
            if filename.endswith(".fits"):
                fits_file = directory + filename
                
                with fits.open(fits_file, mode="readonly") as hdulist:
                    tess_bjds = hdulist[1].data['TIME']
                    #sap_fluxes = hdulist[1].data['SAP_FLUX']
                    pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
                    #var = hdulist[1].data['PSF_CENTR1']
                    hdulist.close()
                
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
                
                for width in [5 + 5*j for j in range(12)]:  # try widths from 10 mins to 2hrs (5 bins to 60 bins)
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
                    plt.plot(valid_times, valid_fluxes, 'ko')
                    plt.tight_layout()
                    graph_filename = '{}/light_curve.pdf'.format(folder)
                    plt.savefig(graph_filename)
                    plt.close()
                    
                    # plot flat light curve
                    plt.figure()
                    plt.xlabel('Time [days]')
                    plt.ylabel('Relative Flux')
                    plt.plot(valid_times, flat, 'ko')
                    plt.tight_layout()
                    graph_filename = '{}/flat_lcur.pdf'.format(folder)
                    plt.savefig(graph_filename)
                    plt.close()
                    
                    # plot correlations
                    window = len(best_results["template"])
                    plt.figure()
                    plt.xlabel('Time [days]')
                    plt.ylabel('Correlation')
                    plt.plot(valid_times[:-window], best_results["correlations"], 'k')
                    plt.plot(valid_times[:-window], [best_results["threshold"] for _ in range(len(best_results["correlations"]))], '--', color='orange')
                    plt.tight_layout()
                    graph_filename = '{}/correlations.pdf'.format(folder)
                    plt.savefig(graph_filename)
                    plt.close()
                
        return file_results
    
def plot_completeness(variable, bins, actual, predicted, num_bins, path_prefix):
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
    plt.plot([str(round(b, 2)) for b in bins], accs, 'k')
    filename = '{}_accuracy.pdf'.format(path_prefix, variable)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    #plot precision
    plt.figure()
    plt.xlabel(variable)
    plt.ylabel('Precision')
    plt.plot([str(round(b, 2)) for b in bins], pres, 'k')
    filename = '{}_precision.pdf'.format(path_prefix, variable)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    # plot recall
    plt.figure()
    plt.xlabel(variable)
    plt.ylabel('Recall')
    plt.plot([str(round(b, 2)) for b in bins], recs, 'k')
    filename = '{}_recall.pdf'.format(path_prefix, variable)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    # plot F1
    plt.figure()
    plt.xlabel(variable)
    plt.ylabel('F1')
    plt.plot([str(round(b, 2)) for b in bins], F1s, 'k')
    filename = '{}_F1.pdf'.format(path_prefix, variable)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

