from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import math
import mock_light_curves as mlc
import confusion_matrix as cf
import os
import scipy.signal
import time
import matplotlib.backends.backend_pdf as pdfs

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
    kernel = 99
    # generate templates varying width from 30mins to 2hrs (15 bins to 60 bins)
    templates = []
    widths = [5*j + 15 for j in range(10)]
    for width in widths: 
        templates.append(mlc.offset_and_normalize(mlc.generate_template(1, width)))
        
    if mock:
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
        
        noise_actual = [[] for _ in range(num_bins)]
        noise_predicted = [[] for _ in range(num_bins)]
        noises = [[] for _ in range(num_bins)]
        
        threshold_actual = [[] for _ in range(num_bins)]
        threshold_predicted = [[] for _ in range(num_bins)]
        thresholds = [[] for _ in range(num_bins)]
        
        total_actual = []
        total_predicted = []
        
        # generate bins
        cosi_bins = [j/num_bins * .01  for j in range(num_bins)]
        P_bins = [np.e**(j*(np.log(27)-np.log(1))/num_bins) for j in range(num_bins)]
        mbh_bins = [15*j/num_bins + 5 for j in range(num_bins)]
        ms_bins = [j/num_bins + 0.5 for j in range(num_bins)]
        noise_bins = np.array([0,50,100,150,250,500,750,1000,1500,2000]) 
        altered_noise_bins = noise_bins * 10**(-6) + 10**(-9)
        threshold_bins = np.array([.01*i for i in range(10)]) 
        
        start = time.time()
        for z in range(1, num_simulations+1):
            
            # randomly generate relevant parameters from known priors
            P = mlc.P_rng()
            M_BH = mlc.mbh_rng()
            M_S = mlc.ms_rng()
            i = mlc.i_rng()
            cosi = math.cos(i)
            noise = np.random.choice(altered_noise_bins)
            threshold = np.random.random() * .1
            
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
            if (best_result != pos_signal and np.random.random() > .98) or (pos_signal and np.random.random() > .99):
                lc_folder = "./{}/lc{}".format(result_foldername, counter)
                if not os.path.exists(lc_folder):
                    os.mkdir(lc_folder)
                mlc.plot_lc(lcur, P, M_BH, i, M_S, filename="{}/lcur{}.pdf".format(lc_folder, counter), EV=EV if pos_signal else None, Beam=Beam if pos_signal else None, SL=SL if pos_signal else None)
                mlc.plot_lc(flat_lcur, P, M_BH, i, M_S, filename="{}/flat_lcur{}.pdf".format(lc_folder, counter))
                mlc.plot_corr(best_correlations, P, M_BH, i, M_S, threshold, noise, "{}/corr{}.pdf".format(lc_folder, counter))
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
                    
            threshold_binned, noise_binned = False, False
            for k in range(num_bins-1,-1,-1):
                if not threshold_binned and threshold >= threshold_bins[k]:
                    thresholds[k].append(i)
                    threshold_actual[k].append(pos_signal)
                    threshold_predicted[k].append(result)
                if not noise_binned and noise == altered_noise_bins[k]:
                    noises[k].append(i)
                    noise_actual[k].append(pos_signal)
                    noise_predicted[k].append(result)
                if noise_binned and threshold_binned:
                    break
                    
            total_actual.append(pos_signal)
            total_predicted.append(best_result) 
            
            # perform completeness analysis
            if z%1000 == 0 and z != 0:
                print('{} simulations complete'.format(z))
                prefix = "./{}/".format(result_foldername)
                plot_completeness(r'$\alpha$', threshold_bins, threshold_actual, threshold_predicted, num_bins, prefix + 'alpha', scale=True)
                plot_completeness('Noise [ppm]', noise_bins, noise_actual, noise_predicted, num_bins, prefix + 'noise', scale=True)
                plot_completeness('cosi', cosi_bins, i_actual, i_predicted, num_bins, prefix + 'cosi', scale=True)
                plot_completeness('Period [days]', P_bins, P_actual, P_predicted, num_bins, prefix + 'period')
                plot_completeness(r'$M_{BH} [M_{\odot}]$', mbh_bins, mbh_actual, mbh_predicted, num_bins, prefix + 'mbh')
                plot_completeness(r'$M_{\star} [M_{\odot}]$', ms_bins, ms_actual, ms_predicted, num_bins, prefix + 'ms')
            
        end = time.time()
        print("{} minutes".format(round((end - start)/60, 2)))
        
        return total_actual, total_predicted
            
    else:
        threshold = 0.1
        counter = 1
        num_files = 0
        results = {'real': set(),
                   'edge': set(),
                   'transit': set()}
        if not os.path.exists(directory+result_foldername):
            os.mkdir(directory+result_foldername)
            os.mkdir(directory+result_foldername+'/real')
            os.mkdir(directory+result_foldername+'/edge')
            os.mkdir(directory+result_foldername+'/transit')
            os.mkdir(directory+result_foldername+'/flare')
            
        for filename in os.listdir(directory):
            num_files += 1
            if num_files%500 == 0:
                print('{} files completed'.format(num_files))
                
            if filename.endswith(".fits"):
                fits_file = directory + filename
                try:
                    with fits.open(fits_file, mode="readonly", memmap=False) as hdulist:
                        tess_bjds = hdulist[1].data['TIME']
                        #sap_fluxes = hdulist[1].data['SAP_FLUX']
                        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
                        centroid = hdulist[1].data['MOM_CENTR1']
                except:
                    print('Could not open file: {}'.format(filename))
                    continue
                
                # Take only valid data points
                notnan_times = []
                notnan_fluxes = []
                for i in range(len(pdcsap_fluxes)):
                    flux = pdcsap_fluxes[i]
                    if not np.isnan(flux):
                        notnan_times.append(tess_bjds[i])
                        notnan_fluxes.append(pdcsap_fluxes[i])
                        
                # determine left and right edges for orbital gap
                left_edge = None
                right_edge = None
                for k in range(len(notnan_times)):
                    if notnan_times[k] - notnan_times[k-1] > 1:
                        left_edge = notnan_times[k-1]
                        right_edge = notnan_times[k]
                        break
                        
                #subtract median filter to get high frequency noise
                flat_lcur_unclipped = notnan_fluxes - scipy.signal.medfilt(notnan_fluxes, kernel)
                flat_lcur_unclipped = mlc.offset_and_normalize(flat_lcur_unclipped)
                
                # remove lone outliers to avoid fitting to outliers
                sigma_clip = 10*np.std(flat_lcur_unclipped)
                length = len(flat_lcur_unclipped)
                flat_lcur = []
                valid_times = []
                valid_fluxes = []
                for i in range(length):
                    flux = flat_lcur_unclipped[i]
                    if abs(flux) > sigma_clip:
                        if i == 0:
                            if flat_lcur_unclipped[i+1] > sigma_clip:
                                flat_lcur.append(flux)
                                valid_times.append(notnan_times[i])
                                valid_fluxes.append(notnan_fluxes[i])
                        elif i == length-1:
                            if flat_lcur_unclipped[i-1] > sigma_clip:
                                flat_lcur.append(flux)
                                valid_times.append(notnan_times[i])
                                valid_fluxes.append(notnan_fluxes[i])
                        else:
                            if flat_lcur_unclipped[i+1] > sigma_clip or flat_lcur_unclipped[i-1] > sigma_clip:
                                flat_lcur.append(flux)
                                valid_times.append(notnan_times[i])
                                valid_fluxes.append(notnan_fluxes[i])
                    else:
                        flat_lcur.append(flux)
                        valid_times.append(notnan_times[i])
                        valid_fluxes.append(notnan_fluxes[i])
                        
                initial = True
                best_result = None
                best_corr = 0
                best_correlations = None
                best_template = None
                prev_positive_corr = None
                prev_result = False
                
                # perform cross-correlation for all template widths
                for template in templates:  
                    correlations = scipy.signal.correlate(flat_lcur, template, mode='valid')
                    highest_corr = max(correlations)
                    result  = highest_corr > threshold
                
                    # choose best correlation result so far 
                    if initial or highest_corr > best_corr:
                        best_corr = highest_corr
                        best_result = result
                        best_correlations = correlations
                        best_template = template
                        initial = False
                        
                    #break if new template has lower correlation than previous template
                    if prev_result:
                        if highest_corr < prev_positive_corr:
                            break
                    if result:
                        prev_positive_corr = highest_corr
                        
                    prev_result = result
                
                if best_result: # get plots for positive results 
                    min_corr = min(best_correlations)
                    corr_length = len(best_correlations)
                    window = len(best_template)
                    start_edge = valid_times[0]
                    end_edge = valid_times[corr_length-1]
                    detection_indices = []
                    prev_detection = False
                    best_detection_info = None
                    need_to_add = False
                    flag = 'real'
                    
                    #find locations of positive detections
                    for j in range(corr_length):
                        if best_correlations[j] > threshold:
                            if not prev_detection:
                                need_to_add = True
                                best_detection_info = (j, best_correlations[j])
                            else:
                                if best_correlations[j] > best_detection_info[1]:
                                    best_detection_info = (j, best_correlations[j])
                            prev_detection = True
                        else:
                            if need_to_add:
                                detection_indices.append(best_detection_info[0])
                                need_to_add = False
                            prev_detection = False
                    
                    nonedge_detections = []
                    # find locations of non-edge detections
                    for detection in detection_indices:
                        t = valid_times[detection]
                        if not is_edge_detection(t, start_edge, end_edge, left_edge, right_edge):
                            nonedge_detections.append(detection)
                            
                    # all detections are edge detections so we flag as 'edge'
                    if len(nonedge_detections) == 0:
                        flag = 'edge'
                        
                    # flag as transit if there is a stronger negative correlation than a positive one
                    if flag == 'real':
                        if abs(min_corr) > 1.5*best_corr:
                            flag = 'transit'
                        
                    # find locations of all real and flare detections
                    flare_detections = []
                    real_detections = []
                    flare_template = mlc.offset_and_normalize(mlc.generate_flare_template(1, window//3))
                    for detection in nonedge_detections:
                        detection_sample = mlc.offset_and_normalize(valid_fluxes[detection:detection+window])
                        gaussian_result = scipy.signal.correlate(detection_sample, best_template, mode='valid')[0]
                        flare_result = scipy.signal.correlate(detection_sample, flare_template, mode='valid')[0]
                        # compare correlation to gaussian and flare templates
                        if flare_result > gaussian_result:
                            flare_detections.append(detection)
                        else:
                            real_detections.append(detection)
                        
                    # all detections are flare detections so we flag as 'flare'
                    if flag == 'real' and len(real_detections) == 0:
                        flag = 'flare'
                            
                    folder = '{}{}/{}'.format(directory, result_foldername, flag)
                    pdf = pdfs.PdfPages('{}/light_curve{}.pdf'.format(folder, counter))
                    counter += 1
                            
                    # plot light curve 
                    fig = plt.figure()
                    plt.xlabel('Time [days]')
                    plt.ylabel('PDCSAP Flux')
                    plt.title(filename)
                    plt.plot(valid_times, valid_fluxes, 'ko', rasterized=True, markersize=1)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close()
                    
                    # plot flat light curve
                    fig = plt.figure()
                    plt.xlabel('Time [days]')
                    plt.ylabel('Relative Flux')
                    plt.plot(valid_times, flat_lcur, 'ko', rasterized=True, markersize=1)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close()
                    
                    # plot correlations
                    fig = plt.figure()
                    plt.xlabel('Time [days]')
                    plt.ylabel('Correlation')
                    plt.plot(valid_times[:corr_length], best_correlations, 'ko', rasterized=True, markersize=1)
                    plt.plot([valid_times[0], valid_times[corr_length-1]], [threshold, threshold], '--', color='orange', rasterized=True)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close()
                    
                    # plot centroid
                    fig = plt.figure()
                    plt.xlabel('Time [days]')
                    plt.ylabel('Centroid')
                    plt.plot(tess_bjds, centroid, 'ko', rasterized=True, markersize=1)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close()
                    
                    # zoomed in plot on location of each real positive detection
                    for detection in real_detections:
                        detection_sample = mlc.offset_and_normalize(valid_fluxes[detection:detection+window])
                        fig = plt.figure()
                        plt.xlabel('Time [days]')
                        plt.ylabel('PDCSAP Flux')
                        plt.title('Detection at t = {}'.format(round(valid_times[detection],2)))
                        plt.plot(valid_times[detection:detection+window], detection_sample, 'ko', rasterized=True, markersize=1)
                        plt.plot(valid_times[detection:detection+window], best_template, 'bo', rasterized=True, markersize=1)
                        plt.plot(valid_times[detection:detection+window], flare_template, 'go', rasterized=True, markersize=1)
                        plt.legend(["Detection", "Gaussian", "Flare"])
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close()
                        
                    # zoomed in plot on location of each flare positive detection
                    for detection in flare_detections:
                        detection_sample = mlc.offset_and_normalize(valid_fluxes[detection:detection+window])
                        fig = plt.figure()
                        plt.xlabel('Time [days]')
                        plt.ylabel('PDCSAP Flux')
                        plt.title('Flare detection at t = {}'.format(round(valid_times[detection],2)))
                        plt.plot(valid_times[detection:detection+window], detection_sample, 'ko', rasterized=True, markersize=1)
                        plt.plot(valid_times[detection:detection+window], best_template, 'bo', rasterized=True, markersize=1)
                        plt.plot(valid_times[detection:detection+window], flare_template, 'go', rasterized=True, markersize=1)
                        plt.legend(["Detection", "Gaussian", "Flare"])
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close()
                    
                    pdf.close()
    
def is_edge_detection(t, start, end, left, right, cutoff=.5):
    return any([t <= start + cutoff, 
                t >= end - cutoff, 
                (t >= left - cutoff) and (t <= left),
                (t <= right + cutoff) and (t >= right)
                ])

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
        plt.plot([b for b in bins], accs, 'ko', rasterized=True)
    else:
        plt.plot([str(round(b, 2)) for b in bins], accs, 'ko', rasterized=True)
    filename = '{}_accuracy.pdf'.format(path_prefix, variable)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    #plot precision
    plt.figure()
    plt.xlabel(variable)
    plt.ylabel('Precision')
    if scale:
        plt.plot([b for b in bins], pres, 'ko', rasterized=True)
    else:
        plt.plot([str(round(b, 2)) for b in bins], pres, 'ko', rasterized=True)
    filename = '{}_precision.pdf'.format(path_prefix, variable)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    # plot recall
    plt.figure()
    plt.xlabel(variable)
    plt.ylabel('Recall')
    if scale:
        plt.plot([b for b in bins], recs, 'ko', rasterized=True)
    else:
        plt.plot([str(round(b, 2)) for b in bins], recs, 'ko', rasterized=True)
    filename = '{}_recall.pdf'.format(path_prefix, variable)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    # plot F1
    plt.figure()
    plt.xlabel(variable)
    plt.ylabel('F1')
    if scale:
        plt.plot([b for b in bins], F1s, 'ko', rasterized=True)
    else:
        plt.plot([str(round(b, 2)) for b in bins], F1s, 'ko', rasterized=True)
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
    
    plot_completeness('Alpha', alpha_values, actuals, predicteds, 10, "./{}/alpha".format(result_foldername), scale=True)
    return actuals, predicteds
    


# =============================================================================
# kernel = 99
# threshold = 0.1
# # generate templates varying width from 30mins to 2hrs (15 bins to 60 bins)
# templates = []
# widths = [5*j + 15 for j in range(10)]
# for width in widths: 
#     templates.append(mlc.offset_and_normalize(mlc.generate_template(1, width)))
#     
# fits_file = './LightCurvesSector20/tess2019357164649-s0020-0000000462637035-0165-s_lc.fits'
# with fits.open(fits_file, mode="readonly", memmap=False) as hdulist:
#     tess_bjds = hdulist[1].data['TIME']
#     #sap_fluxes = hdulist[1].data['SAP_FLUX']
#     pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
#     centroid = hdulist[1].data['MOM_CENTR1']
# 
# # Take only valid data points
# notnan_times = []
# notnan_fluxes = []
# left_edge = None
# right_edge = None
# nan_time = False
# for i in range(len(pdcsap_fluxes)):
#     flux = pdcsap_fluxes[i]
#     if not np.isnan(flux):
#         notnan_times.append(tess_bjds[i])
#         notnan_fluxes.append(pdcsap_fluxes[i])
#     if not nan_time and np.isnan(tess_bjds[i]):
#         left_edge = tess_bjds[i-1]
#         nan_time = True
#     if nan_time and not np.isnan(tess_bjds[i]):
#         right_edge = tess_bjds[i]
#         nan_time = False
#         
# #subtract median filter to get high frequency noise
# flat_lcur_unclipped = notnan_fluxes - scipy.signal.medfilt(notnan_fluxes, kernel)
# flat_lcur_unclipped = mlc.offset_and_normalize(flat_lcur_unclipped)
# 
# # remove lone outliers to avoid fitting to outliers
# sigma_clip = 10*np.std(flat_lcur_unclipped)
# length = len(flat_lcur_unclipped)
# flat_lcur = []
# valid_times = []
# valid_fluxes = []
# for i in range(length):
#     flux = flat_lcur_unclipped[i]
#     if abs(flux) > sigma_clip:
#         if i == 0:
#             if flat_lcur_unclipped[i+1] > sigma_clip:
#                 flat_lcur.append(flux)
#                 valid_times.append(notnan_times[i])
#                 valid_fluxes.append(notnan_fluxes[i])
#         elif i == length-1:
#             if flat_lcur_unclipped[i-1] > sigma_clip:
#                 flat_lcur.append(flux)
#                 valid_times.append(notnan_times[i])
#                 valid_fluxes.append(notnan_fluxes[i])
#         else:
#             if flat_lcur_unclipped[i+1] > sigma_clip or flat_lcur_unclipped[i-1] > sigma_clip:
#                 flat_lcur.append(flux)
#                 valid_times.append(notnan_times[i])
#                 valid_fluxes.append(notnan_fluxes[i])
#     else:
#         flat_lcur.append(flux)
#         valid_times.append(notnan_times[i])
#         valid_fluxes.append(notnan_fluxes[i])
#         
# initial = True
# best_result = None
# best_corr = 0
# best_correlations = None
# best_template = None
# prev_positive_corr = None
# prev_result = False
# 
# # perform cross-correlation for all template widths
# for template in templates:  
#     correlations = scipy.signal.correlate(flat_lcur, template, mode='valid')
#     highest_corr = max(correlations)
#     result  = highest_corr > threshold
# 
#     # choose best correlation result so far 
#     if initial or highest_corr > best_corr:
#         best_corr = highest_corr
#         best_result = result
#         best_correlations = correlations
#         best_template = template
#         initial = False
#         
#     #break on positive
#     if prev_result:
#         if highest_corr < prev_positive_corr:
#             break
#     if result:
#         prev_positive_corr = highest_corr
#         
#     prev_result = result
# 
# if best_result: # get plots for positive results 
#     min_corr = min(best_correlations)
#     corr_length = len(best_correlations)
#     window = len(best_template)
#     start_edge = valid_times[0]
#     end_edge = valid_times[corr_length-1]
#     detection_indices = []
#     prev_detection = False
#     best_detection_info = None
#     need_to_add = False
#     flag = 'real'
#     
#     #find locations of positive detections
#     for j in range(corr_length):
#         if best_correlations[j] > threshold:
#             if not prev_detection:
#                 need_to_add = True
#                 best_detection_info = (j, best_correlations[j])
#             else:
#                 if best_correlations[j] > best_detection_info[1]:
#                     best_detection_info = (j, best_correlations[j])
#             prev_detection = True
#         else:
#             if need_to_add:
#                 detection_indices.append(best_detection_info[0])
#                 need_to_add = False
#             prev_detection = False
#     
#     nonedge_detections = []
#     # find locations of non-edge detections
#     for detection in detection_indices:
#         t = valid_times[detection]
#         if not is_edge_detection(t, start_edge, end_edge, left_edge, right_edge):
#             nonedge_detections.append(detection)
#             
#     # all detections are edge detections so we flag as 'edge'
#     if len(nonedge_detections) == 0:
#         flag = 'edge'
#         
#     # flag as transit if there is a stronger negative correlation than a positive one
#     if flag == 'real':
#         if abs(min_corr) > 1.5*best_corr:
#             flag = 'transit'
#         
#     # find locations of all real and flare detections
#     flare_detections = []
#     real_detections = []
#     for detection in nonedge_detections:
#         detection_sample = mlc.offset_and_normalize(valid_fluxes[detection:detection+window])
#         flare_template = mlc.offset_and_normalize(mlc.generate_flare_template(1, window//3))
#         plt.figure()
#         plt.plot(detection_sample)
#         plt.plot(best_template)
#         plt.plot(flare_template)
#         plt.legend(["detection", "gaussian", "flare"])
#         plt.show()
#         plt.close()
#         
#         plt.figure()
#         plt.plot(centroid[detection:detection+250])
#         plt.show()
#         plt.close()
#         
#         gaussian_result = scipy.signal.correlate(detection_sample, best_template, mode='valid')[0]
#         flare_result = scipy.signal.correlate(detection_sample, flare_template, mode='valid')[0]
#         if flare_result > gaussian_result:
#             flare_detections.append(detection)
#         else:
#             real_detections.append(detection)
#         
#     # all detections are flare detections so we flag as 'flare'
#     if flag == 'real' and len(real_detections) == 0:
#         flag = 'flare'
# =============================================================================
