from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import mock_light_curves as mlc
import os
import scipy.signal

def mf_pipeline(directory, result_foldername, mock=False, mock_params=None):
    if mock:
        pass
    else:
        if not os.path.exists(directory+result_foldername):
            os.mkdir(directory+result_foldername)
        file_results = {}
        num_files = 0
        for filename in os.listdir(directory):
            if filename.endswith(".fits"):
                fits_file = directory + filename
                
                with fits.open(fits_file, mode="readonly") as hdulist:
                    tess_bjds = hdulist[1].data['TIME']
                    sap_fluxes = hdulist[1].data['SAP_FLUX']
                    pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
                    var = hdulist[1].data['PSF_CENTR1']
                
                num_days = len(tess_bjds)
                bin_size = (tess_bjds[-1] - tess_bjds[0]) / num_days
                # Find average flux and rms from valid data points
                total = 0
                total_square = 0
                x = 0
                
                for i in range(len(pdcsap_fluxes)):
                    flux = pdcsap_fluxes[i]
                    if not np.isnan(flux):
                        total += flux
                        total_square += flux**2
                        x += 1
                        
                average = total/x
                rms = (total_square/x)**.5
                
                # Replace invalid data points and normalize light curve
                for i in range(len(pdcsap_fluxes)):
                    if np.isnan(pdcsap_fluxes[i]):
                        pdcsap_fluxes[i] = average
                
                norm_fluxes = (pdcsap_fluxes - average)/rms
                        
                #subtract median filter to get high frequency noise
                flat = norm_fluxes - scipy.signal.medfilt(norm_fluxes, 99)
                
                
                std = np.std(flat)
                amp = 4*std
                initial = True
                best_ratio = 0
                best_results = None
                
                for width in [5 + 5*j for j in range(12)]:  # try widths from 10 mins to 2hrs (5 bins to 60 bins)
                    template = mlc.generate_template(amp, width)
                    threshold = amp**2 * width / 2
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
                    plt.plot(tess_bjds, pdcsap_fluxes, 'k')
                    plt.tight_layout()
                    graph_filename = '{}/light_curve.pdf'.format(folder)
                    plt.savefig(graph_filename)
                    plt.close()
                    
                    # plot flat light curve
                    plt.figure()
                    plt.xlabel('Time [days]')
                    plt.ylabel('Relative Flux')
                    plt.plot(tess_bjds, flat, 'k')
                    plt.tight_layout()
                    graph_filename = '{}/flat_lcur.pdf'.format(folder)
                    plt.savefig(graph_filename)
                    plt.close()
                    
                    # plot correlations
                    window = len(best_results["template"])
                    plt.figure()
                    plt.xlabel('Time [days]')
                    plt.ylabel('Correlation')
                    plt.plot(tess_bjds[:-window], best_results["correlations"], 'k')
                    plt.plot(tess_bjds[:-window], [best_results["threshold"] for _ in range(len(best_results["correlations"]))], '--', color='orange')
                    plt.tight_layout()
                    graph_filename = '{}/correlations.pdf'.format(folder)
                    plt.savefig(graph_filename)
                    plt.close()
                
                num_files += 1
                if num_files > 100:
                    break
                
        return file_results

