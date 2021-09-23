
"""

created on Wed May 26 09:05:31 2021

@author: cfamigli

run anova approach on CARDAMOM output
repeat for various subsets of models in suite

"""

import numpy as np
import readwritebinary as rwb
import os
import glob
import sys
from datetime import date
from pandas import DataFrame, to_pickle
import matplotlib
import matplotlib.pyplot as plt
import anova_utilities as autil
import random

def main():
    
    cur_dir = os.getcwd() + '/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    
    os.chdir(plot_dir+'dists/')
    
    # get list of model ids
    models_full = list(set([el.split('_')[0] for el in glob.glob('*.png')])) 
    
    # remove 101, temporary until 102-->101
    models_full.remove('102')
    os.chdir(cur_dir)
    
    # set lists of variables and pixels
    vrs = ['NBE','cumNBE','LAI','GPP','Reco','Rauto','Rhet','lit','root','som','wood']
    pixels = ['3809','3524','2224','4170','1945','3813','4054','3264','1271','3457']
    
    # set MCMC ID
    mcmc_id = sys.argv[1]
    n_iter = sys.argv[2]
    assim_type = '_longadapted'
    
    nmodels_leave_out = sys.argv[3]
    models = random.sample(models_full, len(models_full)-int(nmodels_leave_out))
    print(models)
    
    # dataframe will hold model structural uncertainty (Ms) and model parametric uncertainty (Mp) for each pixel-var combination
    # n is number of models that make up the suite
    partitioning = DataFrame(columns={'Ms','Mp','n'})
    df_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/processed_df/'
    
    for var in vrs: 
        print('Variable: ' + var)
        
        Mp_pixels = np.zeros(len(pixels))*np.nan # list of Mp for each pixel, for mapping
        for pixel in pixels:
            print('Pixel: ' + pixel)
            
            nsteps = 228 if assim_type=='_longadapted' else 240
            meds, ub, lb = np.zeros((len(models), nsteps))*np.nan, np.zeros((len(models), nsteps))*np.nan, np.zeros((len(models), nsteps))*np.nan # medians, upper bounds, lower bounds of prediction through time
            Mp, n = 0, 0
        
            for model in models:
                print(model)
            
                cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model + '/'
                cbr_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model + '/'
                output_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model + '/'
                parnames = autil.get_parnames(cur_dir+'../../misc/', model)
            
                os.chdir(cur_dir+cbr_dir)
                #files = set(glob.glob('*.cbr')) - set(glob.glob('*MCMC'+mcmc_id+'*.cbr'))
                #files = glob.glob('*MCMC'+mcmc_id+'*.cbr')
                files = set(glob.glob('*MCMC'+mcmc_id+'_'+n_iter+'_*.cbr'))
                
                pixel_chains = autil.find_all_chains(files, pixel) # list of files corresponding to each chain at that pixel, e.g. 2224_1, 2224_2, 2224_3, 2222_4
                pixel_chains.sort()
                n_chains = len(pixel_chains)
                
                if n_chains>0:
                    cbf_pixel = rwb.read_cbf_file(cur_dir + cbf_dir + pixel_chains[0].partition('_MCMC')[0]+'_'+pixel+'.cbf')
                
                    cbr_chain_list = []
                    for pixel_chain in pixel_chains:
                        print(pixel_chain)
                        cbr_chain = rwb.read_cbr_file(pixel_chain, {'nopars': len(parnames)}) # cbr file for one chain
                        cbr_chain_list.append(cbr_chain) # list of separate cbrs for each chain, use for gelman rubin
                        cbr_pixel = np.copy(cbr_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((cbr_pixel, cbr_chain), axis=0) # concatenate all chain cbrs
                        #autil.plot_par_histograms(cbr_chain, parnames=parnames, savepath=cur_dir+plot_dir+'dists/', title=model+'_'+pixel_chain[:-3]+'png')
                        
                        flux_chain = rwb.readbinarymat(cur_dir + output_dir + 'fluxfile_' + pixel_chain[:-3]+'bin', [cbf_pixel['nodays'], autil.get_nofluxes_nopools_lma(model)[0]])
                        pool_chain = rwb.readbinarymat(cur_dir + output_dir + 'poolfile_' + pixel_chain[:-3]+'bin', [cbf_pixel['nodays']+1, autil.get_nofluxes_nopools_lma(model)[1]])
                        #autil.plot_flux_pool_timeseries(cbf_pixel, cbr_chain, flux_chain, pool_chain, autil.get_nofluxes_nopools_lma(model)[2], savepath=cur_dir+plot_dir+'timeseries/', title=model+'_'+pixel_chain[:-3]+'png')
            
                        flux_pixel = np.copy(flux_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((flux_pixel, flux_chain), axis=0) # concatenate all chain flux outputs
                        pool_pixel = np.copy(pool_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((pool_pixel, pool_chain), axis=0) # concatenate all chain pool outputs
                        
                    gr = autil.gelman_rubin(cbr_chain_list) # gelman rubin function from matt
                    gr_thresh = 1.2 # below this value parameters are assumed to be convergent
                    print('%i of %i parameters converged with GR<%.1f' % (sum(gr<gr_thresh), len(parnames), gr_thresh))
                    
                    #autil.plot_par_histograms(cbr_pixel, parnames=parnames, savepath=cur_dir+plot_dir+'dists/', title=model+'_'+pixel_chain[:-6]+'.png')
                    #autil.plot_flux_pool_timeseries(cbf_pixel, cbr_pixel, flux_pixel, pool_pixel, autil.get_nofluxes_nopools_lma(model)[2], savepath=cur_dir+plot_dir+'timeseries/', title=model+'_'+pixel_chain[:-6]+'.png')
                    
                    if (sum(gr<gr_thresh)/len(parnames)<.9): # don't include nonconvergent runs in analysis
                        continue
                    else:
                        fwd_data = autil.get_output(var, model, flux_pixel, pool_pixel, cbr_pixel, autil.get_nofluxes_nopools_lma(model)[2]) # get forward data for var
                        
                        if len(fwd_data)>0:
                            if fwd_data.shape[1]>nsteps:
                                fwd_data = fwd_data[:,:-1]
                            
                            fwd_data = autil.remove_outliers(fwd_data)
                            # fill medians, upper bounds, and lower bounds
                            meds[models.index(model), :] = np.nanmedian(fwd_data, axis=0)
                            ub[models.index(model), :] = np.nanpercentile(fwd_data, 75, axis=0)
                            lb[models.index(model), :] = np.nanpercentile(fwd_data, 25, axis=0)
                            
                            fwd_data = autil.remove_below_25_above_75(fwd_data) # set values outside of 25th-75th range to nan
                            Mp += np.nanvar(fwd_data, axis=0) # sum of intra-ensemble variance, only compute on 25th-75th
                            n += 1
                    
            Ms = np.nanvar(meds, axis=0) # inter-median variance
            Mp = Mp/n if n!=0 else float('nan')
            
            Ms_div_sum = Ms/(Ms+Mp)
            Mp_div_sum = Mp/(Ms+Mp)
            
            partitioning.loc[pixel+'_'+var] = {'Ms': np.nanmean(Ms_div_sum), 'Mp': np.nanmean(Mp_div_sum), 'n': n}
            Mp_pixels[pixels.index(pixel)] = np.nanmean(Mp_div_sum)
                
    print(partitioning.to_string())
    partitioning.sort_index(axis=1).to_pickle(cur_dir+df_dir+'summary' + assim_type + '_MCMC'+mcmc_id + '_' + date.today().strftime("%m%d%y")+'_' +str(len(models)) +'.pkl')
    
    return

if __name__=='__main__':
    main()