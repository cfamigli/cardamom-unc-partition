
"""

created on Mon May 3 13:34:27 2021

@author: cfamigli

run anova approach on CARDAMOM output

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
from read_plot_output import get_parnames, get_nofluxes_nopools_lma, find_all_chains, gelman_rubin
import basic_plots

def get_output(var, model, flux_data, pool_data, cbr_data, lma_ind):
    # return relevant output depending on flux or pool input as 'var'
    return {
        'NBE': np.sum(flux_data[:,:,[2,12,13]], axis=2) - flux_data[:,:,0] + flux_data[:,:,16],
        'cumNBE': np.cumsum(np.sum(flux_data[:,:,[2,12,13]], axis=2) - flux_data[:,:,0] + flux_data[:,:,16], axis=1),
        'LAI': pool_data[:,:,1]/np.expand_dims(cbr_data[:,lma_ind],1),
        'GPP': flux_data[:,:,0],
        'Reco': np.sum(flux_data[:,:,[2,12,13]], axis=2),
        'Rauto': flux_data[:,:,2],
        'Rhet': np.sum(flux_data[:,:,[12,13]], axis=2),
        'lit': pool_data[:,:,4] if int(model)>=400 else [],
        'root': pool_data[:,:,2] if int(model)>=400 else [],
        'som': pool_data[:,:,5] if int(model)>=400 else [],
        'wood': pool_data[:,:,3] if int(model)>=400 else []
    }[var]
    
def remove_outliers(fwd_data):
    meds = np.nanmedian(fwd_data, axis=0)
    for i in range(len(meds)):
        fwd_data[:,i][(fwd_data[:,i]>abs(meds[i]*1e6)) | (fwd_data[:,i]<-1*abs(meds[i]*1e6))] = float('nan')
    return fwd_data

def main():
    
    cur_dir = os.getcwd() + '/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    
    os.chdir(plot_dir+'dists/')
    
    # get list of model ids
    models = list(set([el.split('_')[0] for el in glob.glob('*.png')])) 
    
    # remove 101, temporary until 102-->101
    models.remove('102')
    os.chdir(cur_dir)
    
    # set lists of variables and pixels
    vrs = ['NBE','cumNBE','LAI','GPP','Reco','Rauto','Rhet','lit','root','som','wood']
    pixels = ['3809','3524','2224','4170','1945','3813','4054','3264','1271','3457']
    
    # set MCMC ID
    mcmc_id = sys.argv[1]
    n_iter = sys.argv[2]
    assim_type = '_longadapted'
    
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
                parnames = get_parnames(cur_dir+'../../misc/', model)
            
                os.chdir(cur_dir+cbr_dir)
                #files = set(glob.glob('*.cbr')) - set(glob.glob('*MCMC'+mcmc_id+'*.cbr'))
                #files = glob.glob('*MCMC'+mcmc_id+'*.cbr')
                files = set(glob.glob('*MCMC'+mcmc_id+'_'+n_iter+'_*.cbr'))
                
                pixel_chains = find_all_chains(files, pixel) # list of files corresponding to each chain at that pixel, e.g. 2224_1, 2224_2, 2224_3, 2222_4
                pixel_chains.sort()
                n_chains = len(pixel_chains)
                
                cbf_pixel = rwb.read_cbf_file(cur_dir + cbf_dir + pixel_chains[0].partition('_MCMC')[0]+'_'+pixel+'.cbf')
            
                cbr_chain_list = []
                for pixel_chain in pixel_chains:
                    print(pixel_chain)
                    cbr_chain = rwb.read_cbr_file(pixel_chain, {'nopars': len(parnames)}) # cbr file for one chain
                    cbr_chain_list.append(cbr_chain) # list of separate cbrs for each chain, use for gelman rubin
                    cbr_pixel = np.copy(cbr_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((cbr_pixel, cbr_chain), axis=0) # concatenate all chain cbrs
                    #basic_plots.plot_par_histograms(cbr_chain, parnames=parnames, savepath=cur_dir+plot_dir+'dists/', title=model+'_'+pixel_chain[:-3]+'png')
                    
                    flux_chain = rwb.readbinarymat(cur_dir + output_dir + 'fluxfile_' + pixel_chain[:-3]+'bin', [cbf_pixel['nodays'], get_nofluxes_nopools_lma(model)[0]])
                    pool_chain = rwb.readbinarymat(cur_dir + output_dir + 'poolfile_' + pixel_chain[:-3]+'bin', [cbf_pixel['nodays']+1, get_nofluxes_nopools_lma(model)[1]])
                    #basic_plots.plot_flux_pool_timeseries(cbf_pixel, cbr_chain, flux_chain, pool_chain, get_nofluxes_nopools_lma(model)[2], savepath=cur_dir+plot_dir+'timeseries/', title=model+'_'+pixel_chain[:-3]+'png')
        
                    flux_pixel = np.copy(flux_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((flux_pixel, flux_chain), axis=0) # concatenate all chain flux outputs
                    pool_pixel = np.copy(pool_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((pool_pixel, pool_chain), axis=0) # concatenate all chain pool outputs
                    
                gr = gelman_rubin(cbr_chain_list) # gelman rubin function from matt
                gr_thresh = 1.2 # below this value parameters are assumed to be convergent
                print('%i of %i parameters converged with GR<%.1f' % (sum(gr<gr_thresh), len(parnames), gr_thresh))
                
                #basic_plots.plot_par_histograms(cbr_pixel, parnames=parnames, savepath=cur_dir+plot_dir+'dists/', title=model+'_'+pixel_chain[:-6]+'.png')
                #basic_plots.plot_flux_pool_timeseries(cbf_pixel, cbr_pixel, flux_pixel, pool_pixel, get_nofluxes_nopools_lma(model)[2], savepath=cur_dir+plot_dir+'timeseries/', title=model+'_'+pixel_chain[:-6]+'.png')
                
                if (sum(gr<gr_thresh)/len(parnames)<.9): # don't include nonconvergent runs in analysis
                    continue
                else:
                    fwd_data = get_output(var, model, flux_pixel, pool_pixel, cbr_pixel, get_nofluxes_nopools_lma(model)[2]) # get forward data for var
                    
                    if len(fwd_data)>0:
                        if fwd_data.shape[1]>nsteps:
                            fwd_data = fwd_data[:,:-1]
                        
                        fwd_data = remove_outliers(fwd_data)
                        # fill medians, upper bounds, and lower bounds
                        meds[models.index(model), :] = np.nanmedian(fwd_data, axis=0)
                        ub[models.index(model), :] = np.nanpercentile(fwd_data, 75, axis=0)
                        lb[models.index(model), :] = np.nanpercentile(fwd_data, 25, axis=0)
                        Mp += np.nanvar(fwd_data, axis=0) # sum of intra-ensemble variance
                        n += 1
                    
            Ms = np.nanvar(meds, axis=0) # inter-median variance
            Mp = Mp/n if n!=0 else float('nan')
            
            Ms_div_sum = Ms/(Ms+Mp)
            #Ms_div_sum[np.isnan(Ms_div_sum)] = 0
            Mp_div_sum = Mp/(Ms+Mp)
            #Mp_div_sum[np.isnan(Mp_div_sum)] = 1
            
            partitioning.loc[pixel+'_'+var] = {'Ms': np.nanmean(Ms_div_sum), 'Mp': np.nanmean(Mp_div_sum), 'n': n}
            basic_plots.plot_anova_ts(meds, ub, lb, Ms_div_sum, Mp_div_sum, models, pixel=pixel, var=var, assim_type=assim_type + '_MCMC'+mcmc_id, savepath=cur_dir+plot_dir+'anova_ts')
            Mp_pixels[pixels.index(pixel)] = np.nanmean(Mp_div_sum)
                
        basic_plots.plot_map(nrows=46, ncols=73, land_pixel_list=[file[-8:-4] for file in glob.glob(cur_dir + cbf_dir + '*.cbf')], pixel_value_list=pixels, value_list=Mp_pixels, vmin=0, vmax=1, cbar_label='Fraction of variance due to parameters', savepath=cur_dir+plot_dir+'maps/', title=var + assim_type + '_MCMC'+mcmc_id)
        
    print(partitioning.to_string())
    partitioning.sort_index(axis=1).to_pickle(cur_dir+df_dir+date.today().strftime("%m%d%y")+'.pkl')
    basic_plots.plot_partitioning_grouped(partitioning, pixels, savepath=cur_dir+plot_dir+'anova_summary', savename='pixels' + assim_type + '_MCMC'+mcmc_id)
    basic_plots.plot_partitioning_grouped(partitioning, vrs, savepath=cur_dir+plot_dir+'anova_summary', savename='vars' + assim_type + '_MCMC'+mcmc_id)
    
    return

if __name__=='__main__':
    main()