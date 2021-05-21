
"""

created on Wed May 19 14:37:41 2021

@author: cfamigli

compare ensemble spread to number of iterations for a given model

"""
import numpy as np
import glob
import sys
import os
import readwritebinary as rwb
import anova_utilities as autil
import matplotlib
import matplotlib.pyplot as plt

def main():
    model_id = sys.argv[1]
    run_type = sys.argv[2] # ALL or SUBSET
    metric = sys.argv[3] # spread or RMSE
    assim_type = '_longadapted'
    
    cur_dir = os.getcwd() + '/'
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
    cbr_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
    output_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = autil.get_parnames('../../misc/', model_id)
    
    
    mcmc_ids = ['3', '119']
    n_iters = [['100000', '250000', '500000', '1000000', '1750000', '2500000', '5000000'], ['100000', '250000', '500000', '1000000', '5000000', '10000000', '25000000']]
    vrs = ['NBE','cumNBE','LAI','GPP','Reco','Rauto','Rhet','lit','root','som','wood']
    pixels = ['3809','3524','2224','4170','1945','3813','4054','3264','1271','3457']

    os.chdir(cbr_dir)   
    ens_spread = [np.ones((len(pixels), len(vrs), len(n_iters[0])))*float('nan'), np.ones((len(pixels), len(vrs), len(n_iters[1])))*float('nan')]
    conv = [np.ones((len(pixels), len(n_iters[0])))*float('nan'), np.ones((len(pixels), len(n_iters[1])))*float('nan')]
    for pixel in pixels:
        
        for mcmc_id in mcmc_ids:
            for it in n_iters[mcmc_ids.index(mcmc_id)]:
                files = glob.glob('*MCMC'+mcmc_id+'_'+it+'_'+pixel+'*.cbr')
                pixel_chains = autil.find_all_chains(files, pixel)
                pixel_chains.sort() # filenames
                print(pixel_chains)
                
                cbf_pixel = rwb.read_cbf_file(cur_dir + cbf_dir + pixel_chains[0].partition('_MCMC')[0]+'_'+pixel+'.cbf')
                
                cbr_chain_list = []
                for pixel_chain in pixel_chains:
                    print(pixel_chain)
                    cbr_chain = rwb.read_cbr_file(pixel_chain, {'nopars': len(parnames)})
                    cbr_pixel = np.copy(cbr_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((cbr_pixel, cbr_chain), axis=0)
                    
                    flux_chain = rwb.readbinarymat(cur_dir + output_dir + 'fluxfile_' + pixel_chain[:-3]+'bin', [cbf_pixel['nodays'], autil.get_nofluxes_nopools_lma(model_id)[0]])
                    pool_chain = rwb.readbinarymat(cur_dir + output_dir + 'poolfile_' + pixel_chain[:-3]+'bin', [cbf_pixel['nodays']+1, autil.get_nofluxes_nopools_lma(model_id)[1]])
                    
                    flux_pixel = np.copy(flux_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((flux_pixel, flux_chain), axis=0)
                    pool_pixel = np.copy(pool_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((pool_pixel, pool_chain), axis=0)
                    
                    cbr_chain_list.append(cbr_chain)
                    print(np.shape(cbr_chain))
                    print(np.shape(cbr_pixel))
                    
                gr = autil.gelman_rubin(cbr_chain_list)
                print('%i of %i parameters converged' % (sum(gr<1.2), len(parnames)))
                conv[mcmc_ids.index(mcmc_id)][pixels.index(pixel), n_iters[mcmc_ids.index(mcmc_id)].index(it)] = sum(gr<1.2)/len(parnames)*100
                
                for var in vrs:
                    print(var)
                    
                    try:
                        obs = cbf_pixel['OBS'][var]
                        obs[obs==-9999] = float('nan')
                    except:
                        obs = np.ones(cbf_pixel['nodays'])*np.nan
                    n_obs = np.sum(np.isfinite(obs))
                    
                    fwd_data = autil.get_output(var, model_id, flux_pixel, pool_pixel, cbr_pixel, autil.get_nofluxes_nopools_lma(model_id)[2])
                    
                    if len(fwd_data)>0:
                        if fwd_data.shape[1]>cbf_pixel['nodays']:
                            fwd_data = fwd_data[:,:-1]
                        
                        fwd_data = autil.remove_outliers(fwd_data)
                        med = np.nanmedian(fwd_data, axis=0)
                        ub = np.nanpercentile(fwd_data, 75, axis=0)
                        lb = np.nanpercentile(fwd_data, 25, axis=0)
                        
                        ens_spread[mcmc_ids.index(mcmc_id)][pixels.index(pixel), vrs.index(var), n_iters[mcmc_ids.index(mcmc_id)].index(it)] = np.nanmean(abs(ub - lb)) if metric=='spread' else np.sqrt(np.nansum((med-obs)**2)/n_obs)
                    
    for var in vrs:                
        autil.plot_spread_v_iter(ens_spread, pixels, vrs.index(var), var, n_iters, metric, cur_dir+plot_dir+'spread_v_iter', 'iter_test_'+model_id+'_'+var + '_' + metric)#'iter_test_MCMC'+mcmc_id+'_'+model_id+'_'+var + '_' + metric)
        
    autil.plot_conv_v_iter(conv, pixels, n_iters, cur_dir+plot_dir+'spread_v_iter', 'iter_test_'+model_id + '_conv')
    
    return

if __name__=='__main__':
    main()