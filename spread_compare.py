
"""

created on Wed May 19 14:37:41 2021

@author: cfamigli

compare ensemble spread to number of iterations for models 811 and 911, and DEMCMC versus MCMC

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
    combinations = [['811','119','40000000'],['811','3','1000000'],['911','119','40000000']]
    assim_type = '_longadapted'
    metric = sys.argv[1]

    vrs = ['NBE','cumNBE','LAI','GPP','Reco','Rauto','Rhet','lit','root','som','wood']
    pixels = ['3809','3524','2224','4170','1945','3813','4054','3264','1271','3457']

    ens_spread = np.ones((len(pixels), len(vrs), len(combinations)))*float('nan')
    conv = np.ones((len(pixels), len(combinations)))*float('nan')
    
    cur_dir = os.getcwd() + '/'
    
    for pixel in pixels:
        
        comb_count = 0
        for comb in combinations:
            
            model_id = comb[0]
            mcmc_id = comb[1]
            it = comb[2]
            
            cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
            cbr_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
            output_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
            plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
            parnames = autil.get_parnames(cur_dir+'../../misc/', model_id)
            
            os.chdir(cur_dir+cbr_dir) 
            files = glob.glob('*MCMC'+mcmc_id+'_'+it+'_'+pixel+'*.cbr')
            pixel_chains = autil.find_all_chains(files, pixel)
            pixel_chains.sort() # filenames
            if model_id=='911': pixel_chains = pixel_chains[-4:]
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
            conv[pixels.index(pixel), comb_count] = sum(gr<1.2)/len(parnames)*100
            
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
                    
                    ens_spread[pixels.index(pixel), vrs.index(var), comb_count] = np.nanmean(abs(ub - lb)) if metric=='spread' else np.sqrt(np.nansum((med-obs)**2)/n_obs)
               
            comb_count +=1
        
    for var in vrs:                
        autil.plot_spread_v_iter(ens_spread, pixels, vrs.index(var), var, it, metric, cur_dir+plot_dir+'spread_v_iter', 'iter_test_compare_'+assim_type+'_'+model_id+'_'+var + '_' + metric, single_val=True)#'iter_test_MCMC'+mcmc_id+'_'+model_id+'_'+var + '_' + metric)
        
    autil.plot_conv_v_iter(conv, pixels, it, cur_dir+plot_dir+'spread_v_iter', 'iter_test_compare'+assim_type+'_'+model_id + '_conv', single_val=True)
    
    return

if __name__=='__main__':
    main()