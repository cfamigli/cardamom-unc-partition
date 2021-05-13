
# compare demcmc performance between multiple instances

import numpy as np
import glob
import sys
import os
import readwritebinary as rwb
from read_plot_output import find_all_chains, get_parnames, get_nofluxes_nopools_leaffall
from run_anova import get_output
import matplotlib
import matplotlib.pyplot as plt

def plot_output_ts(cbf_data, fwd_data_short, fwd_data_long, var='', savepath='', title=''):
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(11,11)
    n_steps = cbf_data['nodays']
    
    # short run, e.g. 100000 iterations
    ax[0].plot(np.nanmedian(fwd_data_short, axis=0), color='dodgerblue', linewidth=2, label='short')
    ax[0].fill_between(range(n_steps), np.nanpercentile(fwd_data_short, 25, axis=0), 
        np.nanpercentile(fwd_data_short, 75, axis=0), color='dodgerblue', alpha=0.5)
        
    # long run, e.g. 1000000 iterations
    ax[0].plot(np.nanmedian(fwd_data_long, axis=0), color='darkorange', linewidth=2, label='long')
    ax[0].fill_between(range(n_steps), np.nanpercentile(fwd_data_long, 25, axis=0), 
        np.nanpercentile(fwd_data_long, 75, axis=0), color='darkorange', alpha=0.5)
        
    ax[1].plot(abs(np.nanpercentile(fwd_data_short, 75, axis=0) - np.nanpercentile(fwd_data_short, 25, axis=0)), color='dodgerblue', linewidth=2, label='short')
    ax[1].plot(abs(np.nanpercentile(fwd_data_long, 75, axis=0) - np.nanpercentile(fwd_data_long, 25, axis=0)), color='darkorange', linewidth=2, label='long')
    ax[1].set_xlabel('Months')
    ax[1].set_ylabel('Range (absolute) between 75th and 25th percentile')
    
    obs = cbf_data['OBS'][var]
    obs[obs==-9999] = float('nan')
    ax[0].plot(obs, linewidth=2, label='Obs',color='k')
    ax[0].set_ylabel(var)
    ax[0].set_xlabel('Months')
    
    plt.subplots_adjust(hspace=.5,wspace=.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(savepath + title)
    plt.close()
    return

def main():
    
    model_id = sys.argv[1]
    mcmc_id = sys.argv[2] # 119 for normal, 3 for DEMCMC
    assim_type = '_longadapted'
    
    cur_dir = os.getcwd() + '/'
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
    cbr_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
    output_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = get_parnames('../../misc/', model_id)
    
    os.chdir(cbr_dir)
    files = glob.glob('*MCMC'+mcmc_id+'_*.cbr')
    pixels = ['3809','3524','2224','4170','1945','3813','4054','3264','1271','3457']
    n_iter = ['100000','100250']#,'1000000']
    
    for pixel in pixels:
        pixel_chains = find_all_chains(files, pixel)
        pixel_chains.sort() # filenames
        print(pixel_chains)
        
        cbf_pixel = rwb.read_cbf_file(cur_dir + cbf_dir + pixel_chains[0].partition('_MCMC')[0]+'_'+pixel+'.cbf')
       
        fwd_data = []
        for it in n_iter:
            sub = [p for p in pixel_chains if '_'+it+'_' in p]
            print(sub)
            
            for file in sub:
                cbr_chain = rwb.read_cbr_file(file, {'nopars': len(parnames)})
                cbr_pixel = np.copy(cbr_chain) if sub.index(file)==0 else np.concatenate((cbr_pixel, cbr_chain), axis=0)
                
                flux_chain = rwb.readbinarymat(cur_dir + output_dir + 'fluxfile_' + file[:-3]+'bin', [cbf_pixel['nodays'], get_nofluxes_nopools_leaffall(model_id)[0]])
                pool_chain = rwb.readbinarymat(cur_dir + output_dir + 'poolfile_' + file[:-3]+'bin', [cbf_pixel['nodays']+1, get_nofluxes_nopools_leaffall(model_id)[1]])
                
                flux_pixel = np.copy(flux_chain) if sub.index(file)==0 else np.concatenate((flux_pixel, flux_chain), axis=0)
                pool_pixel = np.copy(pool_chain) if sub.index(file)==0 else np.concatenate((pool_pixel, pool_chain), axis=0)
            
            fwd_data.append(get_output('NBE', model_id, flux_pixel, pool_pixel, cbr_pixel, get_nofluxes_nopools_leaffall(model_id)[2]))
        
        plot_output_ts(cbf_pixel, fwd_data[0], fwd_data[1], var='NBE', savepath=cur_dir+plot_dir+'timeseries/', title='demcmc_compare_'+model_id+'_NBE_'+pixel+'.png')
    
    return

if __name__=='__main__':
    main()