
# compare demcmc performance between multiple instances

import numpy as np
import glob
import sys
import os
import readwritebinary as rwb
import anova_utilities as autil
import matplotlib
import matplotlib.pyplot as plt

def plot_output_ts(cbf_data, fwd_data_list, n_iter=[], var='', savepath='', title=''):
    plt.figure(figsize=(11,5))
    n_steps = cbf_data['nodays']
    
    colors = plt.cm.brg(np.linspace(0,1,len(fwd_data_list)))
    data_count = 0
    for data in fwd_data_list:
        # short run, e.g. 100000 iterations
        plt.plot(np.nanmedian(data, axis=0), color=colors[data_count], linewidth=2, alpha=0.8, label=n_iter[data_count] + ' iterations')
        plt.fill_between(range(n_steps), np.nanpercentile(data, 25, axis=0), 
            np.nanpercentile(data, 75, axis=0), color=colors[data_count], alpha=0.3)
        data_count+=1
        

    obs = cbf_data['OBS'][var]
    obs[obs==-9999] = float('nan')
    plt.plot(obs, linewidth=2, label='Obs',color='k')
    plt.ylabel(var)
    plt.xlabel('Months')
    
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
    parnames = autil.get_parnames('../../misc/', model_id)
    
    os.chdir(cbr_dir)
    files = glob.glob('*MCMC'+mcmc_id+'_*.cbr')
    pixels = ['3809','3524','2224','4170','1945','3813','4054','3264','1271','3457']
    n_iter = ['100000','1000000','2500000']
    
    for pixel in pixels:
        pixel_chains = autil.find_all_chains(files, pixel)
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
                
                flux_chain = rwb.readbinarymat(cur_dir + output_dir + 'fluxfile_' + file[:-3]+'bin', [cbf_pixel['nodays'], autil.get_nofluxes_nopools_lma(model_id)[0]])
                pool_chain = rwb.readbinarymat(cur_dir + output_dir + 'poolfile_' + file[:-3]+'bin', [cbf_pixel['nodays']+1, autil.get_nofluxes_nopools_lma(model_id)[1]])
                
                flux_pixel = np.copy(flux_chain) if sub.index(file)==0 else np.concatenate((flux_pixel, flux_chain), axis=0)
                pool_pixel = np.copy(pool_chain) if sub.index(file)==0 else np.concatenate((pool_pixel, pool_chain), axis=0)
            
            fwd_data.append(autil.get_output('NBE', model_id, flux_pixel, pool_pixel, cbr_pixel, autil.get_nofluxes_nopools_lma(model_id)[2]))
        
        plot_output_ts(cbf_pixel, fwd_data, n_iter, var='NBE', savepath=cur_dir+plot_dir+'timeseries/', title='demcmc_compare_'+model_id+'_NBE_'+pixel+'.png')
    
    return

if __name__=='__main__':
    main()