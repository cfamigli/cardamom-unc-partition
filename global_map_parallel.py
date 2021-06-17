"""

created on Thu May 27 13:11:01 2021

@author: cfamigli

compute convergence for a given pixel
save to a csv file

"""

import numpy as np
import os
import glob
import sys
import readwritebinary as rwb
import anova_utilities as autil
import csv

def main():
    model_id = sys.argv[1]
    run_type = sys.argv[2] # ALL or SUBSET
    mcmc_id = sys.argv[3] # 119 for normal, 3 for DEMCMC
    n_iter = sys.argv[4]
    var_to_plot = sys.argv[5] # GR, a flux or pool, or PARXX
    ens_size = 500
    assim_type = '_longadapted'
    
    cur_dir = os.getcwd() + '/'
    if 'scripts' not in cur_dir:
        cur_dir = cur_dir + 'scripts/'
    
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
    cbr_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
    output_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = autil.get_parnames(cur_dir + '../../misc/', model_id)
    
    os.chdir(cbr_dir)
    files = glob.glob('*MCMC'+mcmc_id+'_'+n_iter+'_*.cbr')
    pixel = sys.argv[6]
    print(pixel)
    
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
        
        if np.shape(cbr_chain)[0]==ens_size:
            cbr_chain_list.append(cbr_chain)
            print(np.shape(cbr_chain))
      
    ### COMPUTE GELMAN RUBIN  
    if len(cbr_chain_list)>1:
        gr = autil.gelman_rubin(cbr_chain_list)
        gr_pixel = sum(gr<1.2)/len(parnames)
    else:
        gr_pixel = -9999.
       
    ### DETERMINE DATA TO WRITE TO FILE
    if var_to_plot == 'GR':
        data = np.copy(gr_pixel)
    elif 'PAR' in var_to_plot:
        parnum = int(var_to_plot.partition('PAR')[-1])
        if gr_pixel>0.9:
            data = np.nanmedian(cbr_pixel[:,parnum-1])
        else:
            data = -9999.
    else:
        if gr_pixel>0.9:
            data = np.nanmean(np.nanmedian(autil.get_output(var_to_plot, model_id, flux_pixel, pool_pixel, cbr_pixel, autil.get_nofluxes_nopools_lma(model_id)[2]), axis=0))
        else:
            data = -9999.
        
    with open(cur_dir + '../../misc/' + model_id + '_' + pixel_chains[0].partition('_MCMC')[0] + '_MCMC' + mcmc_id + '_' + n_iter + '_' + var_to_plot + '.csv','a') as f:
        writer = csv.writer(f)
        new_row = [pixel, data]
        assert len(new_row)==2
        writer.writerow(new_row)
        
    return

if __name__=='__main__':
    main()