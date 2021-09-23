
import numpy as np
import glob
import sys
import os
from random import sample
from pandas import read_csv, read_pickle
import readwritebinary as rwb
import anova_utilities as autil
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def drop_nan(x, y):
    good_inds_y = ~np.any(np.isnan(y), axis=1)
    good_inds_x = ~np.any(np.isnan(x), axis=1)
    print(good_inds_x.shape, good_inds_y.shape)
    return x[(good_inds_x & good_inds_y)], y[(good_inds_x & good_inds_y)]

def plot_scatter_compare(x, y, parnames, savepath, savename):
    x, y = drop_nan(x, y)
    
    npars = len(parnames)
    ncols = 6
    nrows = 0
    while nrows<npars:
        nrows+=ncols
    nrows = int(nrows/6)
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_size_inches(13,12)
    
    count = 0
    for row in range(nrows):
        for col in range(ncols):
            if count<npars:
                
                axs[row, col].scatter(x[:,count], y[:,count], facecolor='dodgerblue', edgecolor='k', linewidth=0.5, s=25)
                
                mx = max([np.nanmax(x[:,count]), np.nanmax(y[:,count])])
                mn = min([np.nanmin(x[:,count]), np.nanmin(y[:,count])])
                axs[row, col].text(0.45,0.05,'R$^2$='+str(round(r2_score(y[:,count], x[:,count]), 2)), transform=axs[row, col].transAxes, weight='bold')
                axs[row, col].set_xlim([mn,mx])
                axs[row, col].set_ylim([mn,mx])
                axs[row, col].plot([mn,mx], [mn,mx], c='k', linewidth=1)
                if npars>1:
                    axs[row, col].set_title(parnames[count][:20])
                count += 1
         
    plt.subplots_adjust(hspace = .5,wspace=.5)
    plt.tight_layout()
    plt.savefig(savepath + savename + '.png')
    plt.close()
    return

def main():
    model_id = sys.argv[1]
    mcmc_id = sys.argv[2] # 119 for normal, 3 for DEMCMC
    n_iter = sys.argv[3]
    ens_size = 500
    assim_type = '_p25adapted'
    
    # EF comparison
    ef_spec = 'clipped_PLS_soilgrids_poolobs_rescaled_forward'
    
    # directories
    cur_dir = os.getcwd() + '/'
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
    cbr_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
    cbr_ef_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'_ef/' + model_id + '/'
    output_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
    output_ef_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'_ef/' + model_id + '/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = autil.get_parnames('../../misc/', model_id)
    
    
    # get cbfs to run through
    os.chdir(cbf_dir)
    cbf_files = glob.glob('*.cbf')
    cbf_files.sort()
    os.chdir(cur_dir + '/../')
    
    opt_preds = np.zeros((len(cbf_files), len(parnames))) * np.nan
    ef_preds = np.zeros((len(cbf_files), len(parnames))) * np.nan
    
    for cbf_file in cbf_files:
        
        pixel = cbf_file[-8:-4]
        print(pixel)
        
        pixel_chains_opt = autil.find_all_chains(glob.glob(cbr_dir+'*_MCMC'+mcmc_id+'_'+n_iter+'_'+pixel+'*.cbr'), pixel)
        pixel_chains_opt.sort() # filenames
        
        pixel_chains_ef = autil.find_all_chains(glob.glob(cbr_ef_dir+'*_MCMC'+mcmc_id+'_'+n_iter+'_'+ef_spec+'_'+pixel+'.cbr'), pixel)
        pixel_chains_ef.sort()
        
        for pc_opt in pixel_chains_opt:
            cbr_chain_opt = rwb.read_cbr_file(pc_opt, {'nopars': len(parnames)})
            cbr_chain_opt = autil.modulus_Bday_Fday(cbr_chain_opt, parnames)
            cbr_pixel_opt = np.copy(cbr_chain_opt) if pixel_chains_opt.index(pc_opt)==0 else np.concatenate((cbr_pixel_opt, cbr_chain_opt), axis=0)
        
        for pc_ef in pixel_chains_ef:
            cbr_chain_ef = rwb.read_cbr_file(pc_ef, {'nopars': len(parnames)})
            cbr_chain_ef = autil.modulus_Bday_Fday(cbr_chain_ef, parnames)
            cbr_pixel_ef = np.copy(cbr_chain_ef) if pixel_chains_ef.index(pc_ef)==0 else np.concatenate((cbr_pixel_ef, cbr_chain_ef), axis=0)
            
        opt_preds[cbf_files.index(cbf_file),:] = np.nanmedian(cbr_pixel_opt, axis=0)
        ef_preds[cbf_files.index(cbf_file),:] = np.nanmedian(cbr_pixel_ef, axis=0)
    
    
    plot_scatter_compare(ef_preds, opt_preds, parnames, plot_dir+'scatters/', model_id+'_MCMC'+mcmc_id+'_'+n_iter)
    
    return

if __name__=='__main__':
    main()