
# compare demcmc vs mcmc parameter retrievals

import numpy as np
import glob
import sys
import os
import readwritebinary as rwb
import anova_utilities as autil
import matplotlib
import matplotlib.pyplot as plt
from pandas import read_pickle, DataFrame

def plot_scatter_compare(demcmc_pred, mcmc_pred, parnames, savepath, savename):
    
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
                x = [demcmc_pred[i][count] for i in range(len(demcmc_pred))]
                y = [mcmc_pred[i][count] for i in range(len(mcmc_pred))]
                axs[row, col].scatter(x, y, facecolor='dodgerblue', edgecolor='k', linewidth=0.5, s=25)
                
                mx = max([np.nanmax(x), np.nanmax(y)])
                mn = min([np.nanmin(x), np.nanmin(y)])
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
    
    # get specifications for run to read
    model_ids = ['811','811']
    assim_type = '_p25adapted'
    ens_size = 500
    mcmc_ids = ['119','3']
    n_iters = ['40000000','1000000']
    
    # set directories
    cur_dir = os.getcwd() + '/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    
    n_pixels = 928
    demcmc_pred, mcmc_pred = [np.ones(34)*np.nan for i in range(n_pixels)], [np.ones(34)*np.nan for i in range(n_pixels)]
    # run through pixels
    for mcmc_id, n_iter, model_id in zip(mcmc_ids, n_iters, model_ids):
            
        # get list of directories
        cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
        cbr_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
        output_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
        parnames = autil.get_parnames('../../misc/', model_id)
        
        # get list of pixels
        pixels = [cbf[-8:-4] for cbf in glob.glob(cbf_dir + '*.cbf')]
        pixels.sort()
        
        # get best chains
        conv_chains = read_pickle(cbr_dir + model_id + assim_type + '_ALL' + '_MCMC'+mcmc_id + '_'+n_iter+'_best_subset.pkl')
        conv_chains.columns = ['pixel','bestchains','conv'] #rename columns for easier access
        
        for pixel in pixels:
            
            ind = pixels.index(pixel)
            
            if (len(glob.glob(cbr_dir + '*MCMC'+mcmc_id+'_'+n_iter+'_' + pixel + '*.cbr'))>0) & (pixel in conv_chains['pixel'].values):
                    
                # read cbf file for that pixel
                cbf_pixel = rwb.read_cbf_file(glob.glob(cbf_dir + '*_' + pixel+'.cbf')[0])
                
                # grab cbrs corresponding to that pixel, MCMCID and number of iterations
                cbr_files = glob.glob(cbr_dir + '*MCMC'+mcmc_id+'_' + n_iter + '_'+ pixel+'*.cbr')
                cbr_files.sort()
            
                # run through cbrs
                best_chains = conv_chains.loc[conv_chains['pixel']==pixel]['bestchains'].values[0][1:]
                print(pixel, best_chains)
                
                cbr_data = []
                conv = conv_chains.loc[conv_chains['pixel']==pixel]['conv'].values[0]
                if conv==1:
                    # aggregate bestchains from optimal posteriors
                    for chain in best_chains:
            
                        file = [i for i in cbr_files if pixel+'_'+chain+'.cbr' in i][0]
                        cbr_data.append(autil.modulus_Bday_Fday(rwb.read_cbr_file(file, {'nopars': len(parnames)}), parnames))
                        
                    cbr_data = np.vstack(cbr_data)
        
                else: cbr_data = np.ones((ens_size, len(parnames)))*np.nan
                
                
            
                if mcmc_id=='119': 
                    mcmc_pred[ind] = np.nanmedian(cbr_data, axis=0)
                elif mcmc_id=='3': 
                    demcmc_pred[ind] = np.nanmedian(cbr_data, axis=0)

    plot_scatter_compare(demcmc_pred, mcmc_pred, parnames, cur_dir+plot_dir+'demcmc_mcmc/', 'par_compare_811')
    
    
    return

if __name__=='__main__':
    main()