
"""

created on Thurs Jun 3 12:08:48 2021

@author: cfamigli

train models to predict a given parameter using PFT fractions

"""

import numpy as np
import readwritebinary as rwb
import os
import glob
import sys
from random import sample
import anova_utilities as autil
from eval_mstmip_lc import no_water_pixels
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

def drop_nan(X, y):
    cond_y = ~np.all(np.isnan(y), axis=0)
    cond_X = ~np.all(np.isnan(X), axis=1)
    
    y_not_nan = y[:, (cond_y & cond_X)]
    X_not_nan = X[(cond_y & cond_X), :]
    return X_not_nan, y_not_nan
    
def plot_scatter_test_pred(y_test, y_pred, parnames, savepath, savename):
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
                axs[row, col].scatter(y_pred[count], y_test[count], facecolor='dodgerblue', edgecolor='k', linewidth=0.5, s=25)
                mx = max([max(y_pred[count]), max(y_test[count])])
                mn = min([min(y_pred[count]), min(y_test[count])])
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
    
    # set run information to read
    model_id = sys.argv[1]
    mcmc_id = sys.argv[2] # 119 for normal, 3 for DEMCMC
    n_iter = sys.argv[3]
    ens_size = 500
    assim_type = '_longadapted'
    
    # set directories
    cur_dir = os.getcwd() + '/'
    misc_dir = cur_dir + '/../../misc/'
    cbf_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbf' + assim_type+'/' + model_id + '/'
    cbr_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbr' + assim_type+'/' + model_id + '/'
    plot_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = autil.get_parnames('../../misc/', model_id)
    
    # load map containing the location of each mstmip pixel on the GEOSCHEM grid
    pixel_nums = np.load(misc_dir + 'mstmip_pixel_nums.npy')
    
    # load map of biome fractions from mstmip
    with np.load(misc_dir + 'mstmip_biome_frac.npz') as data:
        biome_frac = data['arr_0']
    n_classes = biome_frac.shape[0]
        
    # load list of land pixels
    pixels = list(set([file[-8:-4] for file in glob.glob(cbf_dir + '*.cbf')]))
    
    # load list of cbrs
    files = glob.glob(cbr_dir+'*MCMC'+mcmc_id+'_'+n_iter+'_*.cbr')

    # fill X and Y
    n_regr_models = len(parnames)
    X = np.ones((len(pixels), n_classes))*np.nan # shape n_samples, n_features
    y = np.ones((n_regr_models, len(pixels)))*np.nan # shape n_pars, n_samples
    for pixel in pixels:
        ind = pixels.index(pixel)
        if np.mod(ind, 10)==0: print(ind)
        
        # get lc information
        locs = [pixel_nums==float(pixel)][0]
        fracs_at_geos_pixel = no_water_pixels(biome_frac[:,locs])
        av_fracs = np.nanmean(fracs_at_geos_pixel, axis=1) # average biome fraction across mstmip pixels within coarse pixel
        X[ind,:] = av_fracs

        # get parameter information
        pixel_chains = autil.find_all_chains(files, pixel)
        pixel_chains.sort() # filenames
        
        # concatenate across chains
        if len(pixel_chains)>0:
            for pixel_chain in pixel_chains:
                cbr_chain = rwb.read_cbr_file(pixel_chain, {'nopars': len(parnames)})
                cbr_pixel = np.copy(cbr_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((cbr_pixel, cbr_chain), axis=0)
            
            y[:,ind] = np.nanmedian(cbr_pixel, axis=0)
    
    # remove nan values so regression runs
    Xr, yr = drop_nan(X, y)
    
    # set up regression models
    y_test_all_pars, y_pred_all_pars = [],[]
    for regr_model in range(n_regr_models):
        print('running regression for ' + parnames[regr_model] + ' . . . ')
        # split train and test sets, 60-40
        X_train, X_test, y_train, y_test = train_test_split(Xr, yr[regr_model,:], test_size=0.4)
        y_test_all_pars.append(y_test)
        
        # fit regression model on train
        regr = LinearRegression().fit(X_train, y_train)
        
        # make predictions on test set
        y_pred_all_pars.append(regr.predict(X_test))
        
    # make summary scatter plot
    plot_scatter_test_pred(y_test_all_pars, y_pred_all_pars, parnames, plot_dir+'lc_scat/', 'par_preds_'+model_id+'_MCMC'+mcmc_id+'_'+n_iter+assim_type)
        
    return

if __name__=='__main__':
    main()