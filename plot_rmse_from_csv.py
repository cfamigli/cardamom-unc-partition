
"""

created on Mon Aug 16 02:50:37 2021

@author: cfamigli

plot test/train rmse for cardamom parameters

"""

import sys
import os
import numpy as np
from pandas import read_pickle, read_csv
import anova_utilities as autil
import matplotlib
import matplotlib.pyplot as plt


def plot_train_test(x, train_rmse, test_rmse, parnames, savepath, savename):
    npars = len(parnames)
    ncols = 6
    nrows = 0
    while nrows<npars:
        nrows+=ncols
    nrows = int(nrows/6)
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_size_inches(13.5,12)
    
    count = 0
    for row in range(nrows):
        for col in range(ncols):
            if count<npars:

                axs[row, col].plot(x, train_rmse.iloc[:,count+1].values, c='dodgerblue', linewidth=3, label='Train')
                axs[row, col].plot(x, test_rmse.iloc[:,count+1].values, c='crimson', linewidth=3, label='Test')
                if npars>1:
                    axs[row, col].set_title(parnames[count][:20])
        
                if count==npars-1:
                    axs[row, col].legend(loc='best')
                count += 1
                
    plt.subplots_adjust(hspace = .5,wspace=.5)
    plt.tight_layout()
    plt.savefig(savepath + savename + '.png')
    plt.close()
    return



def main():
    
    # get specifications
    model_id = sys.argv[1]
    mcmc_id = sys.argv[2]
    suffix = '_rescaled_wpolys_fs'
    
    # get directories
    cur_dir = os.getcwd() + '/'
    misc_dir = '../../misc'
    plot_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    
    # get parnames
    parnames = autil.get_parnames('../../misc/', model_id)
    
    # get data
    test_rmse = read_csv('../../misc/' +model_id+'_MCMC'+mcmc_id+suffix+'_test.csv', header=0)
    train_rmse = read_csv('../../misc/'+model_id+'_MCMC'+mcmc_id+suffix+'_train.csv', header=0)
    x = test_rmse['n_features_select'].values

    # make plots
    plot_train_test(x, train_rmse, test_rmse, parnames, savepath=plot_dir+'train_test/', savename=model_id+'_MCMC'+mcmc_id+suffix)
    
    return

if __name__=='__main__':
    main()