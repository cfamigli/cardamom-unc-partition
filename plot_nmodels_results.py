
"""

created on Wed May 26 09:29:21 2021

@author: cfamigli

run anova approach on CARDAMOM output
repeat for various subsets of models in suite

"""

import numpy as np
import os
import glob
import sys
from pandas import DataFrame, read_pickle
import anova_utilities as autil

def main():
    
    cur_dir = os.getcwd() + '/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    df_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/processed_df/'
    assim_type = '_longadapted'
    mcmc_id = sys.argv[1]
    
    # set lists of variables and pixels
    vrs = ['NBE','cumNBE','LAI','GPP','Reco','Rauto','Rhet','lit','root','som','wood']
    pixels = ['3809','3524','2224','4170','1945','3813','4054','3264','1271','3457']
    
    for pixel in pixels:
        print(pixel)
        for var in vrs:
            print(var)
            xnmodels = range(3, 12)
            yMs = np.zeros(len(xnmodels))
            yMp = np.zeros(len(xnmodels))
            n = np.zeros(len(xnmodels))
            for i in range(len(xnmodels)):
                files = glob.glob(cur_dir + df_dir + 'summary' + assim_type + '_MCMC' + mcmc_id + '_*_' + str(xnmodels[i]) +'.pkl')
                data = read_pickle(files[-1])
                yMs[i] = data.loc[autil.subset_df_by_substring(data, pixel + '_' + var)]['Ms'].values[0]
                yMp[i] = data.loc[autil.subset_df_by_substring(data, pixel + '_' + var)]['Mp'].values[0]
                n[i] = data.loc[autil.subset_df_by_substring(data, pixel + '_' + var)]['n'].values[0]
            autil.plot_nmodel_test(xnmodels, yMs, yMp, cur_dir+plot_dir+'nmodel_test', pixel + '_' + var + '_MCMC' + mcmc_id)
    return

if __name__=='__main__':
    main()