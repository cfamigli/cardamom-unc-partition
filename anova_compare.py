
"""

created on Mon May 24 12:03:01 2021

@author: cfamigli

compare anova results between MCMC and DEMCMC

"""

import numpy as np
import readwritebinary as rwb
import os
import glob
import sys
from pandas import DataFrame, read_pickle
import matplotlib
import matplotlib.pyplot as plt
import anova_utilities as autil

def main():
    cur_dir = os.getcwd() + '/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    df_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/processed_df/'
    
    assim_type = '_longadapted'
    mcmc_ids = ['3','119']
    Mp = []
    Ms = []
    for mcmc_id in mcmc_ids:
        pkl_files = glob.glob(cur_dir+df_dir+'summary'+assim_type+'_MCMC'+mcmc_id+'*.pkl')
        pkl_files.sort() # get most recent 
        
        Mp.append(read_pickle(pkl_files[-1])['Mp'])
        Ms.append(read_pickle(pkl_files[-1])['Ms'])
        
    autil.plot_scatter_anova_compare(Mp, Ms, ['DEMCMC','MCMC'], cur_dir+plot_dir+'anova_summary', 'mcmc_compare')
    return

if __name__=='__main__':
    main()