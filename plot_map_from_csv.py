
"""

created on Fri May 24 09:39:22 2021

@author: cfamigli

read in a csv that contains values to plot by pixel
csv created by global_map_parallel.py

"""

import numpy as np
import glob
import sys
import os
from pandas import read_csv
import readwritebinary as rwb
import anova_utilities as autil

def main():
    
    cur_dir = os.getcwd() + '/'
    csv_name = sys.argv[1] # name of csv is input by user
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf_longadapted/'+csv_name[:3]+'/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    
    to_plot = read_csv(cur_dir + '../../misc/' + csv_name, header=None) # read csv data
    to_plot = to_plot[(to_plot.iloc[:,0]<5000) & (to_plot.iloc[:,1]>-9999)] # ensure that first column only contains real pixel values (nothing numbered over 5000, which can happen due to read error)
    
    # plot map
    # first column of csv contains pixel numbers
    # second column of csv contains values to plot
    
    pixels_to_plot = to_plot.iloc[:,0].tolist()
    values_to_plot = to_plot.iloc[:,1].tolist()
    autil.plot_map(nrows=46, ncols=73, land_pixel_list=[file[-8:-4] for file in glob.glob(cur_dir + cbf_dir + '*.cbf')], 
        pixel_value_list=list(map(str, pixels_to_plot)), value_list=values_to_plot,
        savepath=cur_dir+plot_dir+'maps/', savename=csv_name[:-4])
    
    return

if __name__=='__main__':
    main()
