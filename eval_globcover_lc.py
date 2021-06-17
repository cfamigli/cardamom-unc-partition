
"""

created on Wed Jun 9 10:55:04 2021

@author: cfamigli

evaluate land cover composition of GEOSCHEM grid using fine scale Globcover maps

"""

import numpy as np
from pandas import read_pickle, read_csv
import os
import glob
import sys
import matplotlib
import matplotlib.pyplot as plt
import anova_utilities as autil

def remove_water_nodata_pixels(types, counts):
    ind_remove = [(types==210) | (types==230)][0]
    counts[ind_remove] = 0
    return types, counts
    
def append_all_types(types, counts, lbls):
    types_not_in_pixel = np.setdiff1d(lbls, types)
    counts_not_in_pixel = np.zeros(len(types_not_in_pixel))
    
    concat_types = np.concatenate((types, types_not_in_pixel))
    concat_counts = np.concatenate((counts, counts_not_in_pixel))
    
    inds_sm_lrg = np.argsort(concat_types)
    return concat_types[inds_sm_lrg], concat_counts[inds_sm_lrg]
    

def plot_pie(pie_data, pixel_num, labels, latlon, savepath, datasource):
    plt.figure(figsize=(3,3))
    cl = dict(zip(labels, plt.cm.Spectral(np.linspace(0, 1, len(labels)))))
    
    # ensure that water(=0) is not plotted
    if np.nansum(pie_data)<1:
        plt.pie(pie_data*1e9, labels=labels, colors=[cl[key] for key in labels])
    else:
        plt.pie(pie_data, labels=labels, colors=[cl[key] for key in labels])
    plt.title('Pixel: ' + pixel_num +('\nlat/lon: ' + str(latlon[0])))
    plt.tight_layout()
    plt.savefig(savepath + pixel_num + '_' + datasource +'.png')
    plt.close()
    return

def plot_dist_of_fracs(max_frac, max_frac_class, savepath, savename):
    fig, ax = plt.subplots(2,1, figsize=(7,7))
    ax[0].hist(max_frac, bins=48, facecolor='dodgerblue', edgecolor='k', rwidth=0.9)
    ax[0].set_xlabel('Fraction of dominant biome \ntype within coarse pixel')
    ax[1].hist(max_frac_class, bins=48, facecolor='dodgerblue', edgecolor='k', rwidth=0.9)
    #ax[1].set_xticks(range(49))
    ax[1].set_xlabel('Class of dominant biome \ntype within coarse pixel')
    plt.tight_layout()
    plt.savefig(savepath + savename + '.png')
    plt.close()
    return

def main():
    
    cur_dir = os.getcwd() + '/'
    misc_dir = cur_dir + '../../misc/'
    cbf_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbf_longadapted/811/'
    plot_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    
    # load globcover data
    gl = read_pickle(misc_dir+'globcover_to_card.pkl')
    
    # load labels
    gl_lbls = list(read_csv(misc_dir+'Globcover2009_Legend.csv')['Value'].values)
        
    # load list of land pixels
    files = glob.glob(cbf_dir + '*.cbf')
    pixels = list(set([file[-8:-4] for file in files]))
    
    # initialize vectors to hold maximum fraction and class of maximum fraction
    max_frac = np.ones(len(pixels))*np.nan
    max_frac_class = np.ones(len(pixels))*np.nan
    
    for pixel in pixels:
        print(pixel)
        ind = pixels.index(pixel)
        
        types_at_geos_pixel, counts_at_geos_pixel = gl.loc[gl['pixel']==pixel]['types'].values[0][0], gl.loc[gl['pixel']==pixel]['counts'].values[0][0]
        
        types_at_geos_pixel, counts_at_geos_pixel = remove_water_nodata_pixels(types_at_geos_pixel, counts_at_geos_pixel)
        types_at_geos_pixel, counts_at_geos_pixel = append_all_types(types_at_geos_pixel, counts_at_geos_pixel, gl_lbls)
        
        if np.sum(counts_at_geos_pixel)>0:
            av_fracs = counts_at_geos_pixel/np.sum(counts_at_geos_pixel) # average biome fraction across mstmip pixels within coarse pixel
            max_frac[ind] = max(av_fracs) # dominant fraction
            max_frac_class[ind] = types_at_geos_pixel[np.argmax(av_fracs)]
    
            plot_pie(av_fracs, pixel, gl_lbls, autil.rowcol_to_latlon([pixel]), plot_dir+'pie/', 'gl')
     
    plot_dist_of_fracs(max_frac, max_frac_class, plot_dir+'pie/', 'summ_gl_')   
    
    return

if __name__=='__main__':
    main()