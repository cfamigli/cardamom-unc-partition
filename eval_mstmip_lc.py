
"""

created on Wed Jun 2 13:16:12 2021

@author: cfamigli

evaluate land cover composition of GEOSCHEM grid using fine scale MSTMIP maps

"""

import numpy as np
import readwritebinary as rwb
import os
import glob
import sys
import anova_utilities as autil
import matplotlib
import matplotlib.pyplot as plt

def no_water_pixels(fracs):
    assert fracs.shape[0]==48 # number of land classes
    fracs = fracs.compress(fracs[0,]!=1, axis=1)
    return fracs

def plot_pie(pie_data, pixel_num, latlon, savepath):
    plt.figure(figsize=(3,3))
    labels = range(48)
    cl = dict(zip(labels, plt.cm.Spectral(np.linspace(0, 1, len(labels)))))
    
    # ensure that water(=0) is not plotted
    if np.nansum(pie_data)<1:
        plt.pie(pie_data*1e9, labels=labels, colors=[cl[key] for key in labels])
    else:
        plt.pie(pie_data, labels=labels, colors=[cl[key] for key in labels])
    plt.title('Pixel: ' + pixel_num +('\nlat/lon: ' + str(latlon[0])))
    plt.tight_layout()
    plt.savefig(savepath + pixel_num + '.png')
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
    misc_dir = cur_dir + '/../../misc/'
    cbf_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbf_longadapted/811/'
    plot_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    
    # load map containing the location of each mstmip pixel on the GEOSCHEM grid
    pixel_nums = np.load(misc_dir + 'mstmip_pixel_nums.npy')
    
    # load map of biome fractions from mstmip
    with np.load(misc_dir + 'mstmip_biome_frac.npz') as data:
        biome_frac = data['arr_0']
        
    # load list of land pixels
    files = glob.glob(cbf_dir + '*.cbf')
    pixels = list(set([file[-8:-4] for file in files]))
    
    # initialize vectors to hold maximum fraction and class of maximum fraction
    max_frac = np.ones(len(pixels))*np.nan
    max_frac_class = np.ones(len(pixels))*np.nan
    
    for pixel in pixels:
        print(pixel)
        ind = pixels.index(pixel)
        
        locs = [pixel_nums==float(pixel)][0]
        fracs_at_geos_pixel = no_water_pixels(biome_frac[:,locs])
        
        av_fracs = np.nanmean(fracs_at_geos_pixel, axis=1) # average biome fraction across mstmip pixels within coarse pixel
        max_frac[ind] = max(av_fracs) # dominant fraction
        max_frac_class[ind] = np.argmax(av_fracs)

        #plot_pie(av_fracs, pixel, autil.rowcol_to_latlon([pixel]), plot_dir+'pie/')
     
    plot_dist_of_fracs(max_frac, max_frac_class, plot_dir+'pie/', 'summ')   
    
    return

if __name__=='__main__':
    main()