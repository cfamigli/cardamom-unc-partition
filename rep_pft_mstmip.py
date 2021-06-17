
"""

created on Fri Jun 4 13:32:02 2021

@author: cfamigli

find most representative pixel for each PFT

"""

import numpy as np
import readwritebinary as rwb
import os
import glob
import sys
from random import sample
import anova_utilities as autil
from eval_mstmip_lc import no_water_pixels
import matplotlib
import matplotlib.pyplot as plt

def find_rep(av_fracs, pixels):
    av_fracs_copy = np.copy(av_fracs)
    max_frac = np.nanmax(av_fracs_copy, axis=0)
    max_frac_dom = np.ones(len(max_frac))*np.nan
    
    av_fracs_copy[av_fracs_copy==max_frac] = np.nan
    second_max_frac = np.nanmax(av_fracs_copy, axis=0)
    second_max_frac_dom = np.ones(len(second_max_frac))*np.nan
    
    av_fracs_copy[av_fracs_copy==second_max_frac] = np.nan
    third_max_frac = np.nanmax(av_fracs_copy, axis=0)
    third_max_frac_dom = np.ones(len(third_max_frac))*np.nan
    
    rep_pixels = []
    for i in range(len(max_frac)):
        rows = np.where(((av_fracs[:,i]==max_frac[i]) | (av_fracs[:,i]==second_max_frac[i]) | (av_fracs[:,i]==third_max_frac[i])))[0]
        rep_pixels.append([pixels[row] for row in rows]) if max_frac[i]>0 else rep_pixels.append([])
        
        if max_frac[i]>0:
            max_frac_rows = np.where(av_fracs[:,i]==max_frac[i])
            second_max_frac_rows = np.where(av_fracs[:,i]==second_max_frac[i])
            third_max_frac_rows = np.where(av_fracs[:,i]==third_max_frac[i])
            
            max_frac_dom[i] = 1 if max_frac[i]==np.max(av_fracs[max_frac_rows]) else 0
            second_max_frac_dom[i] = 1 if second_max_frac[i]==np.max(av_fracs[second_max_frac_rows]) else 0
            third_max_frac_dom[i] = 1 if third_max_frac[i]==np.max(av_fracs[third_max_frac_rows]) else 0
        
    return rep_pixels, max_frac, second_max_frac, third_max_frac, max_frac_dom, second_max_frac_dom, third_max_frac_dom
    
def plot_reps(mxs, mxdoms, lbls, savepath, savename):
    plt.figure(figsize=(14,5))
    colors = plt.cm.brg(np.linspace(0,0.9,len(mxs)))
    for pft in lbls:
        ind = lbls.index(pft)
        mx_count = 0
        for mx, mxdom in zip(mxs, mxdoms):
            if mx[ind]>0:
                if mxdom[ind]==1:
                    plt.scatter(ind, mx[ind], s=150, facecolor=colors[mx_count], edgecolor=colors[mx_count], linewidth=1.5)
                else:
                    plt.scatter(ind, mx[ind], s=150, facecolor='none', edgecolor=colors[mx_count], linewidth=2.5)
            mx_count += 1
        plt.axvline(ind, linewidth=0.5, c='gainsboro', zorder=0)
    plt.xticks(np.arange(len(lbls)), lbls)
    plt.xlabel('Class')
    plt.ylabel('Fraction represented')
    plt.title('Representative pixels')
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
    misc_dir = cur_dir + '../../misc/'
    cbf_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbf' + assim_type+'/' + model_id + '/'
    cbr_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbr' + assim_type+'/' + model_id + '/'
    plot_dir = cur_dir + '../../../../../../../scratch/users/cfamigli/cardamom/plots/'
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
    
    # get list of average pft fractions by pixel
    av_fracs = np.ones((len(pixels), n_classes))*np.nan
    for pixel in pixels:
        ind = pixels.index(pixel)
        #if np.mod(ind, 10)==0: print(ind)
        
        # get lc information
        locs = [pixel_nums==float(pixel)][0]
        fracs_at_geos_pixel = no_water_pixels(biome_frac[:,locs])
        av_fracs[ind,:] = np.nanmean(fracs_at_geos_pixel, axis=1) # average biome fraction across mstmip pixels within coarse pixel
    
    reps, mx, mx2, mx3, mxdom, mx2dom, mx3dom = find_rep(av_fracs, pixels)
    
    plot_reps([mx, mx2, mx3], [mxdom, mx2dom, mx3dom], range(n_classes), plot_dir+'pie/', 'rep_pix_ms')
    
    return

if __name__=='__main__':
    main()