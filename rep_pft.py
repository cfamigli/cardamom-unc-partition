
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
    max_frac = np.nanmax(av_fracs, axis=0)
    rep_pixels = []
    for i in range(len(max_frac)):
        rows = np.where(av_fracs[:,i]==max_frac[i])[0]
        rep_pixels.append([pixels[row] for row in rows])
    return rep_pixels

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
    
    # get list of average pft fractions by pixel
    av_fracs = np.ones((len(pixels), n_classes))*np.nan
    for pixel in pixels:
        ind = pixels.index(pixel)
        #if np.mod(ind, 10)==0: print(ind)
        
        # get lc information
        locs = [pixel_nums==float(pixel)][0]
        fracs_at_geos_pixel = no_water_pixels(biome_frac[:,locs])
        av_fracs[ind,:] = np.nanmean(fracs_at_geos_pixel, axis=1) # average biome fraction across mstmip pixels within coarse pixel
    
    reps = find_rep(av_fracs, pixels)
    
    return

if __name__=='__main__':
    main()