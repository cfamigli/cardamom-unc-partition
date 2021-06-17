
"""

created on Fri Jun 4 10:30:30 2021

@author: cfamigli

using Globcover, assign pixels' dominant pft and aggregate parameters within grouped pixels

"""

import numpy as np
import readwritebinary as rwb
from pandas import read_pickle, read_csv
import os
import glob
import sys
from random import sample
import anova_utilities as autil
from eval_globcover_lc import remove_water_nodata_pixels, append_all_types
import matplotlib
import matplotlib.pyplot as plt

def main():
    
    # set run information to read
    model_id = sys.argv[1]
    mcmc_id = sys.argv[2] # 119 for normal, 3 for DEMCMC
    n_iter = sys.argv[3]
    ens_size = 500
    assim_type = '_longadapted'
    max_chains = 8
    
    # set directories
    cur_dir = os.getcwd() + '/'
    misc_dir = cur_dir + '/../../misc/'
    cbf_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbf' + assim_type+'/' + model_id + '/'
    cbr_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbr' + assim_type+'/' + model_id + '/'
    plot_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = autil.get_parnames('../../misc/', model_id)
    
    # load globcover data
    gl = read_pickle(misc_dir+'globcover_to_card.pkl')
    
    # load labels
    gl_lbls = list(read_csv(misc_dir+'Globcover2009_Legend.csv')['Value'].values)
    n_classes = len(gl_lbls)
        
    # load list of land pixels
    pixels = list(set([file[-8:-4] for file in glob.glob(cbf_dir + '*.cbf')]))
    
    # load list of cbrs
    files = glob.glob(cbr_dir+'*MCMC'+mcmc_id+'_'+n_iter+'_*.cbr')
    
    # assign dominant pft
    dom_pft_lst = []
    for pixel in pixels:
        ind = pixels.index(pixel)
        if np.mod(ind, 10)==0: print(ind)
        
        # get lc information
        types_at_geos_pixel, counts_at_geos_pixel = gl.loc[gl['pixel']==pixel]['types'].values[0][0], gl.loc[gl['pixel']==pixel]['counts'].values[0][0]
        
        types_at_geos_pixel, counts_at_geos_pixel = remove_water_nodata_pixels(types_at_geos_pixel, counts_at_geos_pixel)
        types_at_geos_pixel, counts_at_geos_pixel = append_all_types(types_at_geos_pixel, counts_at_geos_pixel, gl_lbls)
        
        if np.sum(counts_at_geos_pixel)>0:
            av_fracs = counts_at_geos_pixel/np.sum(counts_at_geos_pixel) # average biome fraction across mstmip pixels within coarse pixel
            
        dom_pft_lst.append(types_at_geos_pixel[np.argmax(av_fracs)]) # append class with maximum of average biome fraction vector
        
    autil.plot_map_discrete_cmap(nrows=46, ncols=73, land_pixel_list=pixels, pixel_value_list=pixels, value_list=dom_pft_lst, vmin=min(gl_lbls), vmax=max(gl_lbls), cmap='gist_earth', ncolors=n_classes,
        savepath=plot_dir+'maps/', savename='dom_pft_globcover_' + model_id + assim_type+ '_MCMC' + mcmc_id + '_' + n_iter)
        
    # get list of unique dominant pfts and corresponding indices
    '''dom_pft_unique = np.unique(dom_pft_lst)
    inds_unique = [np.argwhere(i==dom_pft_lst).reshape(-1,) for i in np.unique(dom_pft_lst)]
    
    # initialize list that will contain aggregated parameter sets for each dominant pft
    dom_pft_median_pars = []
    
    # run through each dominant pft
    for pft in range(len(dom_pft_unique)):
        print('pft ' + str(dom_pft_unique[pft]) + ' . . . ')
        
        # get list of pixels with this dominant pft
        pixel_pft = [pixels[i] for i in inds_unique[pft]]
        
        cbr_pft = np.ones((len(pixel_pft)*max_chains, ens_size, len(parnames)))*np.nan # matrix of cbrs (stacked along axis=0) for each pixel-chain
        cbr_count = 0
        for pixel in pixel_pft:
            cbr_files = glob.glob(cbr_dir + '*MCMC'+mcmc_id+'_'+n_iter+'_' + pixel + '_*.cbr')
            cbr_files.sort()
            
            # read parameters and compute gelman rubin
            for cbr_file in cbr_files:
                print(cbr_file[-10:-4])
                cbr_chain = rwb.read_cbr_file(cbr_file, {'nopars': len(parnames)})
                
                if np.shape(cbr_chain)[0]==ens_size:
                    cbr_pft[cbr_count,:,:] = cbr_chain
                else:
                    print('incorrect ensemble size')
                    
                cbr_count+=1
        
        meds = np.nanmedian(np.nanmedian(cbr_pft, axis=1), axis=0)
        assert len(meds)==len(parnames)
        dom_pft_median_pars.append(meds)
        
    print(dom_pft_median_pars)'''
            
        
    return

if __name__=='__main__':
    main()