

"""
 
created on Thu Jun 24 10:06:01 2021

@author: cfamigli

compare rmse between two models

"""

import glob
import os
import sys
import numpy as np
from random import sample
import anova_utilities as autil
import readwritebinary as rwb
import itertools
from netCDF4 import Dataset
from sklearn.metrics import mean_squared_error
from pandas import DataFrame, read_pickle
import matplotlib
from matplotlib import pyplot as plt

def rmse_real_numbers_only(pred, obs):
    # pred is a cardamom time series ensemble
    # obs is a time series
    pred_med = np.nanmedian(pred, axis=0)
    if len(pred_med)>len(obs):
        pred_med=pred_med[:-1]
    good_inds = (pred_med!=-9999) & (obs!=-9999)
    rmse = mean_squared_error(pred_med[good_inds], obs[good_inds], squared=False)
    return rmse 

def plot_bar(rmse_dfs):
    


def main():
    
    ### set specifications
    model_id = sys.argv[1]
    run_type = 'ALL' 
    mcmc_id = '119'
    n_iter = '40000000'
    ens_size = 500
    assim_type = '_longadapted'
    
    ### set directories
    cur_dir = os.getcwd() + '/'
    cbf_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
    cbr_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
    output_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
    plot_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = autil.get_parnames(cur_dir + '../../misc/', model_id)
    
    # get list of cbfs
    os.chdir(cbf_dir)
    cbf_files = glob.glob('*.cbf')
    cbf_files.sort()
    os.chdir(cur_dir) 
    
    # initialize lists of pixel names and rmses 
    pixels_plot = []
    nbe_rmse, lai_rmse = [], []
    
    for cbf_file in cbf_files:
        print(cbf_file, cbf_files.index(cbf_file))
        
        cbf_pixel = rwb.read_cbf_file(cbf_dir + cbf_file)
        pixel = cbf_file[-8:-4]
        
        cbr_files = glob.glob(cbr_dir + '*MCMC'+mcmc_id+'_'+n_iter+'_' + pixel + '_*.cbr')
        cbr_files = sorted(cbr_files, key=lambda x: int(x.partition(pixel+'_')[-1].partition('.cbr')[0]))
        
        # get all possible XX member combinations of cbr files 
        n_chains_to_converge = 4
        cbr_files_all_subsets = [list(i) for i in itertools.combinations(cbr_files, n_chains_to_converge)]
        
        continue_check = True
        for subset in cbr_files_all_subsets:

            if continue_check:
                
                # read parameters and compute gelman rubin
                cbr_chain_list = []
                
                for cbr_file in subset:
                    cbr_chain = rwb.read_cbr_file(cbr_file, {'nopars': len(parnames)})
                    cbr_chain = autil.modulus_Bday_Fday(cbr_chain, parnames)
                    
                    if np.shape(cbr_chain)[0]==ens_size:
                        cbr_chain_list.append(cbr_chain)
                        
                if len(cbr_chain_list)>1:
                    gr = autil.gelman_rubin(cbr_chain_list)
                
                    if sum(gr<1.2)/len(parnames)>=0.9:
                        continue_check = False
                        cbr_agg = np.vstack(cbr_chain_list)
                        pixels_plot.append(pixel)
                        best_subset = subset.copy()
                        
                else:
                    gr = np.nan
        
        # if there is a convergent subset, read fluxes and pools
        if not continue_check: 
            convergent_chain_nums = [el.partition('.cbr')[0].partition(pixel)[-1][1:] for el in best_subset]
            convergent_files = [el.partition('.cbr')[0].partition(model_id+'/')[-1] for el in best_subset]
            
            flux_pixel = []
            pool_pixel = []
    
            for filename in convergent_files: 
                flux_chain = rwb.readbinarymat(output_dir + 'fluxfile_' + filename+'.bin', [cbf_pixel['nodays'], autil.get_nofluxes_nopools_lma(model_id)[0]])
                pool_chain = rwb.readbinarymat(output_dir + 'poolfile_' + filename+'.bin', [cbf_pixel['nodays']+1, autil.get_nofluxes_nopools_lma(model_id)[1]])
                
                if (flux_chain.shape[0]==ens_size) & (pool_chain.shape[0]==ens_size): 
                    flux_pixel.append(flux_chain)
                    pool_pixel.append(pool_chain)
            
            nbe_pred = autil.get_output('NBE', model_id, np.vstack(flux_pixel), np.vstack(pool_pixel), cbr_agg, autil.get_nofluxes_nopools_lma(model_id)[2])
            lai_pred = autil.get_output('LAI', model_id, np.vstack(flux_pixel), np.vstack(pool_pixel), cbr_agg, autil.get_nofluxes_nopools_lma(model_id)[2])
            nbe_obs, lai_obs = cbf_pixel['OBS']['NBE'], cbf_pixel['OBS']['LAI']
            
            nbe_rmse.append(rmse_real_numbers_only(nbe_pred, nbe_obs))
            lai_rmse.append(rmse_real_numbers_only(lai_pred, lai_obs))
            print(rmse_real_numbers_only(nbe_pred, nbe_obs), rmse_real_numbers_only(lai_pred, lai_obs))
            
    
    autil.plot_map(nrows=46, ncols=73, land_pixel_list=[file[-8:-4] for file in glob.glob(cbf_dir + '*.cbf')], pixel_value_list=pixels_plot, value_list=nbe_rmse, savepath=plot_dir+'maps/', savename='rmse_nbe_' + model_id + assim_type+ '_MCMC' + mcmc_id + '_' + n_iter)
    autil.plot_map(nrows=46, ncols=73, land_pixel_list=[file[-8:-4] for file in glob.glob(cbf_dir + '*.cbf')], pixel_value_list=pixels_plot, value_list=lai_rmse, savepath=plot_dir+'maps/', savename='rmse_lai_' + model_id + assim_type+ '_MCMC' + mcmc_id + '_' + n_iter)
    
    rmse_df = DataFrame(list(zip(pixels_plot, nbe_rmse, lai_rmse)))
    rmse_df.columns = ['pixel','nbe_rmse','lai_rmse']
    rmse_df.to_pickle(cur_dir + '../../misc/rmse_' + model_id + assim_type+ '_MCMC' + mcmc_id + '_' + n_iter + '.pkl')
    
    
    #################################################################################################################################################################
    # analyze regionally
    
    '''region_mask = Dataset(cur_dir + '../../misc/fourregion_maskarrays.nc')
    region_mask.set_auto_mask(False)
    regionmat, lat, lon = region_mask['4region'][:], region_mask['lat'][:], region_mask['lon'][:]
    lat[0] = -90
    lat[-1] = 90
    
    model_ids = ['811', '911']
    rmse_dfs = []
    for model_id in model_ids:
        rmse_df = read_pickle(cur_dir + '../../misc/rmse_' + model_id + assim_type+ '_MCMC' + mcmc_id + '_' + n_iter + '.pkl')
        rmse_df.columns = ['pixel','nbe_rmse','lai_rmse']
        
        regions = []
        for pixel in rmse_df[rmse_df.columns[0]].tolist():
            pixlat, pixlon = rwb.rowcol_to_latlon(pixel)
            regions.append(regionmat[np.argwhere(lat==pixlat)[0][0], np.argwhere(lon==pixlon)[0][0]])
        
        rmse_df.insert(loc=1, column='region', value=regions)
        rmse_dfs.append(rmse_df)
    
    print(rmse_dfs[0].groupby('region')['nbe_rmse'].mean(), rmse_dfs[0].groupby('region')['lai_rmse'].mean())
    print(rmse_dfs[1].groupby('region')['nbe_rmse'].mean(), rmse_dfs[1].groupby('region')['lai_rmse'].mean())'''
                        
    return

if __name__=='__main__':
    main()
    