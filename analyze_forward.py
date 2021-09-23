
'''

created on Mon Aug 30 08:56:02 2021

@author: cfamigli

read in PFT, EF, and optimal forward runs and plot/compare

'''

import numpy as np
import readwritebinary as rwb
from pandas import read_csv, read_pickle, DataFrame, to_pickle
from itertools import compress
import os
import glob
import sys
import csv
import warnings
warnings.filterwarnings('ignore')
import anova_utilities as autil
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import pearsonr

def get_types_at_pixel(gl_df, pixel):
    # determine which pfts are present for a given pixel id
    cols = gl_df.columns.values
    mask = gl_df.loc[gl_df['pixel']==int(pixel)].gt(0.0).values
    nonzero_types = [cols[x].tolist() for x in mask][0][1:]
    return nonzero_types
    
def remove_nan(arr, pixel_lst):
    # index array and list based on real numbers in array
    good_inds = np.isfinite(arr)
    return arr[good_inds], list(compress(pixel_lst, good_inds))
    
def remove_nan_from_arrs(arr1, arr2):
    # index arrays to save good inds
    good_inds = np.isfinite(arr1) & np.isfinite(arr2)
    return arr1[good_inds], arr2[good_inds]
    
def best_param_nonancol(arrs, pixel_lst):
    # filter stacked arrs to remove all nan columns
    stacked = np.vstack(arrs)
    good_inds = np.argwhere(~np.isnan(stacked).any(axis=0)).reshape(-1) # indices of columns that have all real numbers
    return np.nanargmin(stacked[:,good_inds], axis=0), [pixel_lst[i] for i in good_inds]
    
def err_rgb_triplets(arrs, pixel_lst):
    # get rgb triplets from stacked array of opt, pft, and ef errors
    n_arrs = len(arrs)
    stacked = np.vstack(arrs)
    
    n_triplets = stacked.shape[1]
    triplets = []
    pixels_nonan = []
    for ind in range(n_triplets):
        vals = stacked[:,ind]
        
        if ~np.isnan(np.sum(vals)):
            triplets.append(list(vals/np.sum(vals))+[1.])
        
        else:
            triplets.append(list(vals/np.sum(vals))+[0.])

    return triplets

    
    
def timeseries_decompose(outputs_opt, outputs_pft, outputs_ef, pixel, savepath, savename):
    
    # decompose each time series
    opt_decomposed = seasonal_decompose(np.nanmedian(outputs_opt, axis=0), model='additive', period=12)
    pft_decomposed = seasonal_decompose(np.nanmedian(outputs_pft, axis=0), model='additive', period=12)
    ef_decomposed = seasonal_decompose(np.nanmedian(outputs_ef, axis=0), model='additive', period=12)
    
    def plot_decomp(res, axes, title, color):
        axes[0].plot(res.observed, c=color)
        axes[0].set_ylabel('Observed')
        axes[0].set_title(title)
        axes[1].plot(res.trend, c=color)
        axes[1].set_ylabel('Trend')
        axes[2].plot(res.seasonal, c=color)
        axes[2].set_ylabel('Seasonal')
        axes[3].plot(res.resid, c=color)
        axes[3].set_ylabel('Residual')
        return
    
    fig, axes = plt.subplots(ncols=3, nrows=4, sharex=True, figsize=(12,8))
    plot_decomp(opt_decomposed, axes[:,0], 'Optimal', color='dodgerblue')
    plot_decomp(pft_decomposed, axes[:,1], 'PFT', color='orangered')
    plot_decomp(ef_decomposed, axes[:,2], 'EF', color='limegreen')
    plt.suptitle('Pixel: '+ str(pixel)+('\nlat/lon: ' + str(autil.rowcol_to_latlon([pixel])[0])))
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(savepath+savename+'.png')
    plt.close()
    
    # get comparison metrics
    #pearsonr
    opt_pft_trend = pearsonr(opt_decomposed.trend[(np.isfinite(opt_decomposed.trend)) & (np.isfinite(pft_decomposed.trend))], pft_decomposed.trend[(np.isfinite(opt_decomposed.trend)) & (np.isfinite(pft_decomposed.trend))])[0]
    opt_ef_trend = pearsonr(opt_decomposed.trend[(np.isfinite(opt_decomposed.trend)) & (np.isfinite(ef_decomposed.trend))], ef_decomposed.trend[(np.isfinite(opt_decomposed.trend)) & (np.isfinite(ef_decomposed.trend))])[0]
    #pearsonr
    opt_pft_seas = pearsonr(opt_decomposed.seasonal[(np.isfinite(opt_decomposed.seasonal)) & (np.isfinite(pft_decomposed.seasonal))], pft_decomposed.seasonal[(np.isfinite(opt_decomposed.seasonal)) & (np.isfinite(pft_decomposed.seasonal))])[0]
    opt_ef_seas = pearsonr(opt_decomposed.seasonal[(np.isfinite(opt_decomposed.seasonal)) & (np.isfinite(ef_decomposed.seasonal))], ef_decomposed.seasonal[(np.isfinite(opt_decomposed.seasonal)) & (np.isfinite(ef_decomposed.seasonal))])[0]
    #mean
    opt_mean = np.nanmean(opt_decomposed.observed)
    opt_mean_25 = np.nanmean(np.nanpercentile(outputs_opt, 25, axis=0))
    opt_mean_75 =np.nanmean(np.nanpercentile(outputs_opt, 75, axis=0))
    pft_mean = np.nanmean(pft_decomposed.observed)
    ef_mean = np.nanmean(ef_decomposed.observed)
    
    pft_mean_within_opt_unc = 1 if (pft_mean<=opt_mean_75) & (pft_mean>=opt_mean_25) else 0
    ef_mean_within_opt_unc = 1 if (ef_mean<=opt_mean_75) & (ef_mean>=opt_mean_25) else 0
    
    return opt_pft_trend, opt_ef_trend, opt_pft_seas, opt_ef_seas, opt_mean, pft_mean, ef_mean, pft_mean_within_opt_unc, ef_mean_within_opt_unc

def plot_decomposed(trends, seas, means, meanbools, savepath, savename):
    fig = plt.figure(figsize=(7,7))
    
    ax = fig.add_subplot(2,2,1)
    ax.hist2d(remove_nan_from_arrs(trends[0], trends[1])[0], remove_nan_from_arrs(trends[0], trends[1])[1], (25,25), cmap=plt.cm.cubehelix_r)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_xlabel('Corr(Opt, PFT)')
    ax.set_ylabel('Corr(Opt, EF)')
    ax.set_title('Trend')
    
    ax = fig.add_subplot(2,2,2)
    ax.hist2d(remove_nan_from_arrs(seas[0], seas[1])[0], remove_nan_from_arrs(seas[0], seas[1])[1], (25,25), cmap=plt.cm.cubehelix_r)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_xlabel('Corr(Opt, PFT)')
    ax.set_ylabel('Corr(Opt, EF)')
    ax.set_title('Seasonal Cycle')
    
    ax = fig.add_subplot(2,2,3)
    ax.scatter(means[0][meanbools[0]==1], means[1][meanbools[0]==1], facecolor='orangered', edgecolor='k', linewidth=0.5, s=35)
    ax.scatter(means[0][meanbools[0]==0], means[1][meanbools[0]==0], facecolor='None', edgecolor='k', linewidth=0.5, s=35)
    ax.set_xlim([-0.5,0.5])
    ax.set_ylim([-0.5,0.5])
    ax.set_xlabel('Optimal')
    ax.set_ylabel('PFT')
    ax.set_title('Mean')
    
    ax = fig.add_subplot(2,2,4)
    ax.scatter(means[0][meanbools[1]==1], means[2][meanbools[1]==1], facecolor='limegreen', edgecolor='k', linewidth=0.5, s=35)
    ax.scatter(means[0][meanbools[1]==0], means[2][meanbools[1]==0], facecolor='None', edgecolor='k', linewidth=0.5, s=35)
    ax.set_xlim([-0.5,0.5])
    ax.set_ylim([-0.5,0.5])
    ax.set_xlabel('Optimal')
    ax.set_ylabel('EF')
    ax.set_title('Mean')
    
    plt.tight_layout()
    plt.savefig(savepath + savename + '.png')
    plt.close()
    
    return

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################


def main():
    
    # set run information to read
    model_id = sys.argv[1]
    mcmc_id = sys.argv[2] # 119 for normal, 3 for DEMCMC
    n_iter = sys.argv[3]
    nbe_optimization = sys.argv[4] # 'OFF' or 'ON'
    ens_size = 500
    assim_type = sys.argv[5]
    
    # set directories
    cur_dir = os.getcwd() + '/'
    misc_dir = cur_dir + '../../misc/'
    cbf_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbf' + assim_type+'/' + model_id + '/'
    cbr_opt_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbr' + assim_type+'/' + model_id + '/'
    cbr_ef_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbr' + assim_type+'_ef/' + model_id + '/'
    cbr_pft_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbr' + assim_type+'_pft/' + model_id + '/'
    output_opt_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
    output_ef_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'_ef/' + model_id + '/'
    output_pft_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'_pft/' + model_id + '/'
    plot_dir = cur_dir + '../../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = autil.get_parnames('../../misc/', model_id)
    
    # get list of cbfs
    os.chdir(cbf_dir)
    cbf_files = glob.glob('*.cbf')
    cbf_files.sort()
    pixel_lst = []
    os.chdir(cur_dir + '/../')
    
    # initialize lists for error maps
    card_unc, opt_obs_err, pft_obs_err, ef_obs_err, obs_std = np.zeros(len(cbf_files))*np.nan, np.zeros(len(cbf_files))*np.nan, np.zeros(len(cbf_files))*np.nan, np.zeros(len(cbf_files))*np.nan, np.zeros(len(cbf_files))*np.nan
    opt_pft_trend, opt_ef_trend, opt_pft_seas, opt_ef_seas, opt_mean, pft_mean, ef_mean = np.zeros(len(cbf_files))*np.nan, np.zeros(len(cbf_files))*np.nan, np.zeros(len(cbf_files))*np.nan, np.zeros(len(cbf_files))*np.nan, np.zeros(len(cbf_files))*np.nan, np.zeros(len(cbf_files))*np.nan, np.zeros(len(cbf_files))*np.nan
    pft_mean_within_opt_unc, ef_mean_within_opt_unc = np.zeros(len(cbf_files))*np.nan, np.zeros(len(cbf_files))*np.nan
    
    
    ################################################## iterate through pixels ##################################################
    ############################################################################################################################
    
    include_ef = True
    include_pft = True
    include_opt = True
    write_txt_sh_pft_rerun = True
    
    # initialize
    n_fluxes = autil.get_nofluxes_nopools_lma(model_id)[0]
    n_pools = autil.get_nofluxes_nopools_lma(model_id)[1]
    
    # load list of globcover labels
    gl_lbls = list(read_csv(misc_dir+'Globcover2009_Legend.csv')['Value'].values)
    n_classes = len(gl_lbls)
    
    # load globcover csv for av_fracs determination
    gl_fracs = read_csv(misc_dir+'globcover_fracs.csv', header=0)
    
    # load bestchains for cbr_files
    conv_chains = read_pickle(cbr_opt_dir + model_id + assim_type + '_ALL' + '_MCMC'+mcmc_id + '_'+n_iter+'_best_subset.pkl')
    conv_chains.columns = ['pixel','bestchains','conv'] #rename columns for easier access
    
    # create csv to track pft reruns
    pft_rerun_filename = 'pft_rerun_'+ model_id + assim_type + '_MCMC'+mcmc_id + '_'+n_iter+'.csv'
    pft_rerun = open(misc_dir + pft_rerun_filename, 'w')
    w = csv.writer(pft_rerun)
    
    # run through all pixels
    for cbf_file in cbf_files: 
        ind = cbf_files.index(cbf_file)
        pixel = cbf_file[-8:-4]
        pixel_lst.append(pixel)
        print(pixel)
        
        # read in fracs and types for pixel
        if int(pixel) in gl_fracs['pixel'].values: 
            
            fracs_at_pixel = gl_fracs.loc[gl_fracs['pixel']==int(pixel)].values[0][1:]
            types_at_pixel = get_types_at_pixel(gl_fracs, pixel)
            
        else:
            
            fracs_at_pixel = np.zeros(len(gl_lbls))
            types_at_pixel = []
        
        # read in cbf
        cbf_pixel = rwb.read_cbf_file(cbf_dir + cbf_file)
        nsteps = cbf_pixel['nodays']
        
        ################################################## get PFT forward runs ##################################################
        ##########################################################################################################################
        
        can_plot_pft = False
        if include_pft:
            
            pixel_rerun = []
            pft_spec = '5rp_'
            
            # initialize matrices to hold weighted average of fluxes and pools
            flux_pft_pixel = np.zeros((1, nsteps, n_fluxes))
            pool_pft_pixel = np.zeros((1, nsteps+1, n_pools))
            #flux_pft_pixel = np.zeros((ens_size, nsteps, n_fluxes))
            #pool_pft_pixel = np.zeros((ens_size, nsteps+1, n_pools))
            
            # read all forward runs (each pft's run) for a given pixel
            print(types_at_pixel)
            for pft in types_at_pixel:
                
                suffix = cbf_file[:-9]+'_MCMC'+mcmc_id+'_'+n_iter+'_PFT'+str(int(pft))+'_forward_'+pixel+'.bin'#cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iter+'_PFT'+str(int(pft))+'_'+pixel+'.bin'

                if (len(glob.glob(output_pft_dir + 'fluxfile_'+suffix))>0) & (len(glob.glob(output_pft_dir + 'poolfile_'+suffix))>0):
                    print(str(int(pft)))
                    
                    flux_pft = rwb.readbinarymat(output_pft_dir + 'fluxfile_'+suffix, [nsteps, n_fluxes])
                    pool_pft = rwb.readbinarymat(output_pft_dir + 'poolfile_'+suffix, [nsteps+1, n_pools])
                    #autil.plot_general_timeseries(autil.get_output('NBE', model_id, flux_pft, pool_pft, cbr_data=[], lma_ind=autil.get_nofluxes_nopools_lma(model_id)[2]), 'NBE', cbf_pixel, plot_dir+'timeseries/pft/', model_id + '_MCMC'+mcmc_id + '_'+n_iter + '_' + pixel + '_'+str(int(pft))+'.png')
                    
                    # add each flux and pool matrix (corresponding to each pft) according to pft fractions, as weighted average
                    flux_pft[np.isnan(flux_pft)] = 0.
                    pool_pft[np.isnan(pool_pft)] = 0.
                    
                    if (flux_pft.shape[0]>0) & (pool_pft.shape[0]>0):
                        
                        lbl_ind = gl_lbls.index(int(pft))
                        flux_pft_pixel += flux_pft * fracs_at_pixel[lbl_ind]
                        pool_pft_pixel += pool_pft * fracs_at_pixel[lbl_ind]

                        can_plot_pft = True
                        
                    else:
                        pixel_rerun.append(pft)
                        
                else:
                    pixel_rerun.append(pft)
                        
            if len(pixel_rerun)>0:
                w.writerow([pixel] + pixel_rerun)
                        
        ################################################ get optimal forward runs ################################################
        ##########################################################################################################################
        
        can_plot_opt = False
        if include_opt:
            
            # get pixel's convergent chain numbers
            
            if pixel in conv_chains['pixel'].values:
                
                best_chains = conv_chains.loc[conv_chains['pixel']==pixel]['bestchains'].values[0][1:]
                flux_opt, pool_opt = [], []
                
                # aggregate best chain outputs into one list 
                for chain in best_chains:
                    suffix = cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iter+'_'+pixel+'_'+chain+'.bin'
                    
                    if (len(glob.glob(output_opt_dir + 'fluxfile_'+suffix))>0) & (len(glob.glob(output_opt_dir + 'poolfile_'+suffix))>0):
                        
                        flux_opt.append(rwb.readbinarymat(output_opt_dir + 'fluxfile_'+suffix, [nsteps, n_fluxes]))
                        pool_opt.append(rwb.readbinarymat(output_opt_dir + 'poolfile_'+suffix, [nsteps+1, n_pools]))
                        
                        can_plot_opt = True
    
                    
                # stack list elements for plotting
                flux_opt = np.vstack(flux_opt)
                pool_opt = np.vstack(pool_opt)
           
            
        ################################################### get EF forward runs ###################################################
        ###########################################################################################################################
        
        can_plot_ef = False
        if include_ef:
            
            ef_spec = 'clipped_PLS_soilgrids_poolobs_rescaled_forward_'
            # if 'wpolys' in ef_spec: use '_MCMC'
            # else: use 'MCMC'
            suffix = cbf_file[:-9]+'_MCMC'+mcmc_id+'_'+n_iter+'_'+ef_spec+pixel+'.bin'#cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iter+'_EF_'+pixel+'.bin'

            if (len(glob.glob(output_ef_dir + 'fluxfile_'+suffix))>0) & (len(glob.glob(output_ef_dir + 'poolfile_'+suffix))>0):
                
                flux_ef = rwb.readbinarymat(output_ef_dir + 'fluxfile_'+suffix, [nsteps, n_fluxes])
                pool_ef = rwb.readbinarymat(output_ef_dir + 'poolfile_'+suffix, [nsteps+1, n_pools])
                
                can_plot_ef = True

        
        ##################################################### plot and compare ####################################################
        ###########################################################################################################################
        
        
        can_decompose = True if (can_plot_opt) & (can_plot_pft) & (can_plot_ef) else False
        
        # plot optimal and pft predictions together
        output_opt = autil.get_output('NBE', model_id, flux_opt, pool_opt, cbr_data=[], lma_ind=autil.get_nofluxes_nopools_lma(model_id)[2]) if (include_opt) & (can_plot_opt) else np.ones(nsteps)*np.nan
        output_pft = autil.get_output('NBE', model_id, flux_pft_pixel, pool_pft_pixel, cbr_data=[], lma_ind=autil.get_nofluxes_nopools_lma(model_id)[2]) if (include_pft) & (can_plot_pft) else np.ones(nsteps)*np.nan
        output_ef = autil.get_output('NBE', model_id, flux_ef, pool_ef, cbr_data=[], lma_ind=autil.get_nofluxes_nopools_lma(model_id)[2]) if (include_ef) & (can_plot_ef) else np.ones(nsteps)*np.nan
        
        card_unc[ind], opt_obs_err[ind], pft_obs_err[ind], ef_obs_err[ind], obs_std[ind] = autil.plot_opt_pft_ef_timeseries(output_opt, output_pft, output_ef, 'NBE', pixel, autil.rowcol_to_latlon([pixel]), 
            cbf_pixel, err_v_obs=False, savepath=plot_dir+'forward_compare/timeseries/'+model_id+'/', title=model_id + '_MCMC'+mcmc_id + '_'+n_iter + '_' + pft_spec+ef_spec+pixel + '.png')
            
        if can_decompose:
            opt_pft_trend[ind], opt_ef_trend[ind], opt_pft_seas[ind], opt_ef_seas[ind], opt_mean[ind], pft_mean[ind], ef_mean[ind], pft_mean_within_opt_unc[ind], ef_mean_within_opt_unc[ind] = timeseries_decompose(output_opt, output_pft, 
                output_ef, pixel, savepath=plot_dir+'forward_compare/decomp/'+model_id+'/',
                savename=model_id + '_MCMC'+mcmc_id + '_'+n_iter + '_' + pft_spec+ef_spec+pixel)
        
    # close csv for rerun tracking
    pft_rerun.close()

    
    # plot decomposition results
    
    plot_decomposed([opt_pft_trend, opt_ef_trend], [opt_pft_seas, opt_ef_seas], [opt_mean, pft_mean, ef_mean], [pft_mean_within_opt_unc, ef_mean_within_opt_unc], savepath=plot_dir+'forward_compare/decomp/'+model_id+'/', savename=model_id+'_MCMC'+mcmc_id+'_'+n_iter+'_'+pft_spec+ef_spec)
    
    
    # plot error maps
    for data, plot_title, vmin, vmax in zip([card_unc, opt_obs_err, pft_obs_err, ef_obs_err, obs_std, opt_obs_err/obs_std, pft_obs_err/obs_std, 
        ef_obs_err/obs_std, pft_obs_err/obs_std-opt_obs_err/obs_std, ef_obs_err/obs_std-opt_obs_err/obs_std, pft_obs_err/obs_std-ef_obs_err/obs_std],
        ['opt_unc','opt_err','pft_err','ef_err','obs_std','norm_opt_err','norm_pft_err','norm_ef_err','norm_pft_minus_norm_opt_err','norm_ef_minus_norm_opt_err','norm_pft_minus_norm_ef_err'],
        [0.,0.,0.,0.,0.,0.,0.,0.,-1.,-1.,-1.], [0.7,0.7,0.7,0.7,0.,2.,2.,2.,1.,1.,1.]):
        
        data_nonan, pixel_lst_nonan = remove_nan(data, pixel_lst)
        
        stipple = card_unc if (plot_title=='ef_err') | (plot_title=='pft_err') else None
        autil.plot_map(nrows=46, ncols=73, land_pixel_list=[file[-8:-4] for file in cbf_files], 
            pixel_value_list=pixel_lst_nonan, value_list=data_nonan, vmin=vmin, vmax=vmax, cmap='bwr',
            savepath=plot_dir+'forward_compare/maps/'+model_id+'/', savename=model_id+'_MCMC'+mcmc_id+'_'+n_iter+'_'+pft_spec+ef_spec+plot_title, stipple=stipple) #vmax=np.nanpercentile(data_nonan, 90)
    
    
    # save errors for comparison analysis
    DataFrame(list(zip(pixel_lst, list(ef_obs_err/obs_std))), columns=['pixels','norm_mae']).to_pickle(misc_dir + 'mae_pkls/' + model_id + '_MCMC'+mcmc_id + '_'+n_iter + '_' + ef_spec + '.pkl')
    DataFrame(list(zip(pixel_lst, list(pft_obs_err/obs_std))), columns=['pixels','norm_mae']).to_pickle(misc_dir + 'mae_pkls/' + model_id + '_MCMC'+mcmc_id + '_'+n_iter + '_' + pft_spec + '.pkl')
    
    
    # plot discrete map showing best parameterization (lowest error) for each pixel
    '''best_param_nonan, pixel_lst_nonan = best_param_nonancol([opt_obs_err, pft_obs_err, ef_obs_err], pixel_lst)
    autil.plot_map(nrows=46, ncols=73, land_pixel_list=[file[-8:-4] for file in cbf_files], pixel_value_list=pixel_lst_nonan, value_list=best_param_nonan, cmap=LinearSegmentedColormap.from_list('mycmap', [(0, 'dodgerblue'), (0.5, 'orangered'), (1., 'limegreen')]),savepath=plot_dir+'forward_compare/maps/'+model_id+'/', savename=model_id+'_MCMC'+mcmc_id+'_'+n_iter+'_'+ef_spec+'best_param')'''
        
    best_param_nonan, pixel_lst_nonan = best_param_nonancol([pft_obs_err, ef_obs_err], pixel_lst)
    autil.plot_map(nrows=46, ncols=73, land_pixel_list=[file[-8:-4] for file in cbf_files], 
        pixel_value_list=pixel_lst_nonan, value_list=best_param_nonan, cmap=LinearSegmentedColormap.from_list(
        'mycmap', [(0, 'orangered'), (1., 'limegreen')]),
        savepath=plot_dir+'forward_compare/maps/'+model_id+'/', savename=model_id+'_MCMC'+mcmc_id+'_'+n_iter+'_'+pft_spec+ef_spec+'best_param')
        
    rgb_triplets = err_rgb_triplets([opt_obs_err, pft_obs_err, ef_obs_err], pixel_lst)
    autil.plot_map_rgb(nrows=46, ncols=73, land_pixel_list=[file[-8:-4] for file in cbf_files], 
        pixel_value_list=pixel_lst, value_list=rgb_triplets,
        savepath=plot_dir+'forward_compare/maps/'+model_id+'/', savename=model_id+'_MCMC'+mcmc_id+'_'+n_iter+'_'+pft_spec+ef_spec+'rgb')
        
    ############################################### create resubmission for pft ###############################################
    ###########################################################################################################################
        
    if write_txt_sh_pft_rerun:
        
        # set additional directories
        mdf_dir = '../code/CARDAMOM_2.1.6c/C/projects/CARDAMOM_MDF/' if nbe_optimization=='OFF' else '../code/CARDAMOM_Uma_2.1.6c-master/C/projects/CARDAMOM_MDF/'
        runmodel_dir = '../code/CARDAMOM_2.1.6c/C/projects/CARDAMOM_GENERAL/' if nbe_optimization=='OFF' else '../code/CARDAMOM_Uma_2.1.6c-master/C/projects/CARDAMOM_GENERAL/'
        cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
        cbf_pft_ic_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'_pft_ic/' + model_id + '/'
        cbr_pft_dir = '../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'_pft/' + model_id + '/'
        output_dir = '../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
        output_pft_dir = '../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'_pft/' + model_id + '/'
    
        if mcmc_id=='119':
            frac_save_out = str(int(int(n_iter)/500))
        elif mcmc_id=='3':
            frac_save_out = str(int(int(n_iter)/500*100)) # n_iterations/ frac_save_out * 100 will be ensemble size
            
        # set up which files to rerun
        pft_rerun = read_csv(misc_dir + pft_rerun_filename, header=None, sep=',', names=['pixel']+gl_lbls)
        txt_filename = 'pft_ic_combined_list_' + model_id + assim_type+'_MCMC'+mcmc_id + '_'+n_iter + '_rerun.txt'
        txt_file = open(txt_filename, 'w')
        
        cl_count, row_count = 1, 0
        for cbf_file in cbf_files:
            pixel = cbf_file[-8:-4]
            
            if int(pixel) in pft_rerun['pixel'].values:
                pixel_classes = pft_rerun.loc[pft_rerun['pixel']==int(pixel)].values[0][1:]
                
                for cl in pixel_classes:
                    if ~np.isnan(cl):
                        f = cbf_file[:-9]+'_PFT'+str(int(cl))+'_'+pixel
                        txt_file.write('%sCARDAMOM_MDF.exe %s%s %s%s %s 0 %s 0.001 %s 1000' % (mdf_dir, cbf_pft_ic_dir[3:], f+'.cbf', cbr_pft_dir, f+'.cbr', n_iter, frac_save_out, mcmc_id))
                        txt_file.write(' && %sCARDAMOM_RUN_MODEL.exe %s%s %s%s %s%s %s%s %s%s %s%s' % (runmodel_dir, cbf_pft_ic_dir[3:], f+'.cbf', cbr_pft_dir, f+'.cbr', 
                            output_pft_dir, 'fluxfile_'+ f +'.bin', output_pft_dir, 'poolfile_'+ f +'.bin', 
                            output_pft_dir, 'edcdfile_'+ f +'.bin', output_pft_dir, 'probfile_'+ f +'.bin'))
                        cl_count += 1
                        
                        if np.mod(cl_count, 5)==0:
                            txt_file.write('\n')
                            row_count += 1
                            
                        else:
                            txt_file.write(' && ')
                
        txt_file.close()
    
        sh_file = open(txt_filename[:-3] + 'sh', 'w')
        autil.fill_in_sh(sh_file, array_size=row_count, n_hours=10, txt_file=txt_filename, combined=True)
    
    return

if __name__=='__main__':
    main()
