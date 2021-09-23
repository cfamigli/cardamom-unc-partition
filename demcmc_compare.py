
# compare demcmc performance between multiple instances

import numpy as np
import glob
import sys
import os
import readwritebinary as rwb
import anova_utilities as autil
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pandas import read_pickle, DataFrame



def plot_output_ts(cbf_data, fwd_data_list, obs, obsunc, lbls=[], var='', savepath='', title=''):
    plt.figure(figsize=(8,4))
    n_steps = cbf_data['nodays']
    
    colors = ['orangered', 'dodgerblue', 'gold', 'mediumturquoise']#plt.cm.Spectral(np.linspace(0,1,len(fwd_data_list)))
    data_count = 0
    for data in fwd_data_list:
        plt.plot(np.nanmedian(data, axis=0), color=colors[data_count], linewidth=2.5, alpha=0.8, label=lbls[data_count])
        plt.fill_between(range(n_steps), np.nanpercentile(data, 95, axis=0), 
            np.nanpercentile(data, 5, axis=0), color=colors[data_count], alpha=0.3)
        data_count+=1
        
    obs[obs==-9999] = float('nan')
    plt.plot(obs, linewidth=1, label='Obs', color='k')
    if obsunc>1:
        plt.fill_between(range(n_steps), obs*obsunc, obs/obsunc, color='darkgray', alpha=0.6)
    plt.ylabel(var)
    plt.xlabel('Months')
    
    plt.subplots_adjust(hspace=.5,wspace=.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(savepath + title)
    plt.close()
    return


def rmse_real_numbers_only(pred, obs):
    # pred is a cardamom time series ensemble
    # obs is a time series
    pred_med = np.nanmedian(pred, axis=0)
    if len(pred_med)>len(obs):
        pred_med=pred_med[:-1]
    good_inds = (pred_med!=-9999) & (obs!=-9999) & (np.isfinite(pred_med)) & (np.isfinite(obs))
    rmse = mean_squared_error(pred_med[good_inds], obs[good_inds], squared=False)

    return rmse 
    
    
def mae_real_numbers_only(pred, obs):
    # pred is a cardamom time series ensemble
    # obs is a time series
    pred_med = np.nanmedian(pred, axis=0)
    if len(pred_med)>len(obs):
        pred_med=pred_med[:-1]
    good_inds = (pred_med!=-9999) & (obs!=-9999) & (np.isfinite(pred_med)) & (np.isfinite(obs))
    mae = np.nanmean(abs(pred_med[good_inds] - obs[good_inds]))
    mae_std = np.nanstd(abs(pred_med[good_inds] - obs[good_inds]))

    return mae, mae_std
    


def plot_dist_compare(fwd_data_list, obs, obsunc, lbls=[], var='', savepath='', title=''):
    
    colors = ['orangered', 'dodgerblue', 'gold', 'mediumturquoise']#plt.cm.Spectral(np.linspace(0,1,len(fwd_data_list)))
    
    obs[obs==-9999] = float('nan')
    
    
    
    if var=='LAI':
        obs_mean = np.nanmean(obs)
        fwd_means = [np.nanmean(fwd, axis=1) for fwd in fwd_data_list]
        
        plt.figure(figsize=(6,5))
        plt.axhline(obs_mean, c='k', linewidth=1.5, linestyle='--', label='Obs')
        plt.fill_between(range(6), obs_mean*obsunc, obs_mean/obsunc, color='dodgerblue', alpha=0.5, label='Obs unc')
        plt.boxplot(fwd_means, widths=0.7, whis=(5,95))
        meds, obs_compare, obs_bounds = [np.nanmedian(f) for f in fwd_means], obs_mean, [obs_mean*obsunc, obs_mean/obsunc]
        plt.ylabel(var)
        plt.ylim([-1,obs_mean*20])
        plt.xlim([0,5])
        plt.xticks([1, 2, 3, 4], [lbl for lbl in lbls])
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(savepath + title + '.png')
        plt.close()
        
        
    if var=='ABGB':
        plot_count = 0
        obs_count = 0
        for el in obs:
            if (np.isnan(el)) | (plot_count>12): continue
            else:
                plt.figure(figsize=(6,5))
                plt.axhline(el, c='k', linewidth=1.5, linestyle='--', label='Obs')
                plt.fill_between(range(6), el*obsunc, el/obsunc, color='dodgerblue', alpha=0.5, label='Obs unc')
                plt.boxplot([fwd[:,obs_count] for fwd in fwd_data_list], widths=0.7, whis=(5,95))
                meds, obs_compare, obs_bounds = [np.nanmedian(f[:,obs_count]) for f in fwd_data_list], el, [el*obsunc, el/obsunc]
                
                plt.ylabel(var)
                plt.ylim([-1,el*20])
                plt.xlim([0,5])
                plt.xticks([1, 2, 3, 4], [lbl for lbl in lbls])
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(savepath + title + str(plot_count) + '.png')
                plt.close()
                
                plot_count += 1
            obs_count += 1
            
    
    if var=='NBE':
        
        fwd_means = [np.nanmean(fwd, axis=1) for fwd in fwd_data_list]
        obs_mean = np.nanmean(obs)
        meds, obs_compare, obs_bounds = [np.nanmedian(f) for f in fwd_means], obs_mean, obsunc
        
        
        n_months = 12
        n_years = int(len(obs)/n_months)
        obs_an_mean = np.nanmean(obs.reshape(-1, n_months), axis=1) # number of elements equal to n_years
        obs_seas_mean = np.nanmean(obs.reshape(-1, n_months), axis=0) # number of elements equal to n_months
        
        fwd_an_means, fwd_seas_means = [], []
        for fwd in fwd_data_list:
            
            an_mean, seas_mean = [], []
            n_ens = fwd.shape[0]
            for ens in range(n_ens):
                an_mean.append(np.nanmean(fwd[ens,:].reshape(-1, n_months), axis=1))
                seas_mean.append(np.nanmean(fwd[ens,:].reshape(-1, n_months), axis=0))
            
            fwd_an_means.append(np.vstack(an_mean))
            fwd_seas_means.append(np.vstack(seas_mean))
            
        for pred_data, obs_data, unc, kind in zip([fwd_seas_means, fwd_an_means], [obs_seas_mean, obs_an_mean], obsunc, ['seasonal', 'annual']):#zip([fwd_an_means, fwd_seas_means], [obs_an_mean, obs_seas_mean], obsunc, ['annual', 'seasonal']):
            
            plot_count = 0
            obs_count = 0
            for step in range(len(obs_data)):
                
                if (np.isnan(obs_data[step])) | (plot_count>1): continue
                else:
                    plt.figure(figsize=(6,5))
                    plt.axhline(obs_data[step], c='k', linewidth=1.5, linestyle='--', label='Obs')
                    plt.fill_between(range(6), obs_data[step]+unc, obs_data[step]-unc, color='dodgerblue', alpha=0.5, label='Obs unc')
                    plt.boxplot([pred[:,step] for pred in pred_data], widths=0.7, whis=(5,95))
                    
                    plt.ylabel(var)
                    plt.ylim([-1*abs(obs_data[step]*20), abs(obs_data[step]*20)])
                    plt.xlim([0,5])
                    plt.xticks([1, 2, 3, 4], [lbl for lbl in lbls])
                    plt.legend(loc='best')
                    plt.tight_layout()
                    plt.savefig(savepath + title + kind+'_'+str(obs_count)+ '.png')
                    plt.close()
                    plot_count += 1
                    
                obs_count += 1

    return


def rank_mae(maes, lbls):
    
    ranks = np.zeros(len(lbls))
    for mae in maes:
        ind_best = mae.index(min(mae))
        ranks[ind_best] += 1
    
    return ranks


def plot_maes(maes, lbls, savepath, title):
    colors = ['orangered', 'dodgerblue', 'gold', 'mediumturquoise']
    
    plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(maes))-0.225, [mae[0] for mae in maes], width=0.15, color=colors[0], edgecolor='k', label='811+MCMC')
    plt.bar(np.arange(len(maes))-0.075, [mae[1] for mae in maes], width=0.15, color=colors[1], edgecolor='k', label='811+DEMCMC')
    plt.bar(np.arange(len(maes))+0.075, [mae[2] for mae in maes], width=0.15, color=colors[2], edgecolor='k', label='911+DEMCMC')
    plt.bar(np.arange(len(maes))+0.225, [mae[3] for mae in maes], width=0.15, color=colors[3], edgecolor='k', label='911+MCMC')
    
    plt.xticks(np.arange(len(maes)), [lbl for lbl in lbls])
    
    if 'ABGB' in title:
        plt.yscale('log')
        
    plt.ylim([0, None])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(savepath + title+'.png')
    plt.close()
    return
    

def main():
    
    # get specifications for run to read
    model_ids = ['811','811','911','911']
    assim_type = '_p25adapted'
    ens_size = 500
    
    # get pixels, ids and number of iterations to read
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_ids[0] + '/'
    pixels = ['3809','3524','2224','4170','1945','3813','4054','3264','1271','3457']
    mcmc_ids = ['119','3','3','119']
    n_iters = ['40000000','1000000','1000000','40000000']
    
    
    nbe_mae, lai_mae, abgb_mae, gpp_mae = [], [], [], []
    
    # run through pixels
    for pixel in pixels:
    
        # get that pixel's outputs for each MCMCID
        nbe_pred, lai_pred, abgb_pred, gpp_pred = [], [], [], []
        for model_id, mcmc_id, n_iter in zip(model_ids, mcmc_ids, n_iters):
            
            # set directories
            cur_dir = os.getcwd() + '/'
            cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
            cbr_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
            output_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
            plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
            parnames = autil.get_parnames('../../misc/', model_id)
            
            # read cbf file for that pixel
            cbf_pixel = rwb.read_cbf_file(glob.glob(cbf_dir + '*_' + pixel+'.cbf')[0])
            
            # read obs and obs unc for that pixel
            nbe_obs, lai_obs, abgb_obs, sif_obs = cbf_pixel['OBS']['NBE'], cbf_pixel['OBS']['LAI'], cbf_pixel['OBS']['ABGB'], cbf_pixel['OBS']['GPP']
            nbe_an_unc, nbe_seas_unc, lai_unc, abgb_unc = cbf_pixel['OBSUNC']['NBE']['annual_unc'], cbf_pixel['OBSUNC']['NBE']['seasonal_unc'], cbf_pixel['OTHER_OBS']['MLAI']['unc'], cbf_pixel['OBSUNC']['ABGB']['unc']
            
                
            conv_chains_pkl = read_pickle(glob.glob(cbr_dir + model_id + assim_type + '*_MCMC'+mcmc_id + '_'+n_iter+'_best_subset.pkl')[0])
            conv_chains_pkl.columns = ['pixel','bestchains','conv']# if model_id!='911' else ['pixel','bestchains'] #rename columns for easier access
            
            # grab cbrs corresponding to that pixel, MCMCID and number of iterations
            files = glob.glob(cbr_dir + '*MCMC'+mcmc_id+'_' + n_iter + '_'+ pixel+'*.cbr')
            files.sort()
            best_chains = conv_chains_pkl.loc[conv_chains_pkl['pixel']==pixel]['bestchains'].values[0][1:]
            
            # run through cbrs
            cbr_chain_list = []
            for chain in best_chains:
                print(chain)
                
                # read cbr for one file and transform Bday, Fday
                file = [i for i in files if pixel+'_'+chain+'.cbr' in i][0]
                cbr_chain = autil.modulus_Bday_Fday(rwb.read_cbr_file(file, {'nopars': len(parnames)}), parnames)
                print(cbr_chain.shape)
            
                
                # read forward run for that cbr
                
                flux_chain = rwb.readbinarymat(output_dir + 'fluxfile_' + file.partition(cbr_dir)[-1][:-3]+'bin', [cbf_pixel['nodays'], autil.get_nofluxes_nopools_lma(model_id)[0]])
                pool_chain = rwb.readbinarymat(output_dir + 'poolfile_' + file.partition(cbr_dir)[-1][:-3]+'bin', [cbf_pixel['nodays']+1, autil.get_nofluxes_nopools_lma(model_id)[1]])
                    
                # add chain to list for GR calculation
                if np.shape(cbr_chain)[0]==ens_size: 
                    
                    cbr_chain_list.append(cbr_chain)
                    
                    # add forward run chain to aggregated matrix
                    flux_pixel = np.copy(flux_chain) if best_chains.index(chain)==0 else np.concatenate((flux_pixel, flux_chain), axis=0)
                    pool_pixel = np.copy(pool_chain) if best_chains.index(chain)==0 else np.concatenate((pool_pixel, pool_chain), axis=0)
                
            # compute gelman rubin
            if len(cbr_chain_list)>1:
                gr = autil.gelman_rubin(cbr_chain_list)
                print('%i of %i parameters converged' % (sum(gr<1.2), len(parnames)))
            else:
                gr = np.nan
                
            cbr_pixel = np.vstack(cbr_chain_list)
            
            
            print(pool_pixel.shape)
            print(cbr_pixel.shape)
            # nbe, lai, and abgb predictions at pixel
            # list with elements corresponding to MCMCIDs considered (e.g. first element is MCMCID 119)
            nbe_pred.append(autil.get_output('NBE', model_id, flux_pixel, pool_pixel, cbr_pixel, autil.get_nofluxes_nopools_lma(model_id)[2]))
            lai_pred.append(autil.get_output('LAI', model_id, flux_pixel, pool_pixel, cbr_pixel, autil.get_nofluxes_nopools_lma(model_id)[2])[:,:-1])
            abgb_pred.append(autil.get_output('ABGB', model_id, flux_pixel, pool_pixel, cbr_pixel, autil.get_nofluxes_nopools_lma(model_id)[2])[:,:-1])
            gpp_pred.append(autil.get_output('GPP', model_id, flux_pixel, pool_pixel, cbr_pixel, autil.get_nofluxes_nopools_lma(model_id)[2]))
            
        # plot time series
        lbls = [model_id+'_MCMC'+mcmc_id for model_id, mcmc_id in zip(model_ids, mcmc_ids)]
        plot_output_ts(cbf_pixel, nbe_pred, nbe_obs, nbe_an_unc, lbls=lbls, var='NBE', savepath=cur_dir+plot_dir+'demcmc_mcmc/', title='all_models'+'_NBE_'+pixel+'.png')
        plot_output_ts(cbf_pixel, lai_pred, lai_obs, lai_unc, lbls=lbls, var='LAI', savepath=cur_dir+plot_dir+'demcmc_mcmc/', title='all_models'+'_LAI_'+pixel+'.png')
        plot_output_ts(cbf_pixel, gpp_pred, sif_obs, 0, lbls=lbls, var='GPP', savepath=cur_dir+plot_dir+'demcmc_mcmc/', title='all_models'+'_GPP_'+pixel+'.png')
        
        # plot box plots
        plot_dist_compare(nbe_pred, nbe_obs, [nbe_an_unc, nbe_seas_unc], lbls=lbls, var='NBE', savepath=cur_dir+plot_dir+'demcmc_mcmc/', title='all_models'+'_NBE_'+pixel+'_dist_')
        plot_dist_compare(lai_pred, lai_obs, lai_unc, lbls=lbls, var='LAI', savepath=cur_dir+plot_dir+'demcmc_mcmc/', title='all_models'+'_LAI_'+pixel+'_dist_')
        plot_dist_compare(abgb_pred, abgb_obs, abgb_unc, lbls=lbls, var='ABGB', savepath=cur_dir+plot_dir+'demcmc_mcmc/', title='all_models'+'_ABGB_'+pixel+'_dist_')

        # plot obs vs median comparison
        nbe_mae.append([mae_real_numbers_only(f, nbe_obs)[0] for f in nbe_pred])
        lai_mae.append([mae_real_numbers_only(f, lai_obs)[0] for f in lai_pred])
        abgb_mae.append([mae_real_numbers_only(f, abgb_obs)[0] for f in abgb_pred])
        
        print(rank_mae(nbe_mae, lbls))
        print(rank_mae(lai_mae, lbls))
        print(rank_mae(abgb_mae, lbls))
    
    plot_maes(nbe_mae, pixels, savepath=cur_dir+plot_dir+'demcmc_mcmc/', title='all_models_NBE_mae')
    plot_maes(lai_mae, pixels, savepath=cur_dir+plot_dir+'demcmc_mcmc/', title='all_models_LAI_mae')
    plot_maes(abgb_mae, pixels, savepath=cur_dir+plot_dir+'demcmc_mcmc/', title='all_models_ABGB_mae')
    
    return

if __name__=='__main__':
    main()