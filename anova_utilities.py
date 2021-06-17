
import numpy as np
import readwritebinary as rwb
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
from pandas import DataFrame, read_csv

def fill_in_sh(sh_file, array_size, n_hours, txt_file, combined=False):
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH --nodes=1\n')
    sh_file.write('#SBATCH -p konings,normal,owners\n')
    sh_file.write('#SBATCH -t %i:00:00\n' % n_hours)
    sh_file.write('#SBATCH --mail-type=END,FAIL\n')
    sh_file.write('#SBATCH --mail-user=cfamigli@stanford.edu\n')
    sh_file.write('#SBATCH --array=0-%i\n\n' % array_size)
    
    sh_file.write('# define the location of the command list file\n')
    sh_file.write('CMD_LIST=./%s\n\n' % txt_file)
    
    sh_file.write('# get the command list index from Slurm\n')
    sh_file.write('CMD_INDEX=$SLURM_ARRAY_TASK_ID\n\n')
    
    sh_file.write('# execute the command\n')
    sh_file.write('$(sed "${CMD_INDEX}q;d" "$CMD_LIST")\n') if not combined else sh_file.write('eval $(sed "${CMD_INDEX}q;d" "$CMD_LIST")\n')
    sh_file.close()
    return

def subset_df_by_substring(df, subset_str):
    # subset a dataframe by a specified substring
    subset = [row for row in df.index if subset_str in row]
    return subset
    
def latlon_to_rowcol(lat,lon):
    
    # Vectors for code in file name
    latspace = np.linspace(-90,90,46)
    lonspace = np.linspace(-180,180,73)
    
    latidx = np.argwhere(latspace == lat)
    lonidx = np.argwhere(lonspace == lon)

    latnum = latidx[0][0] + 1
    lonnum = lonidx[0][0] + 1
    
    if latnum < 10:
        latstr = '0'+ str(latnum)
    else:
        latstr = str(latnum)
        
    if lonnum < 10:
        lonstr = '0' + str(lonnum)
    else:
        lonstr = str(lonnum)
    
    return latstr+lonstr

def rowcol_to_latlon(checkptlist):
    """
    Convert GEOCHEM rows and columns to lats and longs
    """
    
    # Vectors for code in file name
    latspace = np.linspace(-90,90,46)
    lonspace = np.linspace(-180,180,73)
    
    latpts = []
    lonpts = []
    for ckpt in checkptlist:
        
        latpts.append(latspace[int(ckpt[0:2])-1])
        lonpts.append(lonspace[int(ckpt[2:])-1])
    
    latlonzip = list(zip(latpts,lonpts))
    return latlonzip
    
def get_parnames(par_dir, model_id):
    # input model id and get list of parameters from csv
    pars = read_csv(par_dir+'parameters.csv', skiprows=1)
    if model_id=='102':
        model_id = '101'
    if model_id in pars.columns:
        parnames = pars[model_id].dropna()
    else:
        parnames = []
    return parnames
    
def get_nofluxes_nopools_lma(model_id):
    # input a model id and get number of fluxes [0], number of pools [1], and index of lma [2]
    return {
        '809': (30,7,16),
        '811': (30,7,16),
        '831': (30,7,16),
        '101': (28,3,10),
        '102': (28,3,10),
       '1003': (33,8,16),
       '400': (28,6,16),
       '1010': (43,8,16),
       '1000': (32,8,16),
       '900': (29,6,16),
       '901': (29,6,16),
       '1032': (33,8,16),
        '911': (30,7,16)
    }[model_id]
    
def find_all_chains(file_list, pixel):
    # takes a list of file names and a pixel number (string)
    # outputs list of files corresponding to chains run at that pixel
    
    return [file for file in file_list if file[-10:-6]==pixel]
    
def gelman_rubin(x, return_var=False):
    # compute convergence on multiple chains
	if np.shape(x) < (2,):
		raise ValueError(
			'Gelman-Rubin diagnostic requires multiple chains of the same length.')

	try:
		m, n = np.shape(x)
	except ValueError:
		#print(np.shape(x))
		return np.array([gelman_rubin(np.transpose(y)) for y in np.transpose(x)])

	# Calculate between-chain variance
	B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)
	# Calculate within-chain variances
	W = np.sum([(x[i] - xbar) ** 2 for i, xbar in enumerate(np.mean(x, 1))]) / (m * (n - 1))
	# (over) estimate of variance
	s2 = W * (n - 1) / n + B_over_n
	if return_var:
		return s2
	# Pooled posterior variance estimate
	V = s2 + B_over_n / m
	# Calculate PSRF
	R = V / W
	return np.sqrt(R)
	
def get_output(var, model, flux_data, pool_data, cbr_data, lma_ind):
    # return relevant output depending on flux or pool input as 'var'
    return {
        'NBE': np.sum(flux_data[:,:,[2,12,13]], axis=2) - flux_data[:,:,0] + flux_data[:,:,16],
        'cumNBE': np.cumsum(np.sum(flux_data[:,:,[2,12,13]], axis=2) - flux_data[:,:,0] + flux_data[:,:,16], axis=1),
        'LAI': pool_data[:,:,1]/np.expand_dims(cbr_data[:,lma_ind],1),
        'GPP': flux_data[:,:,0],
        'Reco': np.sum(flux_data[:,:,[2,12,13]], axis=2),
        'Rauto': flux_data[:,:,2],
        'Rhet': np.sum(flux_data[:,:,[12,13]], axis=2),
        'lit': pool_data[:,:,4] if int(model)>=400 else [],
        'root': pool_data[:,:,2] if int(model)>=400 else [],
        'som': pool_data[:,:,5] if int(model)>=400 else [],
        'wood': pool_data[:,:,3] if int(model)>=400 else []
    }[var]
    
def remove_outliers(fwd_data):
    # remove outliers (>1e6*median) from forward data
    meds = np.nanmedian(fwd_data, axis=0)
    for i in range(len(meds)):
        fwd_data[:,i][(fwd_data[:,i]>abs(meds[i]*1e6)) | (fwd_data[:,i]<-1*abs(meds[i]*1e6))] = float('nan')
    return fwd_data
    
def remove_below_25_above_75(fwd_data):
    # set values above 75th percentile and below 25th percentile to nan in forward data
    ub = np.nanpercentile(fwd_data, 75, axis=0)
    lb = np.nanpercentile(fwd_data, 25, axis=0)
    for i in range(len(ub)):
        fwd_data[:,i][(fwd_data[:,i]>ub[i]) | (fwd_data[:,i]<lb[i])] = float('nan')
    return fwd_data

def plot_par_histograms(cbr_data, parnames=[], savepath='', title=''):
    # plot posterior parameter distributions
    npars = cbr_data.shape[1]
    ncols = 6
    nrows = 0
    while nrows<npars:
        nrows+=ncols
    nrows = int(nrows/6)
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_size_inches(12,12)
    count = 0
    for row in range(nrows):
        for col in range(ncols):
            if count<npars:
                axs[row, col].hist(cbr_data[:, count], bins=50, color='dodgerblue')
                if len(parnames)>1:
                    axs[row, col].set_title(parnames[count][:20])
                count += 1
    plt.subplots_adjust(hspace = .5,wspace=.5)
    plt.tight_layout()
    plt.savefig(savepath + title)
    plt.close()
    return

def plot_flux_pool_timeseries(cbf_data, cbr_data, flux_data, pool_data, lma_ind, savepath='', title=''):
    # plot GPP, LAI, NBE from forward runs (with spread)
    nrows = 3
    ncols = 1
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_size_inches(9,9)
    
    fluxes = ['GPP','LAI','NBE']
    
    for flux in fluxes:
        if flux=='GPP':
            n_steps = cbf_data['nodays']
            pred = flux_data[:,:,0]
        elif flux=='LAI':
            n_steps = cbf_data['nodays'] + 1
            pred = pool_data[:,:,1]/np.expand_dims(cbr_data[:,lma_ind],1)
        elif flux=='NBE':
            n_steps = cbf_data['nodays']
            pred = np.sum(flux_data[:,:,[2,12,13]], axis=2) - flux_data[:,:,0] + flux_data[:,:,16]
          
        axs[fluxes.index(flux)].plot(np.nanmedian(pred, axis=0), color='dodgerblue', linewidth=3)
        axs[fluxes.index(flux)].fill_between(range(n_steps), np.nanpercentile(pred, 25, axis=0), 
            np.nanpercentile(pred, 75, axis=0), color='dodgerblue', alpha=0.3)
        try:
            obs = cbf_data['OBS'][flux]
            obs[obs==-9999] = float('nan')
            axs[fluxes.index(flux)].plot(obs, linewidth=2, label='Obs',color='k')
        except:
            pass
        axs[fluxes.index(flux)].set_ylabel(flux)
        axs[fluxes.index(flux)].set_xlabel('Months')
    
    plt.subplots_adjust(hspace=.5,wspace=.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(savepath + title)
    plt.close()
    return

def plot_map(nrows, ncols, land_pixel_list, pixel_value_list, value_list, vmin=None, vmax=None, cmap='spring', cbar_label='', savepath='', savename=''):
    # set up array for mapping
    # specify number of rows and columns
    
    # land_pixel_list is a list of all pixel numbers that cbfs exist for (=all land pixels)
    # pixel_value_list is a list of the pixels we want to plot (equivalent to land_pixel_list when we are plotting entire globe)
    # value_list is a list of the values we want to plot, ordered according to pixel_value_list

    latspace = np.linspace(-90,90,nrows)
    lonspace = np.linspace(-180,180,ncols)

    lat = np.tile(latspace.reshape(-1,1), (1,len(lonspace)))
    lon = np.tile(lonspace, (len(latspace),1))
    value_arr = np.zeros((nrows,ncols))*np.nan
    land_arr = np.zeros((nrows,ncols))*np.nan
    
    for pixel in pixel_value_list:
        (latpx,lonpx) = rwb.rowcol_to_latlon(pixel)
        row = np.where((lat==latpx) & (lon==lonpx))[0]
        col = np.where((lat==latpx) & (lon==lonpx))[1]
        value_arr[row,col] = value_list[pixel_value_list.index(pixel)]
        
    for pixel in land_pixel_list:
        (latpx,lonpx) = rwb.rowcol_to_latlon(pixel)
        row = np.where((lat==latpx) & (lon==lonpx))[0]
        col = np.where((lat==latpx) & (lon==lonpx))[1]
        land_arr[row,col] = -9999

    plt.figure(figsize=(9,6))
    land_cmap = colors.ListedColormap(['gainsboro'])
    plt.imshow(np.flipud(land_arr), cmap=land_cmap, zorder=0)
    plt.imshow(np.flipud(value_arr), cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
    plt.axis('off')
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(savepath + savename + '.png')
    plt.close()
    return
    
def plot_map_discrete_cmap(nrows, ncols, land_pixel_list, pixel_value_list, value_list, vmin=None, vmax=None, ncolors=48, cmap='spring', cbar_label='', savepath='', savename=''):
    # set up array for mapping -- DISCRETE COLORBAR
    # specify number of rows and columns
    
    # land_pixel_list is a list of all pixel numbers that cbfs exist for (=all land pixels)
    # pixel_value_list is a list of the pixels we want to plot (equivalent to land_pixel_list when we are plotting entire globe)
    # value_list is a list of the values we want to plot, ordered according to pixel_value_list

    latspace = np.linspace(-90,90,nrows)
    lonspace = np.linspace(-180,180,ncols)

    lat = np.tile(latspace.reshape(-1,1), (1,len(lonspace)))
    lon = np.tile(lonspace, (len(latspace),1))
    value_arr = np.zeros((nrows,ncols))*np.nan
    land_arr = np.zeros((nrows,ncols))*np.nan
    
    for pixel in pixel_value_list:
        (latpx,lonpx) = rwb.rowcol_to_latlon(pixel)
        row = np.where((lat==latpx) & (lon==lonpx))[0]
        col = np.where((lat==latpx) & (lon==lonpx))[1]
        value_arr[row,col] = value_list[pixel_value_list.index(pixel)]
        
    for pixel in land_pixel_list:
        (latpx,lonpx) = rwb.rowcol_to_latlon(pixel)
        row = np.where((lat==latpx) & (lon==lonpx))[0]
        col = np.where((lat==latpx) & (lon==lonpx))[1]
        land_arr[row,col] = -9999

    plt.figure(figsize=(9,6))
    land_cmap = colors.ListedColormap(['gainsboro'])
    plt.imshow(np.flipud(land_arr), cmap=land_cmap, zorder=0)
    
    segmented_cmap = plt.get_cmap(cmap, ncolors)
    plt.imshow(np.flipud(value_arr), cmap=segmented_cmap, vmin=vmin, vmax=vmax, zorder=1)
    plt.axis('off')
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(savepath + savename + '.png')
    plt.close()
    return

def plot_anova_ts(data, ub, lb, Ms_div_sum, Mp_div_sum, models, var, pixel, assim_type, savepath):
    # plot variance partitioning for a given variable/model/pixel
    
    fig, ax = plt.subplots(2,1, figsize=(8,6))
    colors = plt.cm.Spectral(np.linspace(0,1,len(models)))
    for row in range(len(models)):
        if np.sum(np.isfinite(data[row,:]))>0:
            ax[0].plot(data[row,:], linewidth=1.5, c=colors[row], label=models[row])
            ax[0].fill_between(range(data.shape[1]), lb[row,:], ub[row,:], facecolor=colors[row],
                edgecolor=None, alpha=0.2)
    ax[0].set_title('Variable: ' + var + '\nPixel: ' + pixel)
    ax[0].set_xlim([0,data.shape[1]-1])
    ax[0].set_xlabel('Months')
    ax[0].set_ylabel(var)
    ax[0].legend(loc='center right')

    ax[1].fill_between(range(data.shape[1]), 0, Mp_div_sum,
        facecolor='cornflowerblue', edgecolor=None, alpha=0.8, label='Parameters')
    ax[1].fill_between(range(data.shape[1]), Mp_div_sum, 1,
        facecolor='sandybrown', edgecolor=None, alpha=0.8, label='Structure')
        
    ax[1].set_ylim([0,1])
    ax[1].set_xlim([0,data.shape[1]-1])
    ax[1].set_xlabel('Months')
    ax[1].set_ylabel('Fraction of total variance')
    ax[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(savepath + '/' + pixel + '_' + var + '_' + assim_type + '.pdf')
    plt.close()
    return

def plot_partitioning_grouped(df, group, savepath, savename):
    # plot anova summaries by group
    
    fig, ax = plt.subplots(1, len(group), figsize=(len(group)*2.5, 3))
    for el in group:
        df_subset = df.loc[subset_df_by_substring(df, el)].sort_index(axis=1)
        df_subset.drop(columns='n').mean().plot(kind='bar', yerr=df_subset.drop(columns='n').std(),
            ax=ax[group.index(el)], color=['cornflowerblue', 'sandybrown'], edgecolor='k', linewidth=1.5)
        title = el
        ax[group.index(el)].set_ylim([0,None])
        ax[group.index(el)].set_title(title)
    ax[0].set_ylabel('Fraction of total variance')
    plt.tight_layout()
    plt.savefig(savepath + '/' + savename + '.pdf')
    plt.close()
    return

def plot_spread_v_iter(spread_mat, pixel_list, var_ind, var_name, iter_list, metric, savepath, savename, single_val=False):
    # plot ensemble spread versus number of iterations
    
    n_pix = len(pixel_list)
    n_col = 5
    n_row = int(np.ceil(n_pix/n_col))
    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col*2.5, n_row*3))
    
    for pixel in pixel_list:
        p_ind = pixel_list.index(pixel)
        
        count = 0
        col = ['lightskyblue', 'lightcoral', 'gold']
        mcmc = ['DEMCMC', 'MCMC']
        for el in spread_mat:
            if not single_val:
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].plot(iter_list[count], 
                    el[p_ind, var_ind, :], c=col[count], linewidth=1.5, marker='o', markersize=5, label=mcmc[count]) 
            else:
                p_ind = count
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].bar(0, el[var_ind, 0], color=col[0], edgecolor='k', linewidth=1.5)
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].bar(1, el[var_ind, 1], color=col[1], edgecolor='k', linewidth=1.5)
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].bar(2, el[var_ind, 2], color=col[2], edgecolor='k', linewidth=1.5)
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_xticks(np.arange(3))
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_xticklabels(['811+MCMC','811+DEMCMC','911+MCMC'], rotation=60)
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_ylabel('Average ensemble \nspread (' + var_name + ')') if metric=='spread' else  ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_ylabel('RMSE (' + var_name + ')')
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_title('Pixel: ' + pixel)
            count += 1
            
        iter_list_combined = np.unique(np.concatenate(([int(i) for i in iter_list[0]],[int(i) for i in iter_list[1]])))
        if not single_val:
            ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_xticks(np.arange(len(iter_list_combined)))
            ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_xticklabels(iter_list_combined.tolist(), rotation=90)
            ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_xlabel('Number of iterations')
            ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_ylabel('Average ensemble \nspread (' + var_name + ')') if metric=='spread' else  ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_ylabel('RMSE (' + var_name + ')')
            ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_title('Pixel: ' + pixel)
            
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(savepath + '/' + savename + '.pdf')
    plt.close()
    return

def plot_conv_v_iter(conv_mat, pixel_list, iter_list, savepath, savename, single_val=False):
    # plot convergence versus number of iterations
    
    n_pix = len(pixel_list)
    n_col = 5
    n_row = int(np.ceil(n_pix/n_col))
    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col*2.5, n_row*3))
    
    for pixel in pixel_list:
        p_ind = pixel_list.index(pixel)
        
        count = 0
        col = ['lightskyblue', 'lightcoral', 'gold']
        mcmc = ['DEMCMC', 'MCMC']
        for el in conv_mat:
            if not single_val:
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].plot(iter_list[count], 
                    el[p_ind, :], c=col[count], linewidth=1.5, marker='o', markersize=5, label=mcmc[count])
            else:
                p_ind = count
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].bar(0, el[0], color=col[0], edgecolor='k', linewidth=1.5)
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].bar(1, el[1], color=col[1], edgecolor='k', linewidth=1.5)
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].bar(2, el[2], color=col[2], edgecolor='k', linewidth=1.5)
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_xticks(np.arange(3))
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_xticklabels(['811+MCMC','811+DEMCMC','911+MCMC'], rotation=60)
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_ylabel('Percent of parameters \nwith GR<1.2')
                ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_title('Pixel: ' + pixel)
            count += 1
            
        iter_list_combined = np.unique(np.concatenate(([int(i) for i in iter_list[0]],[int(i) for i in iter_list[1]])))
        if not single_val: 
            ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_xticks(np.arange(len(iter_list_combined)))
            ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_xticklabels(iter_list_combined.tolist(), rotation=90)
            ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_xlabel('Number of iterations')
            ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_ylabel('Percent of parameters \nwith GR<1.2')
            ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_title('Pixel: ' + pixel)
            
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(savepath + '/' + savename + '.pdf')
    plt.close()
    return

def plot_scatter_anova_compare(Mp_list, Ms_list, lbl_list, savepath, savename):
    # compare Ms and Mp between MCMC approaches
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].scatter(Mp_list[0], Mp_list[1], facecolor='cornflowerblue', edgecolor='k', s=100, linewidth=1.5)
    ax[0].plot((0, 1), 'k-', linewidth=1, zorder=0)
    ax[0].set_xlim([0,1])
    ax[0].set_ylim([0,1])
    ax[0].set_xlabel('Mp ('+lbl_list[0]+')')
    ax[0].set_ylabel('Mp ('+lbl_list[1]+')')
    
    ax[1].scatter(Ms_list[0], Ms_list[1], facecolor='sandybrown', edgecolor='k', s=100, linewidth=1.5)
    ax[1].plot((0, 1), 'k-', linewidth=1, zorder=0)
    ax[1].set_xlim([0,1])
    ax[1].set_ylim([0,1])
    ax[1].set_xlabel('Ms ('+lbl_list[0]+')')
    ax[1].set_ylabel('Ms ('+lbl_list[1]+')')
    
    plt.tight_layout()
    plt.savefig(savepath + '/' + savename + '.pdf')
    plt.close()
    return

def plot_nmodel_test(n, yMs, yMp, savepath, savename):
    # plot fractions of variance for each subset of models
    plt.figure(figsize=(4.5,4.5))
    plt.plot(n, yMs, color='sandybrown', linewidth=1.5, marker='o', markersize=10, label='Ms')
    plt.plot(n, yMp, color='cornflowerblue', linewidth=1.5, marker='o', markersize=10, label='Mp')
    plt.title(savename)
    plt.legend(loc='upper right')
    plt.xticks(n)
    plt.ylim([0,1])
    plt.xlabel('Number of models')
    plt.ylabel('Fraction of total variance')
    plt.tight_layout()
    plt.savefig(savepath + '/' + savename + '.pdf')
    plt.close()
    return
    
    