
import numpy as np
import readwritebinary as rwb
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
from pandas import DataFrame, read_csv

def subset_df_by_substring(df, subset_str):
    # subset a dataframe by a specified substring
    subset = [row for row in df.index if subset_str in row]
    return subset
    
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
       '1032': (33,8,16)
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
        obs = cbf_data['OBS'][flux]
        obs[obs==-9999] = float('nan')
        axs[fluxes.index(flux)].plot(obs, linewidth=2, label='Obs',color='k')
        axs[fluxes.index(flux)].set_ylabel(flux)
        axs[fluxes.index(flux)].set_xlabel('Months')
    
    plt.subplots_adjust(hspace=.5,wspace=.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(savepath + title)
    plt.close()
    return

def plot_map(nrows, ncols, land_pixel_list, pixel_value_list, value_list, vmin=None, vmax=None, cbar_label='', savepath='', title=''):
    # set up array for mapping
    # specify number of rows and columns

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
    plt.imshow(np.flipud(value_arr), cmap='brg_r', vmin=vmin, vmax=vmax, zorder=1)
    plt.axis('off')
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath + title + '.png')
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

def plot_spread_v_iter(spread_mat, pixel_list, var_ind, var_name, iter_list, savepath, savename):
    # plot ensemble spread versus number of iterations
    
    n_pix = len(pixel_list)
    n_col = 5
    n_row = int(np.ceil(n_pix/n_col))
    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col*2.5, n_row*3))
    
    for pixel in pixel_list:
        p_ind = pixel_list.index(pixel)
        
        ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].plot(spread_mat[p_ind, var_ind, :], c='dodgerblue', linewidth=1.5, marker='o', markersize=5)
        ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_xticks(np.arange(len(iter_list)))
        ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_xticklabels(iter_list, rotation=90)
        ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_xlabel('Number of iterations')
        ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_ylabel('Average ensemble \nspread (' + var_name + ')')
        ax[round((p_ind+1)/n_pix), np.mod(p_ind, n_col)].set_title('Pixel: ' + pixel)
        
    plt.tight_layout()
    plt.savefig(savepath + '/' + savename + '.pdf')
    plt.close()
    return
    