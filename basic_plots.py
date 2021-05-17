
import numpy as np
import readwritebinary as rwb
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
from pandas import DataFrame

def subset_df_by_substring(df, subset_str):
    subset = [row for row in df.index if subset_str in row]
    return subset

def plot_par_histograms(cbr_data, parnames=[], savepath='', title=''):
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


