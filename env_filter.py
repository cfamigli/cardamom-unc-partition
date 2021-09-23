
"""

created on Tues Jul 20 08:51:02 2021

@author: cfamigli

set up environmental filtering models to predict CARDAMOM parameters

"""
 
import numpy as np
import readwritebinary as rwb
import os
import glob
import csv
import sys
import warnings
warnings.filterwarnings('ignore')
import anova_utilities as autil
from itertools import compress
from random import sample
from pandas import read_pickle, read_csv
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt

def fill_X_met_12mo(X, met):
    # retrieve seasonal cycle of each met variable and assign to X
    for metvar in range(met.shape[1]):
        data = met[:, metvar].reshape(-1,12)
        X[(metvar*12):((metvar+1)*12)] = np.nanmean(data, axis=0)
    return X


def reshape_cbr(cbr_data, target_ens_size):
    return np.vstack((cbr_data, np.ones((target_ens_size-cbr_data.shape[0], cbr_data.shape[1]))*np.nan))

def drop_nan(X, y, pixels):
    cond_y = ~np.any(np.isnan(y), axis=0)
    cond_X = ~np.any(np.isnan(X), axis=1)
    
    y_not_nan = y[:, (cond_y & cond_X)]
    X_not_nan = X[(cond_y & cond_X), :]
    pixels_not_nan = list(compress(pixels, (cond_y & cond_X)))
    return X_not_nan, y_not_nan, pixels_not_nan
    
    
def rescaled(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)
    
def w_interactions(X):
    poly = PolynomialFeatures(interaction_only=True,include_bias=False)
    return poly.fit_transform(X)
    
def w_squares(X):
    return np.hstack((X, X**2))
    
def w_all_polys(X):
    poly = PolynomialFeatures(include_bias=False)
    return poly.fit_transform(X)
    
    
def select_features(X_train, y_train, X_test, n_features_select=10):
	# configure to select all features
	fs = SelectKBest(score_func=f_regression, k=n_features_select)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
    
def bic(X, y):
    regr = OLS(y, add_constant(X)).fit()
    return regr.bic
    
def impose_bounds(preds, par_ind):
    # clip regression predictions to CARDAMOM prior range
    lb, ub = autil.get_parminmax_911(par_ind)[0], autil.get_parminmax_911(par_ind)[1]
    preds_clipped = np.copy(preds)
    if (par_ind!=11) & (par_ind!=14): # don't fix bday here
        preds_clipped[preds_clipped<lb] = lb
        preds_clipped[preds_clipped>ub] = ub
    return preds_clipped


def train_and_predict_SLR(X_train, y_train, X_test):
    # fit regression model on train
    regr = LinearRegression().fit(X_train, y_train)
    bic_val = bic(X_train, y_train)
    
    # make predictions on test set
    test_preds = regr.predict(X_test) # predictions of one parameter at n pixels
    train_preds = regr.predict(X_train)#
    return test_preds, train_preds, bic_val
    
def train_and_predict_PLS(X_train, y_train, X_test, n_components=None):
    # fit regression model on train
    regr = PLSRegression(n_components=n_components).fit(X_train, y_train)
    bic_val = bic(X_train, y_train)
    
    # make predictions on test set
    test_preds = regr.predict(X_test) # predictions of one parameter at n pixels
    train_preds = regr.predict(X_train)#
    return test_preds, train_preds, bic_val
    
def append_substring_to_string(string, substring):
    if substring not in string:
        string += substring
    return string


def run_regressions(X, y, pixels, rescale, include_interactions, include_squares, include_all_polys, do_feature_selection, do_PLS, write_to_csv, writer_bic,
    n_features_select, suffix, ens_size, n_regr_models, n_features):
        
    Xr, yr, pixels_r = drop_nan(X, y, pixels) 
    if rescale: 
        Xr = rescaled(Xr)
        suffix = append_substring_to_string(suffix, 'rescaled_')
    if include_interactions: 
        Xr = w_interactions(Xr)
        suffix = append_substring_to_string(suffix, 'wint_')
    if include_squares: 
        Xr = w_squares(Xr)
        suffix = append_substring_to_string(suffix, 'wsq_')
    if include_all_polys: 
        Xr = w_all_polys(Xr)
        suffix = append_substring_to_string(suffix, 'wpolys_')
    print(Xr.shape, yr.shape, len(pixels_r))
    
    # use repeated k-fold to cross validate results
    ns, nr = 10, 1#int(ens_size)
    rkf = RepeatedKFold(n_splits=ns, n_repeats=nr)
    bic_vals = np.zeros((ns*nr, n_regr_models))

    reg_test_preds_list, reg_train_preds_list = [np.zeros((ns*nr, n_regr_models))*np.nan for i in range(len(pixels_r))], [np.zeros((ns*nr, n_regr_models))*np.nan for i in range(len(pixels_r))]  # list of arrays, each array contains each fold (row)'s prediction for each parameter (column)
    card_test_preds_list, card_train_preds_list = [np.zeros(n_regr_models)*np.nan for i in range(len(pixels_r))], [np.zeros(n_regr_models)*np.nan for i in range(len(pixels_r))]
    fold_count = 0
    for train_index, test_index in rkf.split(Xr):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = Xr[train_index, :], Xr[test_index, :]
        
        for regr_model in range(n_regr_models):
            #print('running regression for ' + parnames[regr_model] + ' . . . ')
            y_train, y_test = yr[regr_model, train_index], yr[regr_model, test_index]
            
            # feature selection
            X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, n_features_select=n_features_select)
            
            # fit model and predict
            if not do_PLS:
                if not do_feature_selection:
                    test_preds, train_preds, bic_vals[fold_count, regr_model] = train_and_predict_SLR(X_train, y_train, X_test)
                else:
                    test_preds, train_preds, bic_vals[fold_count, regr_model] = train_and_predict_SLR(X_train_fs, y_train, X_test_fs)
                    
            else:
                test_preds, train_preds, bic_vals[fold_count, regr_model] = train_and_predict_PLS(X_train, y_train, X_test, n_components=n_features_select)
                    
            test_preds = impose_bounds(test_preds, regr_model)
            train_preds = impose_bounds(train_preds, regr_model)
            
            for test_el in test_index:
                index_of_pixel = test_el
                index_of_pred_in_test_set = np.where(test_index==test_el)[0][0]
                
                # append parameter predictions from regression and from CARDAMOM to lists
                reg_test_preds_list[index_of_pixel][fold_count, regr_model] = test_preds[index_of_pred_in_test_set]
                card_test_preds_list[index_of_pixel][regr_model] = y_test[index_of_pred_in_test_set]
            
            for train_el in train_index:#test_el in test_index:
                index_of_pixel = train_el
                index_of_pred_in_train_set = np.where(train_index==train_el)[0][0]
                
                # append parameter predictions from regression and from CARDAMOM to lists
                reg_train_preds_list[index_of_pixel][fold_count, regr_model] = train_preds[index_of_pred_in_train_set]
                card_train_preds_list[index_of_pixel][regr_model] = y_train[index_of_pred_in_train_set]
                
        fold_count += 1
        if np.mod(fold_count, 100)==0: print(fold_count)
       
        
    # make summary scatter plots
    print(X_train_fs.shape)
    if (do_feature_selection) | (do_PLS): 
        suffix = append_substring_to_string(suffix, 'fs'+str(n_features_select)+'_')
        k = X_train_fs.shape[1]
    else:
        k = n_features
        
    #if write_to_csv: writer_bic.writerow(np.concatenate(([int(k)], np.nanmean(bic_vals, axis=0))))
        
    return reg_test_preds_list, card_test_preds_list, reg_train_preds_list, card_train_preds_list, pixels_r, suffix, k

    

def plot_scatter_test_pred(y_card, y_reg, n_features, pixels_r, parnames, writer_fs, writer_preds, savepath, savename, train_full_ens, write_to_csv):
    npars = len(parnames)
    ncols = 6
    nrows = 0
    while nrows<npars:
        nrows+=ncols
    nrows = int(nrows/6)
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_size_inches(13,12)
    
    if write_to_csv: writer_preds.writerow(pixels_r)
    
    
    mse = np.zeros(len(parnames))
    count = 0
    for row in range(nrows):
        for col in range(ncols):
            if count<npars:
                
                if not train_full_ens:
                    x = [np.nanmedian(i[:,count]) for i in y_reg]
                    y = [j[count] for j in y_card] 
                    axs[row, col].scatter(x, y, facecolor='dodgerblue', edgecolor='k', linewidth=0.5, s=25)
                    
                else:
                    x = [i[:,count][np.isfinite(i[:,count])] for i in y_reg]
                    y = [j[:,count] for j in y_card]
                    axs[row, col].scatter(x, y, facecolor='dodgerblue', edgecolor='k', linewidth=0.1, s=4, alpha=0.5)
                
                if write_to_csv: writer_preds.writerow(x)
                
                #print(parnames[count])
                mse[count] = mean_squared_error(np.array(y), np.array(x), squared=False)
                print(mse[count])
                
                mx = max([np.nanmax(x), np.nanmax(y)])
                mn = min([np.nanmin(x), np.nanmin(y)])
                axs[row, col].text(0.45,0.05,'R$^2$='+str(round(r2_score(y, x), 2)), transform=axs[row, col].transAxes, weight='bold')
                axs[row, col].set_xlim([mn,mx])
                axs[row, col].set_ylim([mn,mx])
                axs[row, col].plot([mn,mx], [mn,mx], c='k', linewidth=1)
                if npars>1:
                    axs[row, col].set_title(parnames[count][:20])
                count += 1
                
    if write_to_csv: writer_fs.writerow(np.concatenate(([int(n_features)], mse)))
    
    plt.subplots_adjust(hspace = .5,wspace=.5)
    plt.tight_layout()
    plt.savefig(savepath + savename + '.png')
    plt.close()
    return



def plot_train_test(x, train_rmse, test_rmse, parnames, savepath, savename, norm=False):
    # for comparing with LC regression
    gl_test_rmse = [0.00026479355611842603,0.03615649502425793,0.03167849601519861,0.16479989801492995,0.8981636335269075,
        0.00014033694807500204,0.0011057753630186813,0.000925781267268108,2.852283535699884e-05,0.0037680028461350486,
        2.3874168658481505,102.29716297198993,0.0630840801073977,4.366414382069571,103.80165766466574,10.31191737479003,
        22.499669976290615,80.58633688538525,57.63899562767189,54.876972126712445,2255.853582731921,48.07560057471102,
        8520.440648946878,4.611016505306141,7288.209691433419,27.080644797412344,233.66410378190545,0.17087193970020584,
        0.05913691736090217,0.11570401881711151,0.11187032032939576,0.5324158245838093,0.02482405309900461,0.22571491780717992]#[0.0002638,0.03601,0.03163,0.1646,0.8968,0.0001398,0.001107,0.0009225,0.00002874,0.00377,2.3893,51.6586,0.06288,4.3349,54.9616,10.2951,22.5404,80.04645,57.5894,54.9884,2255.008323,47.651,8467.4876,4.6086,7253.9455,27.0272,233.5505,0.173,0.05904,0.1162,0.1133,0.5305,0.02491,0.2258]
        
    # for normalizing
    global_mean_parvals = [4.69761900e-04, 5.41163003e-01, 4.64404257e-02, 2.57347213e-01, 2.49147622e+00, 2.32355062e-04, 1.73008032e-03, 1.24773028e-03, 
        4.73120576e-05, 3.27273629e-02, 1.40959996e+01, 1.74308464e+02, 1.20270384e-01, 5.36849843e+01, 2.00183656e+02, 6.70741776e+01, 7.13633784e+01,
        8.90620600e+01, 7.91076788e+01, 9.10356840e+01, 2.83672027e+03, 9.49482902e+01, 1.17937196e+04, 2.67169199e+01, 9.11074726e+03, 2.88698402e+01,
        4.48707117e+02, 3.42753578e-01, 5.53779604e-02, 7.30414272e-02, 1.29523041e-01, 3.65134664e+00, 6.59925280e-02, 1.53881627e+00]
    
    npars = len(parnames)
    ncols = 6
    nrows = 0
    while nrows<npars:
        nrows+=ncols
    nrows = int(nrows/6)
    fig, axs = plt.subplots(nrows, ncols)
    fig.set_size_inches(13.5,12)
    
    opt_fs = np.zeros(len(parnames))
    count = 0
    for row in range(nrows):
        for col in range(ncols):
            if count<npars:

                div = global_mean_parvals[count] if norm else 1
                
                axs[row, col].plot(x, train_rmse.iloc[:,count+1].values/div, c='dodgerblue', linewidth=2, label='Train')
                axs[row, col].plot(x, test_rmse.iloc[:,count+1].values/div, c='darkgray', linewidth=1, label='Test')
                
                #axs[row, col].axhline(gl_test_rmse[count]/div, c='k', linewidth=1, linestyle='--', label='LC Regression')
                
                ys = savgol_filter(test_rmse.iloc[:,count+1].values/div, 15, 2)
                axs[row, col].plot(x, ys, c='crimson', linewidth=2,  label='Test (smoothed)')
                #axs[row, col].axvline(x=x[np.argmin(ys)], c='k', linewidth=1, linestyle='--')
                
                #axs[row, col].set_facecolor('honeydew') if min(ys)<gl_test_rmse[count]/div else axs[row, col].set_facecolor('mistyrose')
                
                opt_fs[count] = x[np.argmin(ys)]
                
                if npars>1: axs[row, col].set_title(parnames[count][:20])
        
                #if count==npars-1: axs[row, col].legend(loc='best')
                
                count += 1
                
    plt.subplots_adjust(hspace = .5,wspace=.5)
    plt.tight_layout()
    plt.savefig(savepath + savename + '.png') if not norm else plt.savefig(savepath + savename + 'normed.png')
    plt.close()
    return opt_fs


def retrieve_preds(pixel, opt_fs, suffix, pred_dir):
    # retrieve parameter predictions at a pixel given optimal number of features
    
    preds = np.zeros(len(opt_fs)) # equivalent to len(parnames)
    
    for par_ind in range(len(opt_fs)):
        file = glob.glob(pred_dir + '*test'+suffix.partition('fs')[0]+'fs'+ str(int(opt_fs[par_ind])) + '_*.csv')[0]
        data = read_csv(file)
        if (par_ind==11) | (par_ind==14):
            preds[par_ind] = data[pixel][par_ind] + 365.25
        else:
            preds[par_ind] = data[pixel][par_ind]

    return preds

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################



def main():
    
    # set run information to read
    model_id = sys.argv[1]
    mcmc_id = sys.argv[2] # 119 for normal, 3 for DEMCMC
    n_iter = sys.argv[3]
    nbe_optimization = sys.argv[4] # OFF OR ON
    ens_size = 500
    assim_type = '_p25adapted'
    suffix = '_clipped_'
    
    if mcmc_id=='119':
        frac_save_out = str(int(int(n_iter)/500))
        n_chains_agg = 4
    elif mcmc_id=='3':
        frac_save_out = str(int(int(n_iter)/500*100)) # n_iterations/ frac_save_out * 100 will be ensemble size
        n_chains_agg = 2
    
    # set directories
    cur_dir = os.getcwd() + '/'
    misc_dir = cur_dir + '/../../misc/'
    cbf_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbf' + assim_type+'/' + model_id + '/'
    cbr_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbr' + assim_type+'/' + model_id + '/'
    cbr_ef_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbr' + assim_type+'_ef/' + model_id + '/'
    plot_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = autil.get_parnames('../../misc/', model_id)
    
    # choose which features to use
    include_soilgrids = True
    include_poolobs = True
    include_gl_fracs = False
    
    # choose which model formulation to use
    train_full_ensemble = False
    rescale = True
    include_interactions = False
    include_squares = False 
    include_all_polys = False
    do_feature_selection = False
    do_PLS = True
    n_features_select = int(sys.argv[5])
    write_to_csv = False
    
    # choose which tasks to run
    opt_feature_select = True
    submit_ic_opt = True
    submit_forward = False
    
    
    
    ############################################################################################################################################
    ############################# develop and train EF models ###################################################################################
    
    
    # load list of land pixels
    pixels = list(set([file[-8:-4] for file in glob.glob(cbf_dir + '*.cbf')]))
    pixels.sort()
    
    # load list of cbrs
    cbr_files = glob.glob(cbr_dir+'*MCMC'+mcmc_id+'_'+n_iter+'_*.cbr')
    
    # load bestchains for cbr_files
    conv_chains = read_pickle(cbr_dir + model_id + assim_type + '_ALL' + '_MCMC'+mcmc_id + '_'+n_iter+'_best_subset.pkl')
    conv_chains.columns = ['pixel','bestchains','conv'] #rename columns for easier access
    ic_inds = autil.get_inds_ic(model_id) # get indices of initial condition parameters
    
    
    # load globcover csv for land cover regression comparison
    gl_fracs = read_csv(misc_dir+'globcover_fracs.csv', header=0)
    n_features_gl = len(gl_fracs.columns) - 1
    suffix_gl = 'gl_'
    
    # get number of predictors
    n_features = (rwb.read_cbf_file(glob.glob(cbf_dir + '*.cbf')[0])['nomet'] - 3) * 2 # remove 3 corresponding to day number and CO2, multiply by 2 (mean and sd)
    
    if do_PLS:
        suffix += 'PLS_'
    
    if include_soilgrids: 
        soilgrids = read_csv('../../misc/soilgrids_defined_pixels_manual.csv', header=0)
        n_soilgrids = len(soilgrids.columns) - 1
        n_features += n_soilgrids
        suffix += 'soilgrids_'
        
    if include_poolobs:
        n_poolobs = 4
        n_features += n_poolobs
        suffix += 'poolobs_'
        
    if include_gl_fracs:
        n_features += n_features_gl
        suffix += suffix_gl

    # fill X and Y
    n_regr_models = len(parnames)
    X = np.ones((len(pixels), n_features))*np.nan # shape n_samples, n_features
    y = np.ones((n_regr_models, len(pixels)))*np.nan # shape n_pars, n_samples
    y_full_ens = np.ones((ens_size, n_regr_models, len(pixels)))*np.nan # shape n_pars, n_samples
    
    X_gl = np.ones((len(pixels), n_features_gl))*np.nan
    y_gl = np.ones((n_regr_models, len(pixels)))*np.nan
    
    for pixel in pixels:
        if (len(glob.glob(cbr_dir + '*MCMC'+mcmc_id+'_'+n_iter+'_' + pixel + '*.cbr'))>0) & (pixel in conv_chains['pixel'].values):
            if conv_chains.loc[conv_chains['pixel']==pixel]['conv'].values[0]==0: continue
            else:
                ind = pixels.index(pixel)
                print(pixel)
                
                # get met
                cbf_file = glob.glob(cbf_dir + '*'+ pixel + '.cbf')[0]
                met = rwb.read_cbf_file(cbf_file)['MET']
                met = met[:,[1,2,3,6,7,8]] # don't use index 0, 5 (day numbers) or 4 (Co2)
                X_end = met.shape[1]*2
                X[ind,:X_end] = np.concatenate((np.nanmean(met, axis=0), np.nanstd(met, axis=0)))
                #X[ind,:met.shape[1]*12] = fill_X_met_12mo(X[ind,:met.shape[1]*12], met)#np.nanmean(met, axis=0)
                
                # append to X if include_soil_canopy_vars
                if include_soilgrids:
                    if (int(pixel) in soilgrids['pixel'].values):
                        X[ind,X_end:(X_end+n_soilgrids)] = soilgrids[soilgrids['pixel']==int(pixel)].values[0][1:]
                    X_end = X_end+n_soilgrids
                        
                if include_poolobs:
                    lai, agb, som = rwb.read_cbf_file(cbf_file)['OBS']['LAI'], rwb.read_cbf_file(cbf_file)['OBS']['ABGB'], rwb.read_cbf_file(cbf_file)['OBS']['SOM']
                    
                    if (len(lai)>0) & (len(agb)>0) & (len(som)>0):
                        X[ind,X_end:(X_end+n_poolobs)] = np.array([np.nanmean(lai[lai>0]), np.nanstd(lai[lai>0]), np.nanmean(agb[agb>0]), np.nanmean(som[som>0])])
                    X_end = X_end+n_poolobs
        
                if include_gl_fracs:
                    if (int(pixel) in gl_fracs['pixel'].values):
                        X[ind,X_end:(X_end+n_features_gl)] = gl_fracs.loc[gl_fracs['pixel']==int(pixel)].values[0][1:]
                    X_end = X_end+n_features_gl
    
                # fill globcover X
                if int(pixel) in gl_fracs['pixel'].values: X_gl[ind,:] = gl_fracs.loc[gl_fracs['pixel']==int(pixel)].values[0][1:]
                
                # get parameter information
                # get pixel's convergent chain numbers
                best_chains = conv_chains.loc[conv_chains['pixel']==pixel]['bestchains'].values[0][1:]
                print(best_chains)
                
                # aggregate bestchains from optimal posteriors
                cbr_data = []
                for chain in best_chains:
        
                    file = [i for i in cbr_files if pixel+'_'+chain+'.cbr' in i][0]
                    cbr_data.append(autil.modulus_Bday_Fday(rwb.read_cbr_file(file, {'nopars': len(parnames)}), parnames))
                    #cbr_data.append(rwb.read_cbr_file(file, {'nopars': len(parnames)}))
                    
                cbr_data = np.vstack(cbr_data)
                y[:,ind] = np.nanmedian(cbr_data, axis=0)
                y_gl[:,ind] = np.nanmedian(cbr_data, axis=0)
                
                indices = np.random.choice(cbr_data.shape[0], ens_size, replace=False) # only take a subset of cbr rows
                
                y_full_ens[:,:,ind] = cbr_data[indices,:]#reshape_cbr(cbr_data, ens_size*n_chains_agg)
    
    
    if not train_full_ensemble:
        
        f_bic = open(misc_dir + 'env_filter_manual/fs/bic_fs'+suffix.partition('fs')[0]+model_id+'_MCMC'+mcmc_id+'_'+n_iter+assim_type + '.csv', 'a')
        w_bic = csv.writer(f_bic)
        
        # EF regressions
        reg_test_preds_list, card_test_preds_list, reg_train_preds_list, card_train_preds_list, pixels_r, suffix, k = run_regressions(X, y, pixels, 
            rescale, include_interactions, include_squares, include_all_polys, do_feature_selection, do_PLS, write_to_csv, w_bic, n_features_select, 
            suffix, ens_size, n_regr_models, n_features)
            
        f_bic.close()
        
        # globcover comparison    
        '''gl_reg_test_preds_list, gl_card_test_preds_list, gl_reg_train_preds_list, gl_card_train_preds_list, gl_pixels_r, gl_suffix, gl_k = run_regressions(X_gl, y_gl, pixels, 
            rescale, False, False, False, False, False, False, w_bic, n_features_select, 
            suffix_gl, ens_size, n_regr_models, n_features_gl)'''
            
    else:
        suffix += 'full_ens_'
        
        icount = 0
        for i in sample(range(y_full_ens.shape[0]), 100):
            print(icount)
            rtest, ctest, rtrain, ctrain, pixels_r, suffix, k = run_regressions(X, y_full_ens[i,:,:], pixels, 
                rescale, include_interactions, include_squares, include_all_polys, do_feature_selection, n_features_select,
                suffix, ens_size, n_regr_models, n_features)
            
            reg_test_preds_list = [np.nanmedian(ri, axis=0) for ri in rtest] if icount==0 else [np.vstack((np.nanmedian(ri, axis=0), rfull)) for ri, rfull in zip(rtest, reg_test_preds_list)]
            card_test_preds_list = np.copy(ctest) if icount==0 else [np.vstack((ci, cfull)) for ci, cfull in zip(ctest, card_test_preds_list)]
            reg_train_preds_list = [np.nanmedian(ri, axis=0) for ri in rtrain] if icount==0 else [np.vstack((np.nanmedian(ri, axis=0), rfull)) for ri, rfull in zip(rtrain, reg_train_preds_list)]
            card_train_preds_list = np.copy(ctrain) if icount==0 else [np.vstack((ci, cfull)) for ci, cfull in zip(ctrain, card_train_preds_list)]

            icount += 1
            
    
    # fill csv
    
    f_test = open(misc_dir +'env_filter_manual/fs/fs_test'+suffix.partition('fs')[0]+model_id+'_MCMC'+mcmc_id+'_'+n_iter+assim_type + '.csv', 'a')
    wr_test = csv.writer(f_test)
    
    f_train = open(misc_dir +'env_filter_manual/fs/fs_train'+suffix.partition('fs')[0]+model_id+'_MCMC'+mcmc_id+'_'+n_iter+assim_type + '.csv', 'a')
    wr_train = csv.writer(f_train)
    
    f_test_preds = open(misc_dir +'env_filter_manual/par_preds/par_preds_test'+suffix+model_id+'_MCMC'+mcmc_id+'_'+n_iter+assim_type + '.csv', 'a')
    wr_test_preds = csv.writer(f_test_preds)
    
    f_train_preds = open(misc_dir +'env_filter_manual/par_preds/par_preds_train'+suffix+model_id+'_MCMC'+mcmc_id+'_'+n_iter+assim_type + '.csv', 'a')
    wr_train_preds = csv.writer(f_train_preds)
    
    
    print('TEST:')
    #plot_scatter_test_pred(card_test_preds_list, reg_test_preds_list, k, pixels_r, parnames, wr_test, wr_test_preds, plot_dir+'env_filter/', 'par_preds_test'+suffix+model_id+'_MCMC'+mcmc_id+'_'+n_iter+assim_type, train_full_ensemble, write_to_csv)
    #plot_scatter_test_pred(gl_card_test_preds_list, gl_reg_test_preds_list, gl_k, gl_pixels_r, parnames, wr_test, wr_test_preds, plot_dir+'env_filter/', 'par_preds_test'+gl_suffix+model_id+'_MCMC'+mcmc_id+'_'+n_iter+assim_type, train_full_ensemble, write_to_csv)
    
    print('. . . . . \n\nTRAIN:')
    #plot_scatter_test_pred(card_train_preds_list, reg_train_preds_list, k, pixels_r, parnames, wr_train, wr_train_preds, plot_dir+'env_filter/', 'par_preds_train'+suffix+model_id+'_MCMC'+mcmc_id+'_'+n_iter+assim_type, train_full_ensemble, write_to_csv)
    #plot_scatter_test_pred(gl_card_train_preds_list, gl_reg_train_preds_list, gl_k, gl_pixels_r, parnames, wr_train, wr_train_preds, plot_dir+'env_filter/', 'par_preds_train'+gl_suffix+model_id+'_MCMC'+mcmc_id+'_'+n_iter+assim_type, train_full_ensemble, write_to_csv)
    
    f_test.close()
    f_train.close()
    f_test_preds.close()
    f_train_preds.close()
    
    
    ############################################################################################################################################
    ################################### find optimal number of features for each parameter #####################################################
    
    
    if opt_feature_select:
    
        test_rmse = read_csv(misc_dir +'env_filter_manual/fs/fs_test'+suffix.partition('fs')[0]+model_id+'_MCMC'+mcmc_id+'_'+n_iter+assim_type + '.csv', header=None)
        test_rmse.columns = [item for sublist in [['n_features_select'],parnames] for item in sublist]
        test_rmse.sort_values('n_features_select')
        
        train_rmse = read_csv(misc_dir +'env_filter_manual/fs/fs_train'+suffix.partition('fs')[0]+model_id+'_MCMC'+mcmc_id+'_'+n_iter+assim_type + '.csv', header=None)
        train_rmse.columns = [item for sublist in [['n_features_select'],parnames] for item in sublist]
        train_rmse.sort_values('n_features_select')
        
        x = test_rmse['n_features_select'].values
        
        opt_fs = plot_train_test(x, train_rmse, test_rmse, parnames, savepath=plot_dir+'train_test/', savename=model_id+'_MCMC'+mcmc_id+suffix.partition('fs')[0], norm=False)
        opt_fs = plot_train_test(x, train_rmse, test_rmse, parnames, savepath=plot_dir+'train_test/', savename=model_id+'_MCMC'+mcmc_id+suffix.partition('fs')[0], norm=True)
        print(opt_fs)
        
        '''bic_data = read_csv(misc_dir +'env_filter_manual/fs/bic_fs_soilgrids_poolobs_'+model_id+'_MCMC'+mcmc_id+'_'+n_iter+assim_type + '.csv', header=None)
        bic_data.columns = [item for sublist in [['n_features_select'],parnames] for item in sublist]
        bic_data.columns.sort_values('n_features_select')
        
        x = bic_data['n_features_select'].values
        
        opt_fs = plot_train_test(x, bic_data, bic_data*np.nan, parnames, savepath=plot_dir+'train_test/', savename='bic_'+model_id+'_MCMC'+mcmc_id+suffix.partition('fs')[0])
        print(opt_fs)'''
    
    ############################################################################################################################################
    ################################### copy cbfs and substitute pars for IC optimization ######################################################
    
     # set directories for CARDAMOM runs
    mdf_dir = '../code/CARDAMOM_2.1.6c/C/projects/CARDAMOM_MDF/' if nbe_optimization=='OFF' else '../code/CARDAMOM_Uma_2.1.6c-master/C/projects/CARDAMOM_MDF/'
    runmodel_dir = '../code/CARDAMOM_2.1.6c/C/projects/CARDAMOM_GENERAL/' if nbe_optimization=='OFF' else '../code/CARDAMOM_Uma_2.1.6c-master/C/projects/CARDAMOM_GENERAL/'
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
    cbf_ef_ic_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'_ef_ic/' + model_id + '/'
    cbr_ef_dir = '../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'_ef/' + model_id + '/'
    output_dir = '../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
    output_ef_dir = '../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'_ef/' + model_id + '/'
    
    # select which pixels to submit
    os.chdir(cbf_dir)
    cbf_files = glob.glob('*.cbf')
    cbf_files.sort()
    os.chdir(cur_dir + '/../')
    
    
    if submit_ic_opt:
        
        txt_filename = 'ef_ic_assim_list_' + model_id + assim_type+'_MCMC'+mcmc_id + '_'+n_iter + '.txt'
        txt_file = open(txt_filename, 'w')
        
        for cbf_file in cbf_files:
            print(cbf_file)
    
            cbf_data = rwb.read_cbf_file(cbf_dir + cbf_file)
            cbf_pixel = cbf_file[-8:-4]
            
            if cbf_pixel in pixels_r:
                
                parpriors = np.concatenate((retrieve_preds(cbf_pixel, opt_fs, suffix, misc_dir +'env_filter_manual/par_preds/'), np.ones(50-len(parnames))*-9999.))
                parpriorunc = np.concatenate((np.ones(len(parnames))*1.001, np.ones(50-len(parnames))*-9999.))
                
                
                # except ICs
                for ic_ind in ic_inds:
                    parpriors[ic_ind] = -9999.
                    parpriorunc[ic_ind] = -9999.
                
                # except NBE unc
                if nbe_optimization=='ON': 
                    parpriors[len(parnames)-1] = -9999.
                    parpriorunc[len(parnames)-1] = -9999.
                
                cbf_data['PARPRIORS'] = parpriors.reshape(-1,1)
                cbf_data['PARPRIORUNC'] = parpriorunc.reshape(-1,1)

                fp = cbf_file[:-9]+suffix.partition('fs')[0]+cbf_pixel
                fa = cbf_file[:-9]+'_MCMC'+mcmc_id+'_'+n_iter+suffix.partition('fs')[0]+'assim_'+cbf_pixel
                rwb.CARDAMOM_WRITE_BINARY_FILEFORMAT(cbf_data, cbf_ef_ic_dir + fp +'.cbf')
                
                txt_file.write('%sCARDAMOM_MDF.exe %s%s %s%s %s 0 %s 0.001 %s 1000' % (mdf_dir, cbf_ef_ic_dir[3:], fp+'.cbf', cbr_ef_dir, fa+'.cbr', n_iter, frac_save_out, mcmc_id))
                txt_file.write('\n')
                
        txt_file.close()
    
        sh_file = open(txt_filename[:-3] + 'sh', 'w')
        autil.fill_in_sh(sh_file, array_size=len(pixels_r), n_hours=6, txt_file=txt_filename, combined=False)
    
    
    if submit_forward:
    
        txt_filename = 'ef_ic_forward_list_' + model_id + assim_type+'_MCMC'+mcmc_id + '_'+n_iter + '.txt'
        txt_file = open(txt_filename, 'w')
        
        for cbf_file in cbf_files:
            print(cbf_file)

            cbf_data = rwb.read_cbf_file(cbf_dir + cbf_file)
            cbf_pixel = cbf_file[-8:-4]
            
            if cbf_pixel in pixels_r:
                
                fa = cbf_file[:-9]+'_MCMC'+mcmc_id+'_'+n_iter+suffix.partition('fs')[0]+'assim_'+cbf_pixel
                cbr_assim = rwb.read_cbr_file(glob.glob(cbr_ef_dir+fa+'.cbr')[0], {'nopars': len(parnames)})
                
                ff = cbf_file[:-9]+'_MCMC'+mcmc_id+'_'+n_iter+suffix.partition('fs')[0]+'forward_'+cbf_pixel
                cbr_forward = retrieve_preds(cbf_pixel, opt_fs, suffix, misc_dir +'env_filter_manual/par_preds/')
                for ic_ind in ic_inds:
                    cbr_forward[ic_ind] = np.nanmedian(cbr_assim[:,ic_ind])
                cbr_forward = cbr_forward.reshape(1,len(parnames))
                
                rwb.write_cbr_file(cbr_forward, cbr_ef_dir + ff + '.cbr')
                
                txt_file.write('%sCARDAMOM_RUN_MODEL.exe %s%s %s%s %s%s %s%s %s%s %s%s' % (runmodel_dir, cbf_dir[3:], cbf_file, cbr_ef_dir, ff+'.cbr', 
                    output_ef_dir, 'fluxfile_'+ ff +'.bin', output_ef_dir, 'poolfile_'+ ff +'.bin', 
                    output_ef_dir, 'edcdfile_'+ ff +'.bin', output_ef_dir, 'probfile_'+ ff +'.bin'))
                txt_file.write('\n')
                
        txt_file.close()
    
        sh_file = open(txt_filename[:-3] + 'sh', 'w')
        autil.fill_in_sh(sh_file, array_size=len(pixels_r), n_hours=1, txt_file=txt_filename, combined=False)
    
    return

if __name__=='__main__':
    main()
