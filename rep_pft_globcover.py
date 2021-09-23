
"""

created on Wed Jun 9 13:05:41 2021

@author: cfamigli

find most representative pixel for each PFT (Globcover)

"""

import numpy as np
import readwritebinary as rwb
from pandas import read_pickle, read_csv, DataFrame, to_pickle
import os
import glob
import sys
import csv
import itertools
from random import sample
import anova_utilities as autil
import matplotlib
import matplotlib.pyplot as plt

def remove_nodata_pixels(types, counts):
    # remove Globcover pixels that are water, urban, or no data from sums
    ind_remove = [(types==190) | (types==210) | (types==220) | (types==230)][0]
    counts[ind_remove] = 0
    return types, counts
    
def merge_types(types, counts, pft_remove, pft_accept):
    # combine pft types according to inputs pft_remove adn pft_accept
    counts[types==pft_accept] += counts[types==pft_remove]
    counts[types==pft_remove] = 0
    return types, counts
    
def append_all_types(types, counts, lbls):
    # create list of all pfts and corresponding counts (fill with zero counts if a given pft is not represented)
    types_not_in_pixel = np.setdiff1d(lbls, types)
    counts_not_in_pixel = np.zeros(len(types_not_in_pixel))
    
    concat_types = np.concatenate((types, types_not_in_pixel))
    concat_counts = np.concatenate((counts, counts_not_in_pixel))
    
    inds_sm_lrg = np.argsort(concat_types)
    return concat_types[inds_sm_lrg], concat_counts[inds_sm_lrg]
    

def plot_pie(pie_data, pixel_num, labels, latlon, savepath, datasource):
    plt.figure(figsize=(3,3))
    cl = dict(zip(labels, plt.cm.Spectral(np.linspace(0, 1, len(labels)))))
    
    # ensure that water(=0) is not plotted
    if np.nansum(pie_data)<1:
        plt.pie(pie_data*1e9, labels=labels, colors=[cl[key] for key in labels])
    else:
        plt.pie(pie_data, labels=labels, colors=[cl[key] for key in labels])
    plt.title('Pixel: ' + pixel_num +('\nlat/lon: ' + str(latlon[0])))
    plt.tight_layout()
    plt.savefig(savepath + pixel_num + '_' + datasource +'.png')
    plt.close()
    return
    
def find_rep(av_fracs, pixels, n_reps):
    # identify representative pixels for each pft
    # also return fraction represented and dominance (1 or 0)
    
    mxs = []
    mxfracdoms = []
    
    av_fracs_copy = np.copy(av_fracs)
    for j in range(n_reps):
        max_frac = np.nanmax(av_fracs_copy, axis=0)
        max_frac_dom = np.ones(len(max_frac))*np.nan
        
        mxs.append(max_frac)
        mxfracdoms.append(max_frac_dom)
        av_fracs_copy[av_fracs_copy==max_frac] = np.nan
    
    rep_pixels = []
    for i in range(len(mxs[0])):
        
        rep_pixels_class = []
        if mxs[0][i]>0:
            mxfracrows = []
            
            count = 0
            for mx in mxs:
                max_frac_rows = np.where(av_fracs[:,i]==mx[i])
                mxfracrows.append(max_frac_rows)
                mxfracdoms[count][i] = 1 if mx[i]==np.max(av_fracs[max_frac_rows]) else 0
                rep_pixels_class.append([pixels[row] for row in max_frac_rows[0]][0])
                
                count += 1
        
            rep_pixels.append(rep_pixels_class)
        
        else: rep_pixels.append(['-9999']*n_reps)
        
    #print(rep_pixels)
    return rep_pixels, mxs, mxfracdoms
    
    
def plot_reps(mxs, mxdoms, lbls, savepath, savename):
    # plot representative pixels and dominance for each pft
    # mxs, mxdoms are 3-member lists
    plt.figure(figsize=(9,5))
    colors = plt.cm.rainbow(np.linspace(0,0.95,len(mxs)))
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
    plt.ylim([0,1])
    plt.xlabel('Class')
    plt.ylabel('Fraction represented')
    plt.title('Representative pixels')
    plt.tight_layout()
    plt.savefig(savepath + savename + '.png')
    plt.close()
    return
    
def fill_df(lbls, rep_pixels, mxs, mxdoms):
    # fill dataframe containing representative pixels and dominance for each pft
    df_data = []
    columns = ['pft']
    for lbl in lbls:
        ind = lbls.index(lbl)
        lst = rep_pixels[ind]
        lst.insert(0, lbl)
        
        for mx, mxdom in zip(mxs, mxdoms):
            lst.append(mx[ind])
            lst.append(mxdom[ind])
            
        df_data.append(lst)
    
    for i in range(1,len(mxs)+1):
        columns.append('reppix'+str(i))
    for i in range(1,len(mxs)+1):
        columns.append('reppix'+str(i)+'frac')
        columns.append('reppix'+str(i)+'fracdom')
            
    return DataFrame(df_data, columns=columns)
    
'''def aggregate_parameter_sets_iterate(pixels_dom, all_cbr_files, parnames, ens_size, n_chains_agg):
    # aggregate parameter sets between representative pixels for a given pft, only if representative pixels are also dominant
    
    # get cbrs
    best_cbrs_list = []
    best_cbrs_agg = None
    for pixel in pixels_dom:
        best_subset = []
        print(pixel)
        
        pixel_chains = autil.find_all_chains(all_cbr_files, pixel)
        pixel_chains.sort()
        
        continue_check = True
        for subset in [list(i) for i in itertools.combinations(pixel_chains, n_chains_agg)]:
        # read parameters and compute gelman rubin
            
            if continue_check:
                cbr_chain_list = []
                for cbr_file in subset:
                    cbr_chain = rwb.read_cbr_file(cbr_file, {'nopars': len(parnames)})
                    
                    if np.shape(cbr_chain)[0]==ens_size: cbr_chain_list.append(cbr_chain)
                    
                if len(cbr_chain_list)>1:
                    gr = autil.gelman_rubin(cbr_chain_list)
                    print('%i of %i parameters converged' % (sum(gr<1.2), len(parnames)))
                    
                    if sum(gr<1.2)/len(parnames)>0.9:
                        print('converged')
                        continue_check = False
                        best_subset = subset.copy()
                    else: best_subset = []
                        
                else:
                    gr = np.nan
                    best_subset = []
        
        if len(best_subset)>0:
            best_cbrs_list.extend(cbr_chain_list)
            best_cbrs_agg = np.vstack(best_cbrs_list)
        #else: 
            #best_cbrs_agg = np.ones((ens_size*n_chains_agg, len(parnames)))*np.nan

    
    if best_cbrs_agg is not None:
        
        print(best_cbrs_agg.shape)
        random_rows = np.random.choice(best_cbrs_agg.shape[0], ens_size*n_chains_agg, replace=False)
        best_cbrs_sampled = best_cbrs_agg[random_rows, :]
        print(best_cbrs_sampled.shape)
        return best_cbrs_sampled
        
    else:
        return np.ones((ens_size*n_chains_agg, len(parnames)))*np.nan'''
    
def aggregate_parameter_sets(pixels_dom, all_cbr_files, parnames, ens_size, n_chains_agg, conv_chains_pkl):
    # aggregate parameter sets between representative pixels for a given pft, only if representative pixels are also dominant
    
    # get cbrs
    par_set_agg = []
    for pixel in pixels_dom:
        par_set = []
        
        if pixel in conv_chains_pkl['pixel'].values:
            print(pixel)
            
            # get pixel's convergent chain numbers
            best_chains = conv_chains_pkl.loc[conv_chains_pkl['pixel']==pixel]['bestchains'].values[0][1:]
            print(best_chains)
            
            # aggregate bestchains from optimal posteriors
            par_set_orig = []
            for chain in best_chains:

                file = [i for i in all_cbr_files if pixel+'_'+chain+'.cbr' in i][0]
                par_set.append(autil.modulus_Bday_Fday(rwb.read_cbr_file(file, {'nopars': len(parnames)}), parnames))
            
        else: 
            par_set = np.ones((ens_size*n_chains_agg, len(parnames)))*np.nan
        
        par_set_agg.append(np.vstack(par_set))
        
    par_set_agg = np.vstack(par_set_agg)
    print(par_set_agg.shape)
    
    random_rows = np.random.choice(par_set_agg.shape[0], ens_size*n_chains_agg, replace=False)
    best_cbrs_sampled = par_set_agg[random_rows, :]
    print(best_cbrs_sampled.shape)
    print(np.nanmedian(best_cbrs_sampled, axis=0))
    return best_cbrs_sampled

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################

def main():
    
    # set run information to read
    model_id = sys.argv[1]
    mcmc_id = sys.argv[2] # 119 for normal, 3 for DEMCMC
    n_iter = sys.argv[3]
    nbe_optimization = sys.argv[4] # 'OFF' or 'ON'
    ens_size = 250
    assim_type = sys.argv[5]
    n_chains_agg = 4
    
    # set directories
    cur_dir = os.getcwd() + '/'
    misc_dir = cur_dir + '../../misc/'
    cbf_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbf' + assim_type+'/' + model_id + '/'
    cbr_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbr' + assim_type+'/' + model_id + '/'
    plot_dir = cur_dir + '../../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = autil.get_parnames('../../misc/', model_id)
    
    # decide which tasks to perform
    find_rep_pixels = True
    agg_parameters = True
    submit_ic_opt = True
    submit_forward = False
    
    ############################## Identify and save representative pixels #################################################
    
    n_reps = 5
    if find_rep_pixels:
        # load globcover data
        gl = read_pickle(misc_dir+'globcover_to_card.pkl')
        
        # load labels
        gl_lbls = list(read_csv(misc_dir+'Globcover2009_Legend.csv')['Value'].values)
        n_classes = len(gl_lbls)
        print(gl_lbls)
            
        # load list of land pixels
        pixels = list(set([file[-8:-4] for file in glob.glob(cbf_dir + '*.cbf')]))
        pixels.sort()
        
        
        # open csv for save out
        f = open(misc_dir + 'globcover_fracs.csv','w')
        writer = csv.writer(f)
        writer.writerow([item for sublist in [['pixel'],gl_lbls] for item in sublist])
        
        # get list of average pft fractions by pixel
        av_fracs = np.ones((len(pixels), n_classes))*np.nan
        types_present = []
        for pixel in pixels:
            ind = pixels.index(pixel)
            if np.mod(ind, 100)==0: print(ind)
            
            # get lc information
            types_at_geos_pixel, counts_at_geos_pixel = gl.loc[gl['pixel']==pixel]['types'].values[0][0], gl.loc[gl['pixel']==pixel]['counts'].values[0][0]
            
            types_at_geos_pixel, counts_at_geos_pixel = remove_nodata_pixels(types_at_geos_pixel, counts_at_geos_pixel)
            types_at_geos_pixel, counts_at_geos_pixel = append_all_types(types_at_geos_pixel, counts_at_geos_pixel, gl_lbls)
            types_at_geos_pixel, counts_at_geos_pixel = merge_types(types_at_geos_pixel, counts_at_geos_pixel, 170, 160)
            types_at_geos_pixel, counts_at_geos_pixel = merge_types(types_at_geos_pixel, counts_at_geos_pixel, 180, 160)
            types_present.append(types_at_geos_pixel[counts_at_geos_pixel>0])

            if np.sum(counts_at_geos_pixel)>0:
                av_fracs[ind,:] = counts_at_geos_pixel/np.sum(counts_at_geos_pixel) # average biome fraction across mstmip pixels within coarse pixel
                
                writer.writerow([item for sublist in [[pixel],av_fracs[ind,:]] for item in sublist])
                
                #plot_pie(av_fracs[ind], pixel, gl_lbls, autil.rowcol_to_latlon([pixel]), plot_dir+'pie/', 'gl')
        
        reps, mxs, mxdoms = find_rep(av_fracs, pixels, n_reps)
        plot_reps(mxs, mxdoms, gl_lbls, plot_dir+'pie/', 'rep_pix_gl_merge170+180to160')
        
        rep_df = fill_df(gl_lbls, reps, mxs, mxdoms)
        #rep_df.to_pickle(misc_dir+ 'rep_pixels_globcover.pkl')
        print(rep_df)
    
        f.close()
        
    ############################## Generate aggregated parameter sets ######################################################
    
    ic_inds = autil.get_inds_ic(model_id)
    conv_chains = read_pickle(cbr_dir + model_id + assim_type + '_ALL' + '_MCMC'+mcmc_id + '_'+n_iter+'_best_subset.pkl')
    conv_chains.columns = ['pixel','bestchains','conv'] #rename columns for easier access
    
    if agg_parameters:
         
        #f_pft = open(misc_dir + 'pft/par_preds/par_set_agg_'+ model_id + assim_type+'_MCMC'+mcmc_id + '_'+n_iter + '.csv', 'w')   
        #w_pft = csv.writer(f_pft) 
         
        # load list of cbrs
        files = glob.glob(cbr_dir + '*MCMC'+mcmc_id+'_'+n_iter+'_*.cbr')
        files.sort()
        
        # get aggregated parameter sets from representative pixels
        par_set_agg = []
        for pft in gl_lbls:
            print(pft)
            print('PFT: ' + str(pft))
            # isolate row in dataframe corresponding to given pft
            rep_df_pft = rep_df.loc[rep_df['pft']==int(pft)]
            
            # get list of pixels that are dominant
            rep_pixels_pft = [rep_df_pft['reppix'+str(i)].values[0] for i in range(1,n_reps+1)]
            doms = [rep_df_pft['reppix'+str(i)+'fracdom'].values[0] for i in range(1,n_reps+1)]
            pixels_dom = [pixel for pixel in rep_pixels_pft if doms[rep_pixels_pft.index(pixel)]==1]
            
            if len(pixels_dom)>0: 
                par_set_agg.append(aggregate_parameter_sets(pixels_dom, files, parnames, ens_size, n_chains_agg, conv_chains))
            else:
                par_set_agg.append(np.ones((ens_size*n_chains_agg, len(parnames)))*np.nan)
                
            #w_pft.writerow(np.nanmedian(par_set_agg[gl_lbls.index(pft)], axis=0))
                
            #if np.sum(~np.isnan(par_set_agg[gl_lbls.index(pft)]))>0: autil.plot_par_histograms(par_set_agg[gl_lbls.index(pft)], parnames, savepath=plot_dir+'dists/', title='globcover_agg_PFT'+str(pft)+'_'+model_id+assim_type+'_'+mcmc_id+'_'+n_iter+'.pdf')
                
        #f_pft.close()
            
    ############################################################################################################################################
    ################################### copy cbfs and substitute pars for IC optimization ######################################################
    
    
    # set up cbfs for IC assimilation
    os.chdir(cbf_dir)
    cbf_files = glob.glob('*.cbf')
    cbf_files.sort()
    os.chdir(cur_dir + '/../')
    
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
    
    par_set_csv = read_csv(misc_dir + 'pft/par_preds/par_set_agg_'+ model_id + assim_type+'_MCMC'+mcmc_id + '_'+n_iter + '.csv', header=None).values
    
    if submit_ic_opt:
        
        txt_filename = 'pft_ic_assim_list_' + model_id + assim_type+'_MCMC'+mcmc_id + '_'+n_iter + '.txt'
        txt_file = open(txt_filename, 'w')
        
        
        for cbf_file in cbf_files:
            print(cbf_file)
    
            cbf_data = rwb.read_cbf_file(cbf_dir + cbf_file)
            cbf_pixel = cbf_file[-8:-4]
            
            if cbf_pixel in conv_chains['pixel'].values:
                
                for pft in gl_lbls:

                    if (int(pft) in types_present[pixels.index(cbf_pixel)]) & (~np.isnan(par_set_csv[gl_lbls.index(pft), :]).all()):
                        
                        par_set_agg_cbf = np.copy(par_set_csv[gl_lbls.index(pft), :])
                        # re-transform bday, fday to proper range
                        par_set_agg_cbf[11] += 365.25
                        par_set_agg_cbf[14] += 365.25 
                
                        parpriors = np.concatenate((par_set_agg_cbf, np.ones(50-len(parnames))*-9999.))
                        parpriorunc = np.concatenate((np.ones(len(parnames))*1.001, np.ones(50-len(parnames))*-9999.))
                        
                        for ic_ind in ic_inds:
                            parpriors[ic_ind] = -9999.
                            parpriorunc[ic_ind] = -9999.
                        
                        if nbe_optimization=='ON': 
                            parpriors[len(parnames)-1] = -9999
                            parpriorunc[len(parnames)-1] = -9999
                            
                        cbf_data['PARPRIORS'] = parpriors.reshape(-1,1)
                        cbf_data['PARPRIORUNC'] = parpriorunc.reshape(-1,1)
                        
                        f = cbf_file[:-9]+'_MCMC'+mcmc_id+'_'+n_iter+'_PFT'+str(pft)+'_assim_'+cbf_pixel
                        #rwb.CARDAMOM_WRITE_BINARY_FILEFORMAT(cbf_data, cbf_pft_ic_dir + f +'.cbf')
                        
                        txt_file.write('%sCARDAMOM_MDF.exe %s%s %s%s %s 0 %s 0.001 %s 1000' % (mdf_dir, cbf_pft_ic_dir[3:], f+'.cbf', cbr_pft_dir, f+'.cbr', n_iter, frac_save_out, mcmc_id))
                        txt_file.write('\n') if types_present[pixels.index(cbf_pixel)][-1]==int(pft) else txt_file.write(' && ')
                
        txt_file.close()
    
        sh_file = open(txt_filename[:-3] + 'sh', 'w')
        autil.fill_in_sh(sh_file, array_size=len(conv_chains['pixel'].values), n_hours=48, txt_file=txt_filename, combined=True)
                 
                    
    if submit_forward:
        
        txt_filename = 'pft_ic_forward_list_' + model_id + assim_type+'_MCMC'+mcmc_id + '_'+n_iter + '.txt'
        txt_file = open(txt_filename, 'w')
        
        for cbf_file in cbf_files:
            print(cbf_file)
    
            cbf_data = rwb.read_cbf_file(cbf_dir + cbf_file)
            cbf_pixel = cbf_file[-8:-4]
            
            if cbf_pixel in conv_chains['pixel'].values:
                
                for pft in gl_lbls:
                    
                    if (int(pft) in types_present[pixels.index(cbf_pixel)]) & (~np.isnan(par_set_csv[gl_lbls.index(pft), :]).all()):
                        
                        f = cbf_file[:-9]+'_MCMC'+mcmc_id+'_'+n_iter+'_PFT'+str(pft)+'_assim_'+cbf_pixel
                        
                        if len(glob.glob(cbr_pft_dir+f+'.cbr'))>0:
                            cbr_assim = rwb.read_cbr_file(glob.glob(cbr_pft_dir+f+'.cbr')[0], {'nopars': len(parnames)})
                            
                            ff = cbf_file[:-9]+'_MCMC'+mcmc_id+'_'+n_iter+'_PFT'+str(pft)+'_forward_'+cbf_pixel
                            cbr_forward = par_set_csv[gl_lbls.index(pft), :]
                            for ic_ind in ic_inds:
                                cbr_forward[ic_ind] = np.nanmedian(cbr_assim[:,ic_ind])
                            cbr_forward = cbr_forward.reshape(1,len(parnames))
                    
                            rwb.write_cbr_file(cbr_forward, cbr_pft_dir + ff + '.cbr')
                            
                            txt_file.write('%sCARDAMOM_RUN_MODEL.exe %s%s %s%s %s%s %s%s %s%s %s%s' % (runmodel_dir, cbf_dir[3:], cbf_file, cbr_pft_dir, ff+'.cbr', 
                                output_pft_dir, 'fluxfile_'+ ff +'.bin', output_pft_dir, 'poolfile_'+ ff +'.bin', 
                                output_pft_dir, 'edcdfile_'+ ff +'.bin', output_pft_dir, 'probfile_'+ ff +'.bin'))
                            txt_file.write('\n') if types_present[pixels.index(cbf_pixel)][-1]==int(pft) else txt_file.write(' && ')
        
        txt_file.close()
    
        sh_file = open(txt_filename[:-3] + 'sh', 'w')
        autil.fill_in_sh(sh_file, array_size=len(conv_chains['pixel'].values), n_hours=1, txt_file=txt_filename, combined=True)
    
    return

if __name__=='__main__':
    main()