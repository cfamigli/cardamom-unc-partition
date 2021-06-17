
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

def find_rep(av_fracs, pixels):
    # identify representative pixels for each pft
    # also return fraction represented and dominance (1 or 0)
    av_fracs_copy = np.copy(av_fracs)
    max_frac = np.nanmax(av_fracs_copy, axis=0)
    max_frac_dom = np.ones(len(max_frac))*np.nan
    
    av_fracs_copy[av_fracs_copy==max_frac] = np.nan
    second_max_frac = np.nanmax(av_fracs_copy, axis=0)
    second_max_frac_dom = np.ones(len(second_max_frac))*np.nan
    
    av_fracs_copy[av_fracs_copy==second_max_frac] = np.nan
    third_max_frac = np.nanmax(av_fracs_copy, axis=0)
    third_max_frac_dom = np.ones(len(third_max_frac))*np.nan
    
    rep_pixels = []
    for i in range(len(max_frac)):
        
        if max_frac[i]>0:
            max_frac_rows = np.where(av_fracs[:,i]==max_frac[i])
            second_max_frac_rows = np.where(av_fracs[:,i]==second_max_frac[i])
            third_max_frac_rows = np.where(av_fracs[:,i]==third_max_frac[i])
            
            max_frac_dom[i] = 1 if max_frac[i]==np.max(av_fracs[max_frac_rows]) else 0
            second_max_frac_dom[i] = 1 if second_max_frac[i]==np.max(av_fracs[second_max_frac_rows]) else 0
            third_max_frac_dom[i] = 1 if third_max_frac[i]==np.max(av_fracs[third_max_frac_rows]) else 0
        
        rep_pixels.append([[pixels[row] for row in max_frac_rows[0]][0], 
            [pixels[row] for row in second_max_frac_rows[0]][0], 
            [pixels[row] for row in third_max_frac_rows[0]][0]]) if max_frac[i]>0 else rep_pixels.append(['-9999','-9999','-9999'])
        
    return rep_pixels, max_frac, second_max_frac, third_max_frac, max_frac_dom, second_max_frac_dom, third_max_frac_dom
    
    
def plot_reps(mxs, mxdoms, lbls, savepath, savename):
    # plot representative pixels and dominance for each pft
    # mxs, mxdoms are 3-member lists
    plt.figure(figsize=(9,5))
    colors = plt.cm.brg(np.linspace(0,0.9,len(mxs)))
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
    for lbl in lbls:
        ind = lbls.index(lbl)
        lst = rep_pixels[ind]
        lst.insert(0, lbl)
        
        for mx, mxdom in zip(mxs, mxdoms):
            lst.append(mx[ind])
            lst.append(mxdom[ind])
            
        df_data.append(lst)
            
    return DataFrame(df_data, columns=['pft','reppix1','reppix2','reppix3','reppix1frac','reppix1fracdom','reppix2frac','reppix2fracdom','reppix3frac','reppix3fracdom'])
    
def aggregate_parameter_sets(pixels_dom, all_cbr_files, parnames, ens_size):
    # aggregate parameter sets between representative pixels for a given pft, only if representative pixels are also dominant
    
    # get cbrs
    best_cbrs_list = []
    for pixel in pixels_dom:
        
        pixel_chains = autil.find_all_chains(all_cbr_files, pixel)
        pixel_chains.sort()
        
        continue_check = True
        for subset in [list(i) for i in itertools.combinations(pixel_chains, 4)]:
        # read parameters and compute gelman rubin
            
            if continue_check:
                cbr_chain_list = []
                for cbr_file in subset:
                    cbr_chain = rwb.read_cbr_file(cbr_file, {'nopars': len(parnames)})
                    
                    if np.shape(cbr_chain)[0]==ens_size: cbr_chain_list.append(cbr_chain)
                    
                if len(cbr_chain_list)>1:
                    gr = autil.gelman_rubin(cbr_chain_list)
                    #print('%i of %i parameters converged' % (sum(gr<1.2), len(parnames)))
                    
                    if sum(gr<1.2)/len(parnames)>0.9: 
                        continue_check = False
                        best_subset = subset.copy()
                    else: best_subset = []
                        
                else:
                    gr = np.nan
        
        if len(best_subset)>0:
            best_cbrs_list.extend(cbr_chain_list)
            best_cbrs_agg = np.vstack(best_cbrs_list)
        else: 
            best_cbrs_agg = np.ones((ens_size*4, len(parnames)))*np.nan

    print(best_cbrs_agg.shape)
    random_rows = np.random.choice(best_cbrs_agg.shape[0], ens_size*4, replace=False)
    best_cbrs_sampled = best_cbrs_agg[random_rows, :]
    print(best_cbrs_sampled.shape)

    return best_cbrs_sampled

def main():
    
    # set run information to read
    model_id = sys.argv[1]
    mcmc_id = sys.argv[2] # 119 for normal, 3 for DEMCMC
    n_iter = sys.argv[3]
    ens_size = 500
    assim_type = '_longadapted'
    
    # set directories
    cur_dir = os.getcwd() + '/'
    misc_dir = cur_dir + '../../misc/'
    cbf_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbf' + assim_type+'/' + model_id + '/'
    cbr_dir = cur_dir + '../../../../../../scratch/users/cfamigli/cardamom/files/cbr' + assim_type+'/' + model_id + '/'
    plot_dir = cur_dir + '../../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = autil.get_parnames('../../misc/', model_id)
    
    
    ############################## Identify and save representative pixels #################################################
    
    # load globcover data
    gl = read_pickle(misc_dir+'globcover_to_card.pkl')
    
    # load labels
    gl_lbls = list(read_csv(misc_dir+'Globcover2009_Legend.csv')['Value'].values)
    n_classes = len(gl_lbls)
        
    # load list of land pixels
    pixels = list(set([file[-8:-4] for file in glob.glob(cbf_dir + '*.cbf')]))
    
    # get list of average pft fractions by pixel
    av_fracs = np.ones((len(pixels), n_classes))*np.nan
    for pixel in pixels:
        ind = pixels.index(pixel)
        if np.mod(ind, 10)==0: print(ind)
        
        # get lc information
        types_at_geos_pixel, counts_at_geos_pixel = gl.loc[gl['pixel']==pixel]['types'].values[0][0], gl.loc[gl['pixel']==pixel]['counts'].values[0][0]
        
        types_at_geos_pixel, counts_at_geos_pixel = remove_nodata_pixels(types_at_geos_pixel, counts_at_geos_pixel)
        types_at_geos_pixel, counts_at_geos_pixel = append_all_types(types_at_geos_pixel, counts_at_geos_pixel, gl_lbls)
        types_at_geos_pixel, counts_at_geos_pixel = merge_types(types_at_geos_pixel, counts_at_geos_pixel, 170, 160)
        types_at_geos_pixel, counts_at_geos_pixel = merge_types(types_at_geos_pixel, counts_at_geos_pixel, 180, 160)
        
        if np.sum(counts_at_geos_pixel)>0:
            av_fracs[ind] = counts_at_geos_pixel/np.sum(counts_at_geos_pixel) # average biome fraction across mstmip pixels within coarse pixel
            #plot_pie(av_fracs[ind], pixel, gl_lbls, autil.rowcol_to_latlon([pixel]), plot_dir+'pie/', 'gl')
    
    reps, mx, mx2, mx3, mxdom, mx2dom, mx3dom = find_rep(av_fracs, pixels)
    plot_reps([mx, mx2, mx3], [mxdom, mx2dom, mx3dom], gl_lbls, plot_dir+'pie/', 'rep_pix_gl_merge170+180to160')
    
    rep_df = fill_df(gl_lbls, reps, [mx, mx2, mx3], [mxdom, mx2dom, mx3dom])
    #rep_df.to_pickle(misc_dir+ 'rep_pixels_globcover.pkl')
    print(rep_df)
    
    
    ############################## Generate aggregated parameter sets ######################################################
    
    # load list of cbrs
    files = glob.glob(cbr_dir + '*MCMC'+mcmc_id+'_'+n_iter+'_*.cbr')
    
    pfts_to_run_forward = []
    for pft in gl_lbls:
        print('PFT: ' + str(pft))
        # isolate row in dataframe corresponding to given pft
        rep_df_pft = rep_df.loc[rep_df['pft']==int(pft)]
        
        # get list of pixels that are dominant
        pixels = [rep_df_pft['reppix1'].values[0], rep_df_pft['reppix2'].values[0], rep_df_pft['reppix3'].values[0]]
        doms = [rep_df_pft['reppix1fracdom'].values[0], rep_df_pft['reppix2fracdom'].values[0], rep_df_pft['reppix3fracdom'].values[0]]
        pixels_dom = [pixel for pixel in pixels if doms[pixels.index(pixel)]==1]
        
        if len(pixels_dom)>0: 
            par_set_agg = aggregate_parameter_sets(pixels_dom, files, parnames, ens_size)
            rwb.write_cbr_file(par_set_agg, cbr_dir+files[0].partition(cbr_dir)[-1][:-10]+'PFT'+str(pft)+'_GLREPAGG.cbr')
            pfts_to_run_forward.append(str(pft))
            
            #if np.sum(~np.isnan(par_set_agg))>0: autil.plot_par_histograms(par_set_agg, parnames, savepath=plot_dir+'dists/', title='globcover_agg_PFT'+str(pft)+'_'+model_id+assim_type+'_'+mcmc_id+'_'+n_iter+'.pdf')
                
                
    ############################## Create submission txt and sh files ######################################################
    
    # set directories for CARDAMOM runs
    runmodel_dir = '../code/CARDAMOM_2.1.6c/C/projects/CARDAMOM_GENERAL/'
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
    cbr_dir = '../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
    output_dir = '../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
    
     # select which pixels to submit
    os.chdir(cbf_dir)
    cbf_files = glob.glob('*.cbf')
    os.chdir(cur_dir + '/../')
    
     # set up txt file to contain forward runs
    txt_filename = 'pft_forward_list_' + model_id + '_MCMC'+mcmc_id + '_'+n_iter + '.txt'
    txt_file = open(txt_filename, 'w')
    
    
    # fill in txt file
    # each row is a pixel containing a forward run for each pft
    for cbf_file in cbf_files:
        for pft in pfts_to_run_forward:
                suffix = '_PFT'+pft+'_GLREPAGG'
                txt_file.write('%sCARDAMOM_RUN_MODEL.exe %s%s %s%s %s%s %s%s %s%s %s%s' % (runmodel_dir, cbf_dir[3:], cbf_file, cbr_dir, cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iter+ suffix +'.cbr', 
                    output_dir, 'fluxfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iter+'_'+cbf_file[-8:-4]+ suffix +'.bin', output_dir, 'poolfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iter+'_'+cbf_file[-8:-4]+ suffix +'.bin', 
                    output_dir, 'edcdfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iter+'_'+cbf_file[-8:-4]+ suffix +'.bin', output_dir, 'probfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iter+'_'+cbf_file[-8:-4]+ suffix +'.bin'))
                txt_file.write(' && ') if pfts_to_run_forward.index(pft)<len(pfts_to_run_forward)-1 else txt_file.write('\n')
                
    txt_file.close()
        
    sh_file = open(txt_filename[:-3] + 'sh', 'w')
    autil.fill_in_sh(sh_file, array_size=len(cbf_files), n_hours=1, txt_file=txt_filename, combined=True)
    
    return

if __name__=='__main__':
    main()