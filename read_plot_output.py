
import numpy as np
import glob
import sys
import os
from random import sample
from pandas import read_csv, read_pickle
import readwritebinary as rwb
import anova_utilities as autil

def main():
    model_id = sys.argv[1]
    run_type = sys.argv[2] # ALL or SUBSET
    mcmc_id = sys.argv[3] # 119 for normal, 3 for DEMCMC
    n_iter = sys.argv[4]
    ens_size = 500
    assim_type = '_p25adapted'
    use_bestchains_pkl = False
    
    cur_dir = os.getcwd() + '/'
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'_ef_ic/' + model_id + '/'
    cbr_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'_ef/' + model_id + '/'
    output_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'_ef/' + model_id + '/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = autil.get_parnames('../../misc/', model_id)
    
    
    # load list of land pixels
    pixels = list(set([file[-8:-4] for file in glob.glob(cbf_dir + '*.cbf')])) if run_type=='ALL' else ['3809','3524','2224','4170','1945','3813','4054','3264','1271','3457']
    pixels.sort()
    
    # load list of cbrs
    cbr_files = glob.glob(cbr_dir+'*MCMC'+mcmc_id+'_'+n_iter+'_*PLS*forward*.cbr')

    # for loop over pixels    
    gr_pixels = np.zeros(len(pixels))*np.nan # list of GR for each pixel, for mapping
    par_pixels = np.zeros((len(pixels), len(parnames)))*np.nan
    for pixel in pixels:
        print(pixel, pixels.index(pixel))
        
        pixel_chains = autil.find_all_chains(cbr_files, pixel)
        pixel_chains.sort() # filenames
        
        if use_bestchains_pkl:
            conv_chains_pkl = read_pickle(glob.glob(cbr_dir + model_id + assim_type + '*_MCMC'+mcmc_id + '_'+n_iter+'_best_subset.pkl')[0])
            conv_chains_pkl.columns = ['pixel','bestchains','conv'] #rename columns for easier access
            
            if pixel in conv_chains_pkl['pixel'].values:
                bestchains = conv_chains_pkl.loc[conv_chains_pkl['pixel']==pixel]['bestchains'].values[0][1:]
                print(bestchains)
                pixel_chains = [pixel_chain for pixel_chain in pixel_chains if pixel_chain.partition(pixel+'_')[-1][:-4] in bestchains]
            
            else:
                continue

        #cbf_pixel = rwb.read_cbf_file(cur_dir + cbf_dir + pixel_chains[0].partition('_MCMC')[0]+'_'+pixel+'.cbf')
        cbf_filename = glob.glob(cur_dir + cbf_dir + '*'+pixel+'.cbf')[0]
        cbf_pixel = rwb.read_cbf_file(cbf_filename)
        
        cbr_chain_list = []
        for pixel_chain in pixel_chains:
            print(pixel_chain)
            cbr_chain = rwb.read_cbr_file(pixel_chain, {'nopars': len(parnames)})
            cbr_chain = autil.modulus_Bday_Fday(cbr_chain, parnames)
            cbr_pixel = np.copy(cbr_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((cbr_pixel, cbr_chain), axis=0)
            #autil.plot_par_histograms(cbr_chain, parnames=parnames, savepath=cur_dir+plot_dir+'dists/', title=model_id+'_'+pixel_chain[:-3]+'png')
            
            try:
                flux_chain = rwb.readbinarymat(cur_dir + output_dir + 'fluxfile_' + pixel_chain.partition(cbr_dir)[-1][:-3]+'bin', [cbf_pixel['nodays'], autil.get_nofluxes_nopools_lma(model_id)[0]])
                pool_chain = rwb.readbinarymat(cur_dir + output_dir + 'poolfile_' + pixel_chain.partition(cbr_dir)[-1][:-3]+'bin', [cbf_pixel['nodays']+1, autil.get_nofluxes_nopools_lma(model_id)[1]])
                #autil.plot_flux_pool_timeseries(cbf_pixel, cbr_chain, flux_chain, pool_chain, autil.get_nofluxes_nopools_lma(model_id)[2], savepath=cur_dir+plot_dir+'timeseries/', title=model_id+'_'+pixel_chain[:-3]+'png')
    
                flux_pixel = np.copy(flux_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((flux_pixel, flux_chain), axis=0)
                pool_pixel = np.copy(pool_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((pool_pixel, pool_chain), axis=0)
                
            except Exception as e:
                pass
                
            if np.shape(cbr_chain)[0]==ens_size:
                cbr_chain_list.append(cbr_chain)
                #print(np.shape(cbr_chain))
            
        if len(cbr_chain_list)>1:
            gr = autil.gelman_rubin(cbr_chain_list)
            #print(gr)
            print('%i of %i parameters converged' % (sum(gr<1.2), len(parnames)))
            gr_pixels[pixels.index(pixel)] = sum(gr<1.2)/len(parnames)
        else:
            gr = np.nan

        par_pixels[pixels.index(pixel),:] = np.nanmedian(cbr_pixel, axis=0)
        #autil.plot_par_histograms(cbr_pixel, parnames=parnames, savepath=cur_dir+plot_dir+'dists/', title=model_id+assim_type+'_MCMC'+mcmc_id+'_'+cbf_filename.partition(cbf_dir)[-1][:-4]+'.png')    
        #autil.plot_flux_pool_timeseries(cbf_pixel, cbr_pixel, flux_pixel, pool_pixel, autil.get_nofluxes_nopools_lma(model_id)[2], savepath=cur_dir+plot_dir+'timeseries/', title=model_id+assim_type+'_MCMC'+mcmc_id+'_'+cbf_filename.partition(cbf_dir)[-1][:-4]+'.png')
        
    #vmax = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,210,200,215,6600,195,24000,None,None,None,900,None,None,None,None,None,None,None] #np.nanpercentile(par_pixels[:,par], 90)
    for par in range(len(parnames)): autil.plot_map(nrows=46, ncols=73, land_pixel_list=[file[-8:-4] for file in glob.glob(cur_dir + cbf_dir + '*.cbf')], pixel_value_list=pixels, value_list=par_pixels[:,par], vmax=np.nanpercentile(par_pixels[:,par], 90), savepath=cur_dir+plot_dir+'maps/', savename='par'+str(par)+'_' + model_id +assim_type+ '_MCMC' + mcmc_id +'_'+ n_iter+'_EF_clipped_PLS_soilgrids_poolobs_rescaled_forward')
    #autil.plot_map(nrows=46, ncols=73, land_pixel_list=[file[-8:-4] for file in glob.glob(cur_dir + cbf_dir + '*.cbf')], pixel_value_list=pixels, value_list=np.ones(len(pixels)), savepath=cur_dir+plot_dir+'maps/', title='test_pixels.png')
    #autil.plot_map(nrows=46, ncols=73, land_pixel_list=[file[-8:-4] for file in glob.glob(cur_dir + cbf_dir + '*.cbf')], pixel_value_list=pixels, value_list=gr_pixels*100, savepath=cur_dir+plot_dir+'maps/', savename='gr_' + model_id + assim_type+ '_' +run_type+ '_MCMC' + mcmc_id + '_' + n_iter)
        
    return

if __name__=='__main__':
    main()
