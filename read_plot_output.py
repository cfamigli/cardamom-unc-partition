
import numpy as np
import glob
import sys
import os
from random import sample
from pandas import read_csv
import readwritebinary as rwb
import basic_plots

def get_parnames(par_dir, model_id):
    pars = read_csv(par_dir+'parameters.csv', skiprows=1)
    if model_id=='102':
        model_id = '101'
    if model_id in pars.columns:
        parnames = pars[model_id].dropna()
    else:
        parnames = []
    return parnames
    
def get_nofluxes_nopools_leaffall(model_id):
    return {
        '809': (30,7,15),
        '811': (30,7,15),
        '831': (30,7,15),
        '101': (28,3,9),
        '102': (28,3,9),
       '1003': (33,8,15),
       '400': (28,6,15),
       '1010': (43,8,15),
       '1000': (32,8,15)
    }[model_id]
    
def find_all_chains(file_list, pixel):
    # takes a list of file names and a pixel number (string)
    # outputs list of files corresponding to chains run at that pixel
    
    return [file for file in file_list if file[-10:-6]==pixel]
    
def gelman_rubin(x, return_var=False):
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

def main():
    model_id = sys.argv[1]
    run_type = sys.argv[2] # ALL or SUBSET
    mcmc_id = sys.argv[3] # 119 for normal, 3 for DEMCMC
    n_iter = sys.argv[4]
    assim_type = '_longadapted'
    
    cur_dir = os.getcwd() + '/'
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
    cbr_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
    output_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = get_parnames('../../misc/', model_id)
    
    os.chdir(cbr_dir)
    #files = glob.glob('*.cbr')#sample(glob.glob('*.cbr'), 10)
    files = glob.glob('*MCMC'+mcmc_id+'_'+n_iter+'_*.cbr')
    pixels = list(set([file[-10:-6] for file in files]))
    
    for pixel in pixels:
        pixel_chains = find_all_chains(files, pixel)
        pixel_chains.sort() # filenames
        print(pixel_chains)
        
        cbf_pixel = rwb.read_cbf_file(cur_dir + cbf_dir + pixel_chains[0].partition('_MCMC')[0]+'_'+pixel+'.cbf')
        
        cbr_chain_list = []
        for pixel_chain in pixel_chains:
            print(pixel_chain)
            cbr_chain = rwb.read_cbr_file(pixel_chain, {'nopars': len(parnames)})
            cbr_pixel = np.copy(cbr_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((cbr_pixel, cbr_chain), axis=0)
            basic_plots.plot_par_histograms(cbr_chain, parnames=parnames, savepath=cur_dir+plot_dir+'dists/', title=model_id+'_'+pixel_chain[:-3]+'png')
            
            flux_chain = rwb.readbinarymat(cur_dir + output_dir + 'fluxfile_' + pixel_chain[:-3]+'bin', [cbf_pixel['nodays'], get_nofluxes_nopools_leaffall(model_id)[0]])
            pool_chain = rwb.readbinarymat(cur_dir + output_dir + 'poolfile_' + pixel_chain[:-3]+'bin', [cbf_pixel['nodays']+1, get_nofluxes_nopools_leaffall(model_id)[1]])
            basic_plots.plot_flux_pool_timeseries(cbf_pixel, cbr_chain, flux_chain, pool_chain, get_nofluxes_nopools_leaffall(model_id)[2], savepath=cur_dir+plot_dir+'timeseries/', title=model_id+'_'+pixel_chain[:-3]+'png')

            flux_pixel = np.copy(flux_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((flux_pixel, flux_chain), axis=0)
            pool_pixel = np.copy(pool_chain) if pixel_chains.index(pixel_chain)==0 else np.concatenate((pool_pixel, pool_chain), axis=0)
            
            cbr_chain_list.append(cbr_chain)
            print(np.shape(cbr_chain))
            
        gr = gelman_rubin(cbr_chain_list)
        print(gr)
        print('%i of %i parameters converged' % (sum(gr<1.2), len(parnames)))
            
        basic_plots.plot_par_histograms(cbr_pixel, parnames=parnames, savepath=cur_dir+plot_dir+'dists/', title=model_id+'_'+pixel_chain[:-6]+'.png')    
        basic_plots.plot_flux_pool_timeseries(cbf_pixel, cbr_pixel, flux_pixel, pool_pixel, get_nofluxes_nopools_leaffall(model_id)[2], savepath=cur_dir+plot_dir+'timeseries/', title=model_id+'_'+pixel_chain[:-6]+'.png')
        
    #basic_plots.plot_map(nrows=46, ncols=73, land_pixel_list=[file[-8:-4] for file in glob.glob(cur_dir + cbf_dir + '*.cbf')], 
        #pixel_value_list=pixels, value_list=np.ones(len(pixels)), savepath=cur_dir+plot_dir+'maps/', title='test_pixels.png')
        
    return

if __name__=='__main__':
    main()
