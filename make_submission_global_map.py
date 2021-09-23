
"""

created on Thu May 27 16:34:26 2021

@author: cfamigli

create submission txt and sh files for running a global map

"""

import glob
import os
import sys
import anova_utilities as autil

def main():
    model_id = sys.argv[1]
    run_type = sys.argv[2] # ALL or SUBSET
    mcmc_id = sys.argv[3] # 119 for normal, 3 for DEMCMC
    n_iter = sys.argv[4]
    var_to_plot = sys.argv[5]
    assim_type = '_longadapted'
    
    cur_dir = os.getcwd() + '/'
    cbr_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
    files = glob.glob(cur_dir+cbr_dir+'*MCMC'+mcmc_id+'_'+n_iter+'_*.cbr')
    pixels = list(set([file[-10:-6] for file in files]))
    
    txt_file_dir = cur_dir+'../'
    txt_filename = 'global_map_parallel_' + model_id + '_MCMC' + mcmc_id + '_' + n_iter + '_' + var_to_plot + '.txt'
    txt_file = open(txt_file_dir + txt_filename, 'w')
    for pixel in pixels:
        txt_file.write('python3 scripts/global_map_parallel.py %s %s %s %s %s %s\n' % (model_id, run_type, mcmc_id, n_iter, var_to_plot, pixel))
    txt_file.close()
    
    sh_file = open(txt_file_dir + txt_filename[:-3] + 'sh', 'w')
    autil.fill_in_sh(sh_file, array_size=len(pixels), n_hours=1, txt_file=txt_filename)
    
    return

if __name__=='__main__':
    main()