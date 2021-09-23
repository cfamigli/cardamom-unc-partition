
"""

created on Tues Jul 20 08:51:02 2021

@author: cfamigli

set up environmental filtering models to predict CARDAMOM parameters

"""

import numpy as np
import readwritebinary as rwb
import os
import glob
import sys
import anova_utilities as autil

def select_cbf_files(all_filenames, pixel_list):
    reduced_list = []
    for pixel in pixel_list:
        reduced_list.append([el for el in all_filenames if pixel+'.cbf' in el][0])
    return reduced_list

def main():
    
    # set run information to read
    model_id = sys.argv[1]
    run_type = sys.argv[2] # ALL OR SUBSET
    mcmc_id = sys.argv[3] # 119 for normal, 3 for DEMCMC
    n_iter = sys.argv[4]
    nbe_optimization = sys.argv[5] # OFF OR ON
    runtime_assim = int(sys.argv[6])
    ens_size = 500
    assim_type = '_p25adapted'
    
    # set directories
    cur_dir = os.getcwd() + '/'
    mdf_dir = '../code/CARDAMOM_2.1.6c/C/projects/CARDAMOM_MDF/' if nbe_optimization=='OFF' else '../code/CARDAMOM_Uma_2.1.6c-master/C/projects/CARDAMOM_MDF/'
    runmodel_dir = '../code/CARDAMOM_2.1.6c/C/projects/CARDAMOM_GENERAL/' if nbe_optimization=='OFF' else '../code/CARDAMOM_Uma_2.1.6c-master/C/projects/CARDAMOM_GENERAL/'
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
    cbf_ic_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/ic_test/' + model_id + '/'
    cbr_pft_dir = '../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'_pft/' + model_id + '/'
    cbr_ic_dir = '../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'_pft/ic_test/' + model_id + '/'
    output_ic_dir = '../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'_pft/ic_test/' + model_id + '/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    
    # get model specific information
    parnames = autil.get_parnames('../../misc/', model_id)
    ic_inds = autil.get_inds_ic(model_id) # get indices of initial condition parameters
    
    if mcmc_id=='119':
        frac_save_out = str(int(int(n_iter)/500))
    elif mcmc_id=='3':
        frac_save_out = str(int(int(n_iter)/500*100)) # n_iterations/ frac_save_out * 100 will be ensemble size
    
    # select which pixels to submit
    os.chdir(cbf_dir)
    if run_type=='ALL':
        cbf_files = glob.glob('*.cbf')
    elif run_type=='SUBSET_INPUT':
        cbf_files = select_cbf_files(glob.glob('*.cbf'), ['3809','3524','2224','4170','1945','3813','4054','3264','1271','3457'])
    os.chdir(cur_dir + '/../')
    
    cbf_files.sort()
    
    ############################################################################################################################################
    
    # run through pixel cbfs
    for cbf_file in cbf_files:
        
        pixel = cbf_file[-8:-4]
        cbf_data = rwb.read_cbf_file(cbf_dir + cbf_file)
        
        # get list of pft cbrs for pixel
        cbr_files = glob.glob(cbr_pft_dir + '*' + pixel + '*.cbr')
        
        for cbr_file in cbr_files:
            
            cbr_data = rwb.read_cbr_file(cbr_file, {'nopars': len(parnames)})
            parpriors = np.concatenate((np.nanmedian(cbr_data, axis=0), np.ones(50-len(parnames))*-9999.))
            parpriorunc = np.concatenate((np.ones(len(parnames))*1.001, np.ones(50-len(parnames))*-9999.))
            
            parpriors[ic_inds[0]:ic_inds[1]] = -9999.
            parpriorunc[ic_inds[0]:ic_inds[1]] = -9999.
            
            cbf_data['PARPRIORS'] = parpriors.reshape(-1,1)
            cbf_data['PARPRIORUNC'] = parpriorunc.reshape(-1,1)
            
            #rwb.CARDAMOM_WRITE_BINARY_FILEFORMAT(cbf_data, cbf_ic_dir + cbr_file.partition(cbr_pft_dir)[-1].partition('cbr')[0]+'cbf')
            
    ############################################################################################################################################
    
    txt_filename = 'combined_assim_forward_list_' + model_id + '_' + run_type  + assim_type+ '_MCMC'+mcmc_id + '_'+n_iter + '_ic_test.txt'
    txt_file = open(txt_filename, 'w')
    
    for cbf_ic_file in glob.glob(cbf_ic_dir + '*.cbf'):
        f = cbf_ic_file.partition(cbf_ic_dir)[-1]
        txt_file.write('%sCARDAMOM_MDF.exe %s%s %s%s %s 0 %s 0.001 %s 1000' % (mdf_dir, cbf_ic_dir[3:], f, cbr_ic_dir, f[:-4] + '.cbr', n_iter, frac_save_out, mcmc_id))
        txt_file.write(' && %sCARDAMOM_RUN_MODEL.exe %s%s %s%s %s%s %s%s %s%s %s%s' % (runmodel_dir, cbf_ic_dir[3:], f, cbr_ic_dir, f[:-4] + '.cbr', 
            output_ic_dir, 'fluxfile_'+ f[:-4] +'.bin', output_ic_dir, 'poolfile_'+ f[:-4] +'.bin', 
            output_ic_dir, 'edcdfile_'+ f[:-4] +'.bin', output_ic_dir, 'probfile_'+ f[:-4] +'.bin'))
        txt_file.write('\n')
                
    txt_file.close()
    
    sh_file = open(txt_filename[:-3] + 'sh', 'w')
    autil.fill_in_sh(sh_file, array_size=len(glob.glob(cbf_ic_dir + '*.cbf')), n_hours=runtime_assim, txt_file=txt_filename, combined=True)
    
    return

if __name__=='__main__':
    main()