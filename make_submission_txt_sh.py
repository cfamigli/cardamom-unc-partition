
# AUTOMATES SHERLCOK SUBMISSION BATCH SCRIPTS

import glob
import os
import sys
from random import sample
import anova_utilities as autil

def select_cbf_files(all_filenames, pixel_list):
    reduced_list = []
    for pixel in pixel_list:
        reduced_list.append([el for el in all_filenames if pixel+'.cbf' in el][0])
    return reduced_list

def main():
    model_id = sys.argv[1]
    run_type = sys.argv[2] # ALL or SUBSET
    mcmc_id = sys.argv[3] # 119 for normal, 3 for DEMCMC
    assim_type = '_longadapted'
    
    cur_dir = os.getcwd()
    mdf_dir = '../code/CARDAMOM_2.1.6c/C/projects/CARDAMOM_MDF/'
    runmodel_dir = '../code/CARDAMOM_2.1.6c/C/projects/CARDAMOM_GENERAL/'
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
    cbr_dir = '../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
    output_dir = '../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
    
    n_iterations = sys.argv[4]
    runtime_assim = int(sys.argv[5])
    n_chains = int(sys.argv[6])
    chain_num = '_' + sys.argv[7] if n_chains==1 else ''
        
    if mcmc_id=='119':
        frac_save_out = str(int(int(n_iterations)/500))
    elif mcmc_id=='3':
        frac_save_out = str(int(int(n_iterations)/500*100)) # n_iterations/ frac_save_out * 100 will be ensemble size
    
    os.chdir(cbf_dir)
    if run_type=='ALL':
        cbf_files = glob.glob('*.cbf')
    elif run_type=='SUBSET_RANDOM':
        cbf_files = sample(glob.glob('*.cbf'), 10)
    elif run_type=='SUBSET_INPUT':
        cbf_files = select_cbf_files(glob.glob('*.cbf'), ['3809','3524','2224','4170','1945','3813','4054','3264','1271','3457'])
    os.chdir(cur_dir + '/../')
        
    assim_txt_filename = 'assimilation_list_' + model_id + '_' + run_type  + '_MCMC'+mcmc_id + '_'+n_iterations + chain_num+ '.txt'
    assim_txt_file = open(assim_txt_filename, 'w')
    
    forward_txt_filename = 'forward_list_' + model_id + '_' + run_type  + '_MCMC'+mcmc_id + '_'+n_iterations + chain_num+ '.txt'
    forward_txt_file = open(forward_txt_filename, 'w')
    for cbf_file in cbf_files:
         for chain in range(1,n_chains+1):
             c = chain_num if n_chains==1 else '_'+str(chain)
             assim_txt_file.write('%sCARDAMOM_MDF.exe %s%s %s%s %s 0 %s 0.001 %s 1000\n' % (mdf_dir, cbf_dir[3:], cbf_file, cbr_dir, cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+ c +'.cbr', n_iterations, frac_save_out, mcmc_id))
             forward_txt_file.write('%sCARDAMOM_RUN_MODEL.exe %s%s %s%s %s%s %s%s %s%s %s%s\n' % (runmodel_dir, cbf_dir[3:], cbf_file, cbr_dir, cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+ c +'.cbr', 
                output_dir, 'fluxfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+ c +'.bin', output_dir, 'poolfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+ c +'.bin', 
                output_dir, 'edcdfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+ c +'.bin', output_dir, 'probfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+ c +'.bin'))
    assim_txt_file.close()
    forward_txt_file.close()
    
    assim_sh_file = open(assim_txt_filename[:-3] + 'sh', 'w')
    autil.fill_in_sh(assim_sh_file, array_size=len(cbf_files)*n_chains, n_hours=runtime_assim, txt_file=assim_txt_filename)
    
    forward_sh_file = open(forward_txt_filename[:-3] + 'sh', 'w')
    autil.fill_in_sh(forward_sh_file, array_size=len(cbf_files)*n_chains, n_hours=1, txt_file=forward_txt_filename)
    
    return

if __name__=='__main__':
    main()