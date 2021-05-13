
# AUTOMATES SHERLCOK SUBMISSION BATCH SCRIPTS

import glob
import os
import sys
from random import sample

def fill_in_sh(sh_file, array_size, n_hours, txt_file):
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#SBATCH --nodes=1\n')
    sh_file.write('#SBATCH -p konings,normal,owners\n')
    sh_file.write('#SBATCH -t %i:00:00\n' % n_hours)
    sh_file.write('#SBATCH --mail-type=END,FAIL\n')
    sh_file.write('#SBATCH --mail-user=cfamigli@stanford.edu\n')
    sh_file.write('#SBATCH --array=0-%i\n\n' % array_size)
    
    sh_file.write('# define the location of the command list file\n')
    sh_file.write('CMD_LIST=./%s\n\n' % txt_file)
    
    sh_file.write('# get the command list index from Slurm\n')
    sh_file.write('CMD_INDEX=$SLURM_ARRAY_TASK_ID\n\n')
    
    sh_file.write('# execute the command\n')
    sh_file.write('$(sed "${CMD_INDEX}q;d" "$CMD_LIST")\n')
    sh_file.close()
    return

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
    mdf_dir = '../../code/CARDAMOM_2.1.6c/C/projects/CARDAMOM_MDF/'
    runmodel_dir = '../../code/CARDAMOM_2.1.6c/C/projects/CARDAMOM_GENERAL/'
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
    cbr_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
    output_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
    
    n_iterations = '1000000' 
    n_chains = 2
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
        
    assim_txt_file = open('assimilation_list_' + model_id + '_' + run_type  + '_MCMC'+mcmc_id + '.txt', 'w')
    forward_txt_file = open('forward_list_' + model_id + '_' + run_type  + '_MCMC'+mcmc_id + '.txt', 'w')
    for cbf_file in cbf_files:
         for chain in range(1,n_chains+1):
             assim_txt_file.write('%sCARDAMOM_MDF.exe %s%s %s%s %s 0 %s 0.001 %s 1000\n' % (mdf_dir, cbf_dir, cbf_file, cbr_dir, cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+'_'+str(chain)+'.cbr', n_iterations, frac_save_out, mcmc_id))
             forward_txt_file.write('%sCARDAMOM_RUN_MODEL.exe %s%s %s%s %s%s %s%s %s%s %s%s\n' % (runmodel_dir, cbf_dir, cbf_file, cbr_dir, cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+'_'+str(chain)+'.cbr', 
                output_dir, 'fluxfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+'_'+str(chain)+'.bin', output_dir, 'poolfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+'_'+str(chain)+'.bin', 
                output_dir, 'edcdfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+'_'+str(chain)+'.bin', output_dir, 'probfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+'_'+str(chain)+'.bin'))
    assim_txt_file.close()
    forward_txt_file.close()
    
    assim_sh_file = open('assimilation_list_' + model_id + '_' + run_type + '_MCMC'+mcmc_id + '.sh', 'w')
    fill_in_sh(assim_sh_file, array_size=len(cbf_files)*n_chains, n_hours=18, txt_file='assimilation_list_' + model_id + '_' + run_type + '_MCMC'+mcmc_id + '.txt')
    
    forward_sh_file = open('forward_list_' + model_id + '_' + run_type  + '_MCMC'+mcmc_id + '.sh', 'w')
    fill_in_sh(forward_sh_file, array_size=len(cbf_files)*n_chains, n_hours=1, txt_file='forward_list_' + model_id + '_' + run_type  + '_MCMC'+mcmc_id + '.txt')
    
    return

if __name__=='__main__':
    main()