

"""

created on Fri May 8 09:21:42 2021

@author: cfamigli

make sherlock submission scripts for nonconvergent pixels after checking GR

"""

import glob
import os
import sys
import numpy as np
from random import sample
import anova_utilities as autil
import readwritebinary as rwb
import itertools
from pandas import DataFrame, read_pickle

def select_cbf_files(all_filenames, pixel_list):
    reduced_list = []
    for pixel in pixel_list:
        reduced_list.append([el for el in all_filenames if pixel+'.cbf' in el][0])
    return reduced_list

def main():
    model_id = sys.argv[1]
    run_type = sys.argv[2] # ALL or SUBSET
    mcmc_id = sys.argv[3] # 119 for normal, 3 for DEMCMC
    nbe_optimization = sys.argv[4] # 'OFF' or 'ON'
    assim_type = '_p25adapted'
    
    cur_dir = os.getcwd() + '/'
    mdf_dir = '../code/CARDAMOM_2.1.6c/C/projects/CARDAMOM_MDF/' if nbe_optimization=='OFF' else '../code/CARDAMOM_Uma_2.1.6c-master/C/projects/CARDAMOM_MDF/'
    runmodel_dir = '../code/CARDAMOM_2.1.6c/C/projects/CARDAMOM_GENERAL/' if nbe_optimization=='OFF' else '../code/CARDAMOM_Uma_2.1.6c-master/C/projects/CARDAMOM_GENERAL/'
    cbf_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf'+assim_type+'/' + model_id + '/'
    cbr_dir = '../../../../../scratch/users/cfamigli/cardamom/files/cbr'+assim_type+'/' + model_id + '/'
    output_dir = '../../../../../scratch/users/cfamigli/cardamom/files/output'+assim_type+'/' + model_id + '/'
    plot_dir = '../../../../../../scratch/users/cfamigli/cardamom/plots/'
    parnames = autil.get_parnames('../../misc/', model_id)
    
    n_iterations = sys.argv[5]
    runtime_assim = int(sys.argv[6])
    resubmit_num = sys.argv[7]
    n_chains_resubmit = 4
    ens_size = 500
        
    if mcmc_id=='119':
        frac_save_out = str(int(int(n_iterations)/500))
    elif mcmc_id=='3':
        frac_save_out = str(int(int(n_iterations)/500*100)) # n_iterations/ frac_save_out * 100 will be ensemble size
    
    # select which pixels to submit
    os.chdir(cbf_dir)
    if run_type=='ALL':
        cbf_files = glob.glob('*.cbf')
    elif run_type=='SUBSET_RANDOM':
        cbf_files = sample(glob.glob('*.cbf'), 10)
    elif run_type=='SUBSET_INPUT':
        cbf_files = select_cbf_files(glob.glob('*.cbf'), ['3809','3524','2224','4170','1945','3813','4054','3264','1271','3457'])
    os.chdir(cur_dir + '/../')
    
    cbf_files.sort()
       
    # create one combined submission file with all assimilation and forward commands for each pixel's chain on one line

    txt_filename = 'combined_assim_forward_list_' + model_id + '_' + run_type  + assim_type+ '_MCMC'+mcmc_id + '_'+n_iterations + '_resubmit'+resubmit_num+'.txt'
    txt_file = open(txt_filename, 'w')
    
    resubmit_count = 0
    gr_pixels = np.zeros(len(cbf_files))*np.nan # list of GR for each pixel, for mapping
    pixels = []
    best_subset = []
    conv_bool_lst = []
    for cbf_file in cbf_files:
        best_subset_pixel = [] 
        resubmit = False
        print(cbf_file, cbf_files.index(cbf_file))
        
        cbf_pixel = rwb.read_cbf_file(cur_dir + cbf_dir + cbf_file)
        pixel = cbf_file[-8:-4]
        
        cbr_files = glob.glob(cur_dir + '../' + cbr_dir + '*MCMC'+mcmc_id+'_'+n_iterations+'_' + pixel + '_*.cbr')
        cbr_files = sorted(cbr_files, key=lambda x: int(x.partition(pixel+'_')[-1].partition('.cbr')[0]))
        
        if len(cbr_files)>=n_chains_resubmit: pixels.append(pixel)
        #cbr_files = cbr_files[:16] ############ TEMP
        
        if len(cbr_files)>0:
            end_chain = int(cbr_files[-1].partition(pixel+'_')[-1].partition('.cbr')[0]) 
            #print('ENDCHAIN: '+str(end_chain))
        else:
            end_chain = 0
            resubmit = True
        
        # get all possible XX member combinations of cbr files
        n_chains_to_converge = n_chains_resubmit
        cbr_files_all_subsets = [list(i) for i in itertools.combinations(cbr_files, n_chains_to_converge)]
        continue_check = True
        for subset in cbr_files_all_subsets:
            if continue_check:
                
                # read parameters and compute gelman rubin
                cbr_chain_list = []
                chain_nums = ['0']
                
                for cbr_file in subset:
                    #print(cbr_file[-10:-4])
                    cbr_chain = rwb.read_cbr_file(cbr_file, {'nopars': len(parnames)})
                    cbr_chain = autil.modulus_Bday_Fday(cbr_chain, parnames)
                    chain_nums.append(cbr_file.partition('.cbr')[0].partition(pixel+'_')[-1]) # append chain number
                    
                    if np.shape(cbr_chain)[0]==ens_size:
                        cbr_chain_list.append(cbr_chain)
                        #print(np.shape(cbr_chain))
                    else:
                        print('incorrect ensemble size)')
                        resubmit = True
                    
                if len(cbr_chain_list)>1:
                    gr = autil.gelman_rubin(cbr_chain_list)
                    #print(gr)
                    print('%i/%i' % (sum(gr<1.2), len(parnames))) #print('%i of %i parameters converged' % (sum(gr<1.2), len(parnames)))
                    
                    if (np.isnan(gr_pixels[cbf_files.index(cbf_file)])):
                        gr_pixels[cbf_files.index(cbf_file)] = sum(gr<1.2)/len(parnames)
                        #if len(cbr_files_all_subsets)==1: best_subset_pixel.append(chain_nums)
                        
                    
                    if sum(gr<1.2)/len(parnames)<0.9:
                        #print('gr too low')
                        resubmit = True
                        
                        if (sum(gr<1.2)/len(parnames) >= gr_pixels[cbf_files.index(cbf_file)]):
                            gr_pixels[cbf_files.index(cbf_file)] = sum(gr<1.2)/len(parnames)
                            best_subset_pixel.append(chain_nums)
                            conv_bool = 0
                        
                    else:
                        resubmit = False
                        continue_check = False
                        gr_pixels[cbf_files.index(cbf_file)] = sum(gr<1.2)/len(parnames)
                        best_subset_pixel.append(chain_nums)
                        conv_bool = 1
                        
                else:
                    gr = np.nan
                    print('gr undefined')
                    best_subset_pixel.append(chain_nums)
                    conv_bool = 0
                    resubmit = True
                    
        
        if len(best_subset_pixel)>0: 
            best_subset.append(best_subset_pixel[-1])
            conv_bool_lst.append(conv_bool)
        
        # write into text file if pixel needs to be resubmitted
        if resubmit:
            first_resubmit_chain = end_chain+1
            last_resubmit_chain = end_chain+n_chains_resubmit
            for chain in range(first_resubmit_chain, last_resubmit_chain+1):
                c = '_'+str(chain)
                txt_file.write('%sCARDAMOM_MDF.exe %s%s %s%s %s 0 %s 0.001 %s 1000' % (mdf_dir, cbf_dir[3:], cbf_file, cbr_dir, cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+ c +'.cbr', n_iterations, frac_save_out, mcmc_id))
                txt_file.write(' && %sCARDAMOM_RUN_MODEL.exe %s%s %s%s %s%s %s%s %s%s %s%s' % (runmodel_dir, cbf_dir[3:], cbf_file, cbr_dir, cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+ c +'.cbr', 
                    output_dir, 'fluxfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+ c +'.bin', output_dir, 'poolfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+ c +'.bin', 
                    output_dir, 'edcdfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+ c +'.bin', output_dir, 'probfile_'+cbf_file[:-8]+'MCMC'+mcmc_id+'_'+n_iterations+'_'+cbf_file[-8:-4]+ c +'.bin'))
                txt_file.write(' && ') if chain<last_resubmit_chain else txt_file.write('\n')
            resubmit_count += 1
                
    txt_file.close()
    
    sh_file = open(txt_filename[:-3] + 'sh', 'w')
    autil.fill_in_sh(sh_file, array_size=resubmit_count, n_hours=runtime_assim, txt_file=txt_filename, combined=True)
    
    autil.plot_map(nrows=46, ncols=73, land_pixel_list=pixels, 
        pixel_value_list=pixels, value_list=gr_pixels*100, savepath=cur_dir+plot_dir+'maps/', savename='gr_' + model_id + assim_type+ '_' +run_type+ '_MCMC' + mcmc_id + '_' + n_iterations + '_resubmit'+resubmit_num)

    #print(pixels, best_subset, conv_bool_lst)
    print(len(pixels), len(best_subset), len(conv_bool_lst))
    DataFrame(list(zip(pixels, best_subset, conv_bool_lst))).to_pickle(cur_dir + '../' + cbr_dir + model_id + assim_type+ '_' +run_type+ '_MCMC' + mcmc_id + '_' + n_iterations + '_best_subset.pkl')
    
    return

if __name__=='__main__':
    main()
