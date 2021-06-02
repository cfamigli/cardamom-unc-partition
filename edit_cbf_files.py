
# TEST SCRIPT for editing cbfs

import readwritebinary as rwb
import numpy as np
import anova_utilities as autil
import matplotlib
from matplotlib import pyplot as plt
from pandas import read_csv
import glob
import os
import sys
import copy

def edit_model_id(infile, outfile, ID):
    # read cbf file using readwritebinary and edit ID flag
    cbf = rwb.read_cbf_file(infile)
    cbf['ID'] = ID
    rwb.CARDAMOM_WRITE_BINARY_FILEFORMAT(cbf, outfile)
    return

def edit_model_met(infile, outfile):
    # read cbf file using readwritebinary and edit ID flag
    cbf = rwb.read_cbf_file(infile)
    cbf['MET'] = cbf['MET'][:,:-2]
    rwb.CARDAMOM_WRITE_BINARY_FILEFORMAT(cbf, outfile)
    return

def edit_model_met_shape(infile, outfile, n_met_to_drop):
    # remove columns from met for certain models
    cbf = rwb.read_cbf_file(infile)
    cbf['MET'] = cbf['MET'][:,:(n_met_to_drop*-1)]
    rwb.CARDAMOM_WRITE_BINARY_FILEFORMAT(cbf, outfile)
    return

def edit_model_fire(infile, outfile):
    # remove columns from met for certain models
    cbf = rwb.read_cbf_file(infile)
    print(cbf['MET'].shape)
    cbf['MET'][:,6] = -9999
    cbf['OTHER_OBS']['MFire']['mean'] = -9999
    cbf['OTHER_OBS']['MFire']['unc'] = -9999
    rwb.CARDAMOM_WRITE_BINARY_FILEFORMAT(cbf, outfile)
    return

def shift_model_parpriors(infile, outfile, inds_out):
    # read cbf file using readwritebinary and edit parpriors, parpriorunc
    # inds is a list
    cbf = rwb.read_cbf_file(infile)
    
    count_real = 0
    for parprior, parpriorunc in zip(cbf['PARPRIORS'], cbf['PARPRIORUNC']):
        if parprior==-9999:
            continue
        else:
            ind_in = np.where(cbf['PARPRIORS']==parprior)[0][0]
            cbf['PARPRIORS'][inds_out[count_real]] = parprior
            cbf['PARPRIORUNC'][inds_out[count_real]] = parpriorunc
            if ind_in!=inds_out[count_real]:
                cbf['PARPRIORS'][ind_in] = -9999
                cbf['PARPRIORUNC'][ind_in] = -9999
            count_real += 1
    rwb.CARDAMOM_WRITE_BINARY_FILEFORMAT(cbf, outfile)
    return

def delete_model_parpriors(infile, outfile, ind_out):
    # read cbf file using readwritebinary and edit parpriors, parpriorunc
    # inds is a list
    cbf = rwb.read_cbf_file(infile)
    for ind in ind_out:
        cbf['PARPRIORS'][ind_out] = -9999
        cbf['PARPRIORUNC'][ind_out] = -9999
    rwb.CARDAMOM_WRITE_BINARY_FILEFORMAT(cbf, outfile)
    return

def edit_101(infile, outfile, ID, inds_shift, inds_del):
    cbf = rwb.read_cbf_file(infile)
    cbf['ID'] = ID
    cbf['MET'] = cbf['MET'][:,:-2]
    '''cbf['MET'][:,6] = 0
    cbf['OTHER_OBS']['MFire']['mean'] = -9999
    cbf['OTHER_OBS']['MFire']['unc'] = -9999'''
    count_real = 0
    for parprior, parpriorunc in zip(cbf['PARPRIORS'], cbf['PARPRIORUNC']):
        if parprior==-9999:
            continue
        else:
            ind_in = np.where(cbf['PARPRIORS']==parprior)[0][0]
            cbf['PARPRIORS'][inds_shift[count_real]] = parprior
            cbf['PARPRIORUNC'][inds_shift[count_real]] = parpriorunc
            if ind_in!=inds_shift[count_real]:
                cbf['PARPRIORS'][ind_in] = -9999
                cbf['PARPRIORUNC'][ind_in] = -9999
            count_real += 1
            
    for ind in inds_del:
        cbf['PARPRIORS'][ind] = -9999
        cbf['PARPRIORUNC'][ind] = -9999
        
    rwb.CARDAMOM_WRITE_BINARY_FILEFORMAT(cbf, outfile)
    return

def retrieve_nbe_unc(unc_mat, rowcol_mat, rowcol_pix):
    return unc_mat[rowcol_mat==rowcol_pix]

def main():
    model_id_start = sys.argv[1]
    model_id_target = sys.argv[2]
    
    cur_dir = os.getcwd() + '/'
    infile_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf_longruns/cbf/'
    compare_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf/'
    
    mod_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf_longadapted_wrongobsunc/'
    outfile_dir = '../../../../../../scratch/users/cfamigli/cardamom/files/cbf_longadapted/'
    
    os.chdir(infile_dir)
    files = glob.glob('*.cbf')
    
    for file in files:
        print(file)
        orig = rwb.read_cbf_file(glob.glob(cur_dir+infile_dir+'/*'+file[-8:-4]+'*.cbf')[0])
        obsunc = orig['OBSUNC']
        
        cbf_longadapted = rwb.read_cbf_file(glob.glob(cur_dir+mod_dir+model_id_target+'/*'+file[-8:-4]+'*.cbf')[0])
        cbf_longadapted['OBSUNC'] = obsunc.copy()
        
        rwb.CARDAMOM_WRITE_BINARY_FILEFORMAT(cbf_longadapted, cur_dir+outfile_dir+model_id_target+'/'+file)
        testout = rwb.read_cbf_file(glob.glob(cur_dir+outfile_dir+model_id_target+'/*'+file[-8:-4]+'*.cbf')[0])
        assert testout['OBSUNC']['NBE']['annual_unc']==0.02
    
    
    
    #for file in files[:1]:
        #print(rwb.read_cbf_file(glob.glob(cur_dir+infile_dir+'/*'+file[-8:-4]+'*.cbf')[0])) 
        #print(rwb.read_cbf_file(glob.glob(cur_dir+outfile_dir+model_id_target+'/*'+file[-8:-4]+'*.cbf')[0])['MET'].shape) 
        '''print(file)
        cbf_long = rwb.read_cbf_file(file)
        
        cbf_compare = rwb.read_cbf_file(glob.glob(cur_dir+compare_dir+model_id_start+'/*'+file[-8:-4]+'*.cbf')[0])
        doy_start = cbf_compare['MET'][:,0][0]
        ind_doy_start_long = np.where(cbf_long['MET'][:,0]==doy_start)[0][0]
        
        cbf_out = copy.deepcopy(cbf_long)
        
        cbf_out['ID'] = model_id_target
        cbf_out['nodays'] = len(cbf_long['MET'][ind_doy_start_long:,0])
        
        if int(model_id_target)>400:
            cbf_out['MET'] = cbf_out['MET'][ind_doy_start_long:,:]
        else:
            cbf_out['MET'] = cbf_out['MET'][ind_doy_start_long:,:7]
            
        cbf_out['nomet'] = cbf_out['MET'].shape[1]
        
        for el in cbf_out['OBS']:
            if len(cbf_out['OBS'][el])>0:
                cbf_out['OBS'][el] = cbf_out['OBS'][el][ind_doy_start_long:]
        
        parprior = cbf_out['PARPRIORS']
        parpriorunc = cbf_out['PARPRIORUNC']

        parpriorout = np.ones(parprior.shape)*-9999.
        parprioruncout = np.ones(parpriorunc.shape)*-9999.
        
        if int(model_id_target)>102:
            if (int(model_id_target)==900) | (int(model_id_target)==901):
                parpriorout[1] = parprior[1]
                parpriorout[10] = 17.5
                
                parprioruncout[1] = parpriorunc[1]
                parprioruncout[10] = 1.5
                
            else:
                parpriorout[1] = parprior[1]
                parpriorout[10] = 17.5
                parpriorout[11] = parprior[10]
                parpriorout[14] = parprior[13]
                
                parprioruncout[1] = parpriorunc[1]
                parprioruncout[10] = 1.5
                parprioruncout[11] = parpriorunc[10]
                parprioruncout[14] = parpriorunc[13]
        else:
            parpriorout[1] = parprior[1]
            parpriorout[4] = 17.5
            parpriorout[5] = parprior[10]
            parpriorout[8] = parprior[13]
            
            parprioruncout[1] = parpriorunc[1]
            parprioruncout[4] = 1.5
            parprioruncout[5] = parpriorunc[10]
            parprioruncout[8] = parpriorunc[13]
            
        cbf_out['PARPRIORS'] = parpriorout
        cbf_out['PARPRIORUNC'] = parprioruncout'''
        
        #print(cbf_out['OBSUNC']['NBE'])
        
        #rwb.CARDAMOM_WRITE_BINARY_FILEFORMAT(cbf_out, cur_dir+outfile_dir+model_id_target+'/'+file)
        

    '''model_id_target = ['811','400','831','1003','1000','1010','101','102']
    for file in files[:5]:
        print(file[-8:-4])
        for model in model_id_target:
            if (model=='400'):# | (model==400) | (model=='102'):
                #print(rwb.read_cbf_file(cur_dir+outfile_dir+model+'/'+file)['PARPRIORS'][5])
                #edit_101(file, cur_dir+outfile_dir+model+'/'+file, '101', [1, 4, 5, 8], [5, 8])
                #shift_model_parpriors(file, cur_dir+outfile_dir+model+'/'+file, [1, 4, 5, 8])
                #edit_model_id(cur_dir+outfile_dir+model+'/'+file, cur_dir+outfile_dir+model+'/'+file, model)
                #edit_model_met_shape(cur_dir+outfile_dir+model+'/'+file, cur_dir+outfile_dir+model+'/'+file, 2)
                #delete_model_parpriors(cur_dir+outfile_dir+model+'/'+file, cur_dir+outfile_dir+'102'+'/'+file, [5,8])
                #edit_model_fire(cur_dir+outfile_dir+model+'/'+file, cur_dir+outfile_dir+'102'+'/'+file)
                #print(cur_dir+outfile_dir+model+'/'+file)
                print(rwb.read_cbf_file(cur_dir+outfile_dir+model+'/'+file)['ID'])
                print(rwb.read_cbf_file(cur_dir+outfile_dir+model+'/'+file)['MET'].shape)
                print(rwb.read_cbf_file(cur_dir+outfile_dir+model+'/'+file)['OTHER_OBS']['MFire']['mean'])
                #print(rwb.read_cbf_file(cur_dir+outfile_dir+model+'/'+file)['PARPRIORS'])'''

    '''for file in files[:1]:
        #print(rwb.read_cbf_file(glob.glob(cur_dir+outfile_dir_2+model_id_start+'/*'+file[-8:-4]+'*.cbf')[0]))
        print(file)
        cbf_long = rwb.read_cbf_file(file)
        print(cbf_long)
        #print(cbf_long)
        abgb_long = cbf_long['OBS']['ABGB']
        #som_long = cbf_long['OBS']['SOM']
        nbe_long = cbf_long['OBS']['NBE']
        doy_long = cbf_long['MET'][:,0]
        
        cbf_compare = rwb.read_cbf_file(glob.glob(cur_dir+outfile_dir+model_id_start+'/*'+file[-8:-4]+'*.cbf')[0])
        print(cbf_compare)
        #print(cbf_compare)
        abgb_compare = cbf_compare['OBS']['ABGB']
        #som_compare = cbf_compare['OBS']['SOM']
        nbe_compare = cbf_compare['OBS']['NBE']
        doy_compare = cbf_compare['MET'][:,0]
        
        doy_start = doy_compare[0]
        doy_end = doy_long[-1]
        
        ind_doy_start_long = np.where(doy_long==doy_start)[0][0]
        ind_doy_end_long = np.where(doy_long==doy_end)[0][0]
        
        abgb_sub = np.ones(len(doy_compare))*-9999.
        abgb_sub[:len(abgb_long[ind_doy_start_long:ind_doy_end_long+1])] = abgb_long[ind_doy_start_long:ind_doy_end_long+1]

        #som_sub = np.ones(len(doy_compare))*-9999.
        #som_sub[:len(som_long[ind_doy_start_long:ind_doy_end_long+1])] = som_long[ind_doy_start_long:ind_doy_end_long+1]
        
        nbe_sub = np.ones(len(doy_compare))*-9999.
        nbe_sub[:len(nbe_long[ind_doy_start_long:ind_doy_end_long+1])] = nbe_long[ind_doy_start_long:ind_doy_end_long+1]
        
        cbf_compare['OBS']['ABGB'] = abgb_sub
        #cbf_compare['OBS']['SOM'] = som_sub
        cbf_compare['OBS']['NBE'] = nbe_sub
        
        
        
        assert 'oak' not in cur_dir+outfile_dir_2+model_id_start+'/'+file
        #rwb.CARDAMOM_WRITE_BINARY_FILEFORMAT(cbf_compare, cur_dir+outfile_dir_2+model_id_start+'/'+file)'''
        
        
    '''for file in files[:5]:
        cbf = rwb.read_cbf_file(file)
        cbf_compare = rwb.read_cbf_file(glob.glob(cur_dir+outfile_dir+model_id_start+'/*'+file[-8:-4]+'*.cbf')[0])
        
        plt.figure()
        nbe_org = cbf['OBS']['NBE']
        nbe_org[nbe_org==-9999] = float('nan')
        plt.plot(nbe_org)
        
        nbe_cmp = cbf_compare['OBS']['NBE']
        nbe_cmp[nbe_cmp==-9999] = float('nan')
        plt.plot(nbe_cmp)
        plt.savefig('test'+file[-8:-4]+'.png')'''
    
    '''pixel_list = [file[-8:-4] for file in files]
    
    lat, lon, arr = set_tile(nrows=46, ncols=73)
    
    for pixel in pixel_list:
        (latpx,lonpx) = rwb.rowcol_to_latlon(pixel)
        row = np.where((lat==latpx) & (lon==lonpx))[0]
        col = np.where((lat==latpx) & (lon==lonpx))[1]
        arr[row,col] = 1.
    
    basic_plots.plot_map(np.flipud(arr), savepath='../cardamom_plots/maps/', title='test')'''
        
    return

if __name__=='__main__':
    main()
