import numpy as np
from ovito.io import import_file
from sklearn.metrics import mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from gmm import get_NLL_perFrame, get_frmse_perFrame, get_frmse_perAtom,  get_NLL_perAtom, get_ferr_perAtom, load_GMM, PCA_setup
import matplotlib
from matplotlib.colors import LogNorm
matplotlib.use('Agg') # for running on cluster
import matplotlib.pyplot as plt

# This script is used to calculate the force rmse and NLL for the test set after you obtain a fitted GMM model#

## Specify the data set and the GMM model ##
STORE_PATH='GMM_275/feature_npz'
DATA_SET='test'
PCA_COMP=8 
PCA_FIT='perSet' # 'perSet' or 'all'
SUB_TRAIN_SIZE=50
GMM_DIR=f'PCA_perSet_M={SUB_TRAIN_SIZE}gmm' 
#'PCA_perSet_M=50gmm' # 'PCA_sub_train+test_M=600gmm'
GMM_NAME=f'{GMM_DIR}/gmm_n=100'
dft_file='DFT/test.xyz'
choice_lst=['perAtom'] # 'perFrame' or 'perAtom'
metric_lst=['max', 'sum'] # 'max' or 'sum'
display_num_atoms=True # 64, 128 belongs to HTP, 144 belongs to LTP

## load GMM model ##
loaded_gmm = load_GMM(GMM_NAME)

## read training & testing data ##
train_path=f'{STORE_PATH}/{DATA_SET}_train.npz'
Train=np.load(train_path, allow_pickle=True)
train_features_fr = Train['features_fr']
test_path=f'{STORE_PATH}/{DATA_SET}_test.npz'
Test=np.load(test_path, allow_pickle=True)
test_features_fr = Test['features_fr']
test_edge_index_fr = Test['edge_index_fr']
test_forces_fr = Test['forces_fr']
test_part_type_fr = Test['part_type_fr']

## reduce feature dimension for training & testing data ##
pca = PCA_setup(PCA_COMP, PCA_FIT, train_features_fr, test_features_fr, SUB_TRAIN_SIZE)

for metric in metric_lst:
    for choice in choice_lst:
        ## force rmse ##
        if choice == 'perFrame':
            f_rmse, num_atoms = get_frmse_perFrame(dft_file, test_forces_fr, return_num_atoms=display_num_atoms)
            test_NLL = get_NLL_perFrame(test_features_fr, loaded_gmm, num_atoms=num_atoms, pca=pca, metric=metric, reduce_dim=PCA_COMP)
        elif choice == 'perAtom':
            #f_rmse = get_ferr_perAtom(dft_file, test_forces_fr)
            f_rmse, num_atoms = get_frmse_perAtom(dft_file, test_forces_fr, return_num_atoms=display_num_atoms)
            test_NLL = get_NLL_perAtom(test_features_fr, test_edge_index_fr, loaded_gmm, pca=pca, metric=metric, reduce_dim=PCA_COMP)
        else:
            raise ValueError('choice must be either perFrame or perAtom')
        print('---plotting metric = ', metric, 'choice = ', choice, '---')
        ## plot f_rmse vs NLL ##
        plt.figure(figsize=(8, 6))
        plt.rcParams.update({'font.size': 16})
        if display_num_atoms:
            for num in set(num_atoms):
                plt.scatter(f_rmse[num_atoms==num], test_NLL[num_atoms==num], alpha=0.5, label=num)
            plt.legend()
        else:
            plt.scatter(f_rmse, test_NLL, alpha=0.5)
        plt.title(f'{choice} estimation, metric = {metric}')
        # plt.hist2d(f_rmse, test_NLL, bins=100, cmap='Blues')
        # plt.colorbar()
        plt.xlabel('f_rmse [meV/Å]')
        # plt.xlabel('f_err_percent [%]')
        plt.ylabel('Negative Log-Likelihood')
        # plt.ylim(-40, 100)
        plt.savefig(f'{GMM_DIR}/{choice}_{metric}.png')
        plt.close()

    ## plot f_rmse by atom type ##
    ## must use perAtom ##
    print('---plotting by atom type, metric = ', metric, '---')
    f_rmse = get_frmse_perAtom(dft_file, test_forces_fr)
    test_NLL = get_NLL_perAtom(test_features_fr, test_edge_index_fr, loaded_gmm, pca=pca, metric=metric, reduce_dim=PCA_COMP)
    part_type = np.concatenate(test_part_type_fr, axis=0)
    dict_elem = {1:'Cs', 2:'P', 3:'H', 4:'O'}
    for element in [1,2,3,4]:
        plt.figure(figsize=(8, 6))
        plt.rcParams.update({'font.size': 16})
        #plt.scatter(f_rmse[part_type==element], test_NLL[part_type==element], alpha=0.5, label=element)
        plt.hist2d(f_rmse[part_type==element], test_NLL[part_type==element], bins=60, cmap='viridis', norm=LogNorm())
        plt.colorbar()
        plt.title(f'metric = {metric}, {dict_elem[element]}')
        plt.xlabel('f_rmse [meV/Å]')
        plt.ylabel('Negative Log-Likelihood')
        # plt.ylim(18, 180)
        # plt.xlim(0, 300)
        plt.savefig(f'{GMM_DIR}/metric_{metric}_type{element}.png')
        plt.close()