import numpy as np
from ovito.io import import_file
import matplotlib
matplotlib.use('Agg') # for running on cluster
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from _reader import DataReader
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import sys, os
import pandas as pd
import time
from atomD import edge2atom_allFrame

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

def get_dft_forces_fr(dft_file, return_num_atoms=False):
    """Get DFT forces for each frame.
    Args:
        dft_file: str, path to DFT file
        return_num_atoms: bool, whether to return the number of atoms in each frame
    Returns:
        dft_forces_fr: list of np.array, each array is the (N,3) forces of a frame,
        where N is the number of atoms in the frame
        num_atom_fr: np.array of int, number of atoms in each frame"""
    pipeline = import_file(dft_file)
    dft_forces_fr = []
    num_atom_fr = []
    for i in range(pipeline.source.num_frames):
        data_i = pipeline.compute(i)
        dft_forces_fr.append(data_i.particles['Force'].array)
        if return_num_atoms:
           num_atom_fr.append(data_i.particles.count)
    if return_num_atoms:
        return dft_forces_fr, np.array(num_atom_fr)
    else:
        return dft_forces_fr

def get_frmse_perFrame(dft_file, test_forces_fr, return_num_atoms=False):
    # get DFT forces for each frame
    if return_num_atoms:
        dft_forces_fr, num_atom_fr = get_dft_forces_fr(dft_file, return_num_atoms=True)
    else:   
        dft_forces_fr = get_dft_forces_fr(dft_file)
    f_rmse_fr = []
    for i in range(len(dft_forces_fr)):
        # calculate the RMSE for each frame in meV/Å
        f_rmse = 1e3*mean_squared_error(dft_forces_fr[i],test_forces_fr[i], squared=False)
        f_rmse_fr.append(f_rmse)
    if return_num_atoms:
        return np.array(f_rmse_fr), num_atom_fr
    else:
        return np.array(f_rmse_fr), None

def get_frmse_perAtom(dft_file, test_forces_fr, return_num_atoms=False):
    # get DFT forces
    if return_num_atoms:
        dft_forces_fr, num_atoms_fr = get_dft_forces_fr(dft_file,return_num_atoms=True)
        f_rmse, num_atoms = [], []
        for i in range(len(dft_forces_fr)):
            f_rmse_i = 1e3*np.sqrt(np.mean(np.square(dft_forces_fr[i]-test_forces_fr[i]), axis=1))
            num_atoms_i = np.ones(len(f_rmse_i))*num_atoms_fr[i]
            f_rmse.append(f_rmse_i)
            num_atoms.append(num_atoms_i)
        return np.concatenate(f_rmse), np.concatenate(num_atoms)
    else:
        dft_forces_fr = get_dft_forces_fr(dft_file)
        # cooncatenate per-atom forces from all frames
        dft_forces_atom = np.concatenate(dft_forces_fr, axis=0)
        test_forces_atom = np.concatenate(test_forces_fr, axis=0)
        # calculate the RMSE for each atom
        f_rmse = 1e3*np.sqrt(np.mean(np.square(dft_forces_atom-test_forces_atom), axis=1))
        return f_rmse, None

        
def get_ferr_perAtom(dft_file, test_forces_fr):
    # get DFT forces
    dft_forces_fr = get_dft_forces_fr(dft_file)
    # cooncatenate per-atom forces from all frames
    dft_forces_atom = np.concatenate(dft_forces_fr, axis=0)
    test_forces_atom = np.concatenate(test_forces_fr, axis=0)
    # calculate error percentage of forces on each atom
    f_rmse = 1e2*np.sqrt(np.mean(np.square(dft_forces_atom-test_forces_atom), axis=1))/np.sqrt(np.mean(np.square(dft_forces_atom), axis=1))
    return f_rmse


def get_NLL_perFrame(features_fr, gmm, num_atoms, pca=None, metric='max', reduce_dim=8):
    """Calculate the negative log likelihood for each MD frame by fitting the GMM model.
    Args:
        features_fr: list of np.array
            each array contains all feature vectors of a frame
        gmm: sklearn.mixture.GaussianMixture
        num_atoms: number of atoms in each frame
        pca: sklearn.decomposition.PCA, fitted on all data
        metric: str, 'max' or 'sum'
    Returns:
        NLL_fr: array of NLL for each frame"""
    if pca is None:
        pca = PCA(n_components=reduce_dim)
        pca.fit(np.concatenate(features_fr, axis=0))
    NLL_fr = []
    for features in features_fr:
        # estimate NLL for each feature vector
        test_X_pca = pca.transform(features)
        NLL = -gmm.score_samples(test_X_pca)
        if metric == 'max':
            NLL_fr.append(np.max(NLL))
        elif metric == 'sum':
            NLL_fr.append(np.sum(NLL))
        else:
            raise ValueError('metric must be max or sum.')
    if metric == 'sum':
        return np.array(NLL_fr)/num_atoms
    else:
        return np.array(NLL_fr)

def get_NLL_perAtom(features_fr, edge_index_fr, gmm, pca=None, metric='max', reduce_dim=8):
    """Calculate the negative log likelihood for each atom by fitting the GMM model.
    Args:
        features_fr: list of np.array
            each array contains all feature vectors of a frame
        edge_index_fr: list of np.array
            each array is the edge index of a frame
            with shape (2, num_edges) and row 0 is the source node (atom) index and row 1 is the target node (atom) index
        gmm: sklearn.mixture.GaussianMixture
        pca: sklearn.decomposition.PCA, fitted on all data
        metric: str, 'max' or 'sum'
    Returns:
        NLL_atom: array of NLL for each atom
    Description:
        To define the NLL of an atom, we pick the maximum NLL of all edges that belong to the same source atom."""
    if pca is None:
        pca = PCA(n_components=reduce_dim)
        pca.fit(np.concatenate(features_fr, axis=0))
    score_atom = []
    for features, edge_index in zip(features_fr, edge_index_fr):
        # estimate NLL for each feature vector
        test_X_pca = pca.transform(features)
        score_features = -gmm.score_samples(test_X_pca)
        # get the max score from all feature vectors of the same source atom
        source_node_index = edge_index[0]
        num_atoms = np.max(source_node_index) + 1
        for i in range(num_atoms):
            if metric == 'max':
                score_atom.append(np.max(score_features[source_node_index == i]))
            elif metric == 'sum':
                score_atom.append(np.sum(score_features[source_node_index == i]))
            else:
                raise ValueError('metric must be max or sum.')
    return np.array(score_atom)

def get_NLL_use_atomD(features_fr, gmm, pca=None, reduce_dim=8):
    """get NLL using per-atom features"""
    ### Need to double check the implementation
    if pca is None:
        pca = PCA(n_components=reduce_dim)
        test_X_pca = pca.fit_transform(np.concatenate(features_fr, axis=0))
        NLL = -gmm.score_samples(test_X_pca)
    #NLL_fr = []
    # for features in features_fr:
    #     num_atom = features.shape[0]
    #     # estimate NLL for each feature vector
    #     for i in range(num_atom):
    #         test_X_pca = pca.transform(features[i])
            
    #         NLL_fr.append(NLL)
    return NLL

def load_GMM(model_name):
    """Load the GMM model from model_name."""
    means = np.load(model_name + '_means.npy')
    covar = np.load(model_name + '_covariances.npy')
    loaded_gmm = GaussianMixture(n_components = len(means), covariance_type='full')
    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    loaded_gmm.weights_ = np.load(model_name + '_weights.npy')
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar
    return loaded_gmm

def PCA_setup(pca_comp, pca_fit, train_features_fr, test_features_fr, sub_train_size):
    """Setup PCA model.
    Args:
        pca_comp: int, number of PCA components
        pca_fit: str, 'all' or 'perSet'
            'all': fit PCA on all data
            'perSet': fit PCA on training and testing data separately
    Returns:
        pca: sklearn.decomposition.PCA
    """
    if pca_fit == 'all':
        pca = PCA(n_components=pca_comp)
        np.random.seed(42)
        test_X = np.concatenate(test_features_fr, axis=0)
        train_select_idx = np.random.choice(train_features_fr.shape[0], size=sub_train_size, replace=False)
        train_features_select = [train_features_fr[i] for i in train_select_idx]
        train_X_sub = np.concatenate(train_features_select, axis=0)
        # fit pca on all data
        all_X = np.concatenate((train_X_sub, test_X), axis=0)
        pca.fit(all_X)
    elif pca_fit == 'perSet':
        pca = None
    else:
        raise ValueError('pca_fit must be all or perSet.')
    return pca

if __name__ == '__main__':
    ## fit a GMM model with M frames of training data ##
    ### read training & testing data ###
    STORE_PATH='GMM_275/feature_npz'
    DATA_SET='test' 
    PCA_COMP=8 
    SUB_TRAIN_SIZE=50
    GMM_DIR=f'{DATA_SET}_PCA_perSet_M={SUB_TRAIN_SIZE}gmm' 
    train_path=f'{STORE_PATH}/{DATA_SET}_train.npz'
    Train=np.load(train_path, allow_pickle=True)
    train_features_fr = Train['features_fr']
    train_edge_index_fr = Train['edge_index_fr']
    train_part_type_fr = Train['part_type_fr']
    test_path=f'{STORE_PATH}/{DATA_SET}_test_atom.npz'
    Test=np.load(test_path, allow_pickle=True)
    test_features_fr = Test['atom_features_fr_norm']
    test_forces_fr = Test['forces_fr']

    ## reduce feature dimension for training & testing data ##
    # pca = PCA(n_components=PCA_COMP)
    ### fit pca on all data ###
    # train_X = np.concatenate(train_features_fr, axis=0)
    # test_X = np.concatenate(test_features_fr, axis=0)
    # all_X = np.concatenate((train_X, test_X), axis=0)
    # pca.fit(all_X)

    ## randomly select M frames for training ##
    np.random.seed(42)
    train_select_idx = np.random.choice(train_features_fr.shape[0], size=SUB_TRAIN_SIZE, replace=False)
    train_features_select = [train_features_fr[i] for i in train_select_idx]
    train_edge_index_select = [train_edge_index_fr[i] for i in train_select_idx]
    train_part_type_select = [train_part_type_fr[i] for i in train_select_idx]
    ### convert to atomic features ###
    train_features_select = edge2atom_allFrame(train_features_select, train_edge_index_select, train_part_type_select)
    train_X_sub = np.concatenate(train_features_select, axis=0)
    ## fit pca on each set ##
    pca = PCA(n_components=PCA_COMP)
    train_X_sub_pca = pca.fit_transform(train_X_sub)
    ## fit pca on each frame ##
    # train_X_sub_pca = []
    # for features in train_features_select:
    #     pca = PCA(n_components=8)
    #     pca.fit(features)
    #     train_X_sub_pca.append(pca.transform(features))
    # train_X_sub_pca = np.concatenate(train_X_sub_pca, axis=0)
    ## fit pca on all data ##
    # all_X = np.concatenate((train_X_sub, test_X), axis=0)
    # pca.fit(all_X)
    # train_X_sub_pca = pca.transform(train_X_sub)
    

    ## fit multiple k-component GMM model ##
    print(f'---fitting GMM with M={SUB_TRAIN_SIZE} frames---')
    t0 = time.time()
    n_choice = [1,5,10,20] #np.arange(60, 100+1, 5)
    models = [
        GaussianMixture(n_components=n, covariance_type="full", random_state=4242, max_iter=int(3e4))
        for n in n_choice
        ]
    bics = []
    for model in models:
        print(f'---fitting GMM with {model.n_components} components---')
        model.fit(train_X_sub_pca)
        bics.append(model.bic(train_X_sub_pca))
    convergence = [model.converged_ for model in models]
    t1 = time.time()
    print(f'---GMM fitting time: {(t1-t0)/60:.2f} min---')

    ## save BIC ##
    if not os.path.exists(GMM_DIR):
        os.makedirs(GMM_DIR)
    df = pd.DataFrame({"n_components": n_choice, "bic": bics, "convergence": convergence})
    if os.path.exists(f"{GMM_DIR}/M={SUB_TRAIN_SIZE}_bic.csv"):
        df.to_csv(f"{GMM_DIR}/M={SUB_TRAIN_SIZE}_bic.csv", index=False, mode="a", header=False)
    else:
        df.to_csv(f"{GMM_DIR}/M={SUB_TRAIN_SIZE}_bic.csv", index=False)

    ## predict ##
    gmm = models[np.argmin(bics)]
    test_NLL = get_NLL_use_atomD(test_features_fr, gmm, reduce_dim=PCA_COMP)
    ## save the best GMM model ##
    print(f'---saving the best GMM model---')
    gmm_name = f'{GMM_DIR}/gmm_n={gmm.n_components}'
    np.save(gmm_name + '_weights', gmm.weights_, allow_pickle=False)
    np.save(gmm_name + '_means', gmm.means_, allow_pickle=False)
    np.save(gmm_name + '_covariances', gmm.covariances_, allow_pickle=False)

    ## get per-atom force rmse ##
    dft_file = 'DFT/test_CDP.xyz'
    f_rmse,_ = get_frmse_perAtom(dft_file, test_forces_fr)

    ## plot f_rmse vs NLL ##
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 16})
    plt.scatter(f_rmse, test_NLL, alpha=0.5)
    plt.title('per-atom estimation')
    # plt.hist2d(f_rmse, test_NLL, bins=200, cmap='Blues')
    # plt.colorbar()
    plt.xlabel('f_rmse [meV/Å]')
    plt.ylabel('Negative Log-Likelihood')
    #plt.ylim(-10, 100)
    plt.savefig(f'{GMM_DIR}/M={SUB_TRAIN_SIZE}=n={n_choice[np.argmin(bics)]}.png', dpi=300)



    ## fit multiple k-component GMM model ##
    # param_grid = {
    #     "n_components": np.arange(30, 80+1, 5),
    # }

    # grid_search = GridSearchCV(
    #     GaussianMixture(covariance_type='full', max_iter=int(1e4), random_state=4242), param_grid=param_grid, scoring=gmm_bic_score, n_jobs=-1, verbose=4
    # )

    # grid_search.fit(train_X_pca)

    ## save the BIC score ##
    # df = pd.DataFrame(grid_search.cv_results_)[
    #     ["param_n_components", "mean_test_score"]
    # ]
    # df["mean_test_score"] = -df["mean_test_score"]
    # df = df.rename(
    #     columns={
    #         "param_n_components": "Number of components",
    #         "param_covariance_type": "Type of covariance",
    #         "mean_test_score": "BIC score",
    #     }
    # )
    ### check if file exists, if yes, append to it ###
    # if os.path.exists("gmm_bic.csv"):
    #     df.to_csv("gmm_bic.csv", index=False, mode="a", header=False)
    # else:
    #     df.to_csv("gmm_bic.csv", index=False)
    # ## np save the best model ##
    # # gmm = grid_search.best_estimator_
    # ### check whether the folder exists ###
    # if not os.path.exists("saved_gmm"):
    #     os.makedirs("saved_gmm")
    # gmm_name = f'saved_gmm/gmm_k={gmm.n_components}'
    # np.save(gmm_name + '_weights', gmm.weights_, allow_pickle=False)
    # np.save(gmm_name + '_means', gmm.means_, allow_pickle=False)
    # np.save(gmm_name + '_covariances', gmm.covariances_, allow_pickle=False)
    # print(f'---GMM convergence: {gmm.converged_}')