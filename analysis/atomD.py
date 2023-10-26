import numpy as np

### convert edge_features into per-atom features ###


def edge2atom(edge_features, edge_index, part_type, normalize=True):
    # make part_type start from 0
    part_type = part_type-1
    num_atom = part_type.shape[0]
    num_type = np.unique(part_type).shape[0]
    D_edge = edge_features.shape[1]
    # reserve space for sum each type per atom
    atoms_feature = np.zeros((num_atom, num_type, D_edge))
    if normalize:
        count_type = np.zeros((num_atom, num_type))
        # iterate over source atoms
        for source_idx in edge_index[0]:
            for target_idx in edge_index[1]:
                target_type = part_type[target_idx]
                count_type[source_idx, target_type] += 1
                atoms_feature[source_idx, target_type] += edge_features[target_idx]
        atoms_feature = atoms_feature/np.sqrt(count_type)[:,:,None]
    else:
        for source_idx in edge_index[0]:
            for target_idx in edge_index[1]:
                target_type = part_type[target_idx]
                atoms_feature[source_idx, target_type] += edge_features[target_idx]
    atoms_feature_concat = np.zeros((num_atom, num_type*D_edge))
    for i in range(num_atom):
        atoms_feature_concat[i]=np.concatenate(atoms_feature[i], axis=0)
    return atoms_feature_concat


def edge2atom_allFrame(features_fr, edge_index_fr, part_type_fr, normalize=True):
    # get atom features
    atom_features_fr = np.empty(len(features_fr), dtype=object)
    for i in range(len(features_fr)):
        atom_features_fr[i] = (edge2atom(features_fr[i], edge_index_fr[i], part_type_fr[i], normalize=True))
    return atom_features_fr

if __name__ == '__main__':
    ### read training & testing data ###
    STORE_PATH='GMM_275/feature_npz'
    DATA_SET='test'
    test_path=f'{STORE_PATH}/{DATA_SET}_train.npz'
    Test=np.load(test_path, allow_pickle=True)
    features_fr = Test['features_fr']
    edge_index_fr = Test['edge_index_fr']
    part_type_fr = Test['part_type_fr']
    forces_fr = Test['forces_fr']

    ### get per-atom features ###
    atom_features_fr_norm = edge2atom_allFrame(features_fr, edge_index_fr, part_type_fr, normalize=True)
    atom_features_fr = edge2atom_allFrame(features_fr, edge_index_fr, part_type_fr, normalize=False)

    ### save per-atom features ###
    save_dict = {'atom_features_fr': atom_features_fr, 'atom_features_fr_norm': atom_features_fr_norm}
    for key in Test:
        save_dict[key] = Test[key]
    np.savez(f'{STORE_PATH}/{DATA_SET}_train_atom.npz', **save_dict)
