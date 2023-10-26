from _reader import DataReader
import numpy as np


# improve the loading speed
def convert_save_npz(infile, outfile, edge_dim=64):
    Data = DataReader(infile, edge_dim)
    features_fr = Data.edge_features_fr
    edge_index_fr = Data.edge_index_fr
    part_type_fr = Data.get_atom_type_fr()
    edge_energy_fr = Data.edge_energy_fr
    forces_fr = Data.get_forces_fr()
    # get object array
    M = len(features_fr)
    features_fr_store = np.empty(M, object)
    features_fr_store[:] = features_fr
    edge_index_fr_store = np.empty(M, object)
    edge_index_fr_store[:] = edge_index_fr
    part_type_fr_store = np.empty(M, object)
    part_type_fr_store[:] = part_type_fr
    edge_energy_fr_store = np.empty(M, object)
    edge_energy_fr_store[:] = edge_energy_fr
    train_forces_fr_store = np.empty(M, object)
    train_forces_fr_store[:] = forces_fr
    # save
    np.savez_compressed(outfile, features_fr=features_fr_store, edge_index_fr=edge_index_fr_store, part_type_fr=part_type_fr_store, edge_energy_fr=edge_energy_fr_store, forces_fr=train_forces_fr_store)

if __name__ == "__main__":
    
    ### ovito data reader, too slow ###
    file_path='test.xyz'
    Data = DataReader(file_path, edge_dim=64)
    features_fr = Data.edge_features_fr
    edge_index_fr = Data.edge_index_fr
    part_type_fr = Data.get_atom_type_fr()
    edge_energy_fr = Data.edge_energy_fr

    ### convert and save ###
    work='GMM_275/'
    convert_save_npz(work+'test.xyz', work+'test_with_features.xyz', edge_dim=64)