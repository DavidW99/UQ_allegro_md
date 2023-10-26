import numpy as np
from ovito.io import import_file
# import matplotlib.pyplot as plt

# allegro edge representation reader
class DataReader:
    def __init__(self, file_path, edge_dim=128):
        # trajectory properties
        self.file_path = file_path
        self.pipeline = import_file(self.file_path)
        self.num_frames = self.pipeline.source.num_frames
        # allegro properties
        self.edge_dim = edge_dim
        # per frame properties
        self.num_particles_fr = np.zeros(self.num_frames)
        self.energy_fr = np.zeros(self.num_frames, dtype=np.float32)
        self.forces_fr = []
        self.edge_index_fr = []
        self.edge_features_fr = []
        self.edge_energy_fr = []
        for i in range(self.num_frames):
            # get data at frame i
            data_i = self.pipeline.compute(i)
            # get the energy at frame i
            self.energy_fr[i] = data_i.attributes['energy']
            # get the forces at frame i
            self.forces_fr.append(data_i.particles['Force'].array)
            # get the number of particles at frame i
            self.num_particles_fr[i] = data_i.particles.count
            # get the edge index at frame i
            index_lst = data_i.attributes['edge_index'].replace(']', '').replace('[', '').replace('_JSON', '').replace(' ', '').split(',')
            edge_index = np.array(index_lst).astype(np.int32).reshape(2, -1)
            self.edge_index_fr.append(edge_index)
            # get the edge features at frame i
            feature_lst = data_i.attributes['edge_features'].replace(']', '').replace('[', '').replace('_JSON', '').replace(' ', '').split(',')
            edge_features = np.array(feature_lst).astype(np.float32).reshape(-1, self.edge_dim)
            self.edge_features_fr.append(edge_features)
            # get the edge energy at frame i
            edge_energy_lst = data_i.attributes['edge_energy'].replace(']', '').replace('[', '').replace('_JSON', '').replace(' ', '').split(',')
            edge_energy = np.array(edge_energy_lst).astype(np.float32).reshape(-1, 1)
            self.edge_energy_fr.append(edge_energy)
            # check the dimension of edge energy
            assert edge_energy.shape[0] == edge_features.shape[0], \
                f'Parser Error: at frame {i}, the number of edge energy and the number of edges are not equal!'
            # check the dimension of edge features and edge index
            assert edge_index.shape[1] == edge_features.shape[0], \
                f'Parser Error: at frame {i}, the number of edge features and the number of edges are not equal!'
            # check the dimension of edge features
            assert edge_features.shape[1] == self.edge_dim, \
                'Parser Error: the dimension of edge features is not equal to the edge_dim!'

    def get_energy_fr(self):
        return self.energy_fr
    
    def get_forces_fr(self):
        return self.forces_fr

    def get_edge_index_fr(self):
        return self.edge_index_fr
    
    def get_edge_features_fr(self):
        return self.edge_features_fr

    def get_number_of_particles_fr(self):
        return self.num_particles_fr

    def get_pipeline(self):
        return self.pipeline

    def get_file_path(self):
        return self.file_path

    def get_edge_dim(self):
        return self.edge_dim
    
    def get_atom_type_fr(self):
        pipeline = self.get_pipeline()
        self.atom_type_fr = []
        for i in range(self.num_frames):
            data_i = pipeline.compute(i)
            self.atom_type_fr.append(np.array(data_i.particles['Particle Type']))
        return self.atom_type_fr

