from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
from sklearn.metrics import pairwise_distances
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

""" Define a class to contain the data that will be included in the dataloader 
sent to the GCN model """

class GCN_Dataset(Dataset):
  
    def __init__(self, data_file):
        super(GCN_Dataset, self).__init__()
        self.data_file = data_file
        self.data_dict = {}
        self.data_list = []
        self.data_hdf = h5py.File(data_file, 'r')
        
        # retrieve PDB id's and affinities from hdf file
        with h5py.File(data_file, 'r') as f:
            for pdbid in f.keys():
                affinity = np.asarray(f[pdbid].attrs['affinity']).reshape(1, -1)
                self.data_list.append((pdbid, affinity))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        pdbid, affinity = self.data_list[idx]
        node_feats, coords = None, None
        coords = h5py.File(self.data_file,'r')[pdbid][:,0:3]
        dists = pairwise_distances(coords, metric='euclidean')
        x = torch.from_numpy(node_feats).float()
        y = torch.FloatTensor(affinity).view(-1, 1)
        
        affinity = self.data_hdf[pdbid].attrs['affinity'].reshape(1,-1)
        vdw_radii = (self.data_hdf[pdbid].attrs['van_der_waals'].reshape(-1, 1))
        node_feats = np.concatenate([vdw_radii, self.data_hdf[pdbid][:, 3:22]], axis=1)
        edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float()) 
        x = torch.from_numpy(node_feats).float()
        y = torch.FloatTensor(affinity).view(-1, 1)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y)
        
        return data
