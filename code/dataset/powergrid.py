# PowerGrid dataset is licensed under a CC BY-SA 4.0 license.

"""
General file to load the Inmemory datasets (UK, IEEE24, IEEE39, IEEE118)

"""


import os.path as osp
import torch
import mat73
from sklearn.model_selection import train_test_split
import os
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
import torch
from torch_geometric.data import Data
import numpy as np
import scipy
from utils.gen_utils import from_edge_index_to_adj, padded_datalist


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def index_edgeorder(edge_order):
    return torch.tensor(edge_order["bList"]-1)


class PowerGrid(InMemoryDataset):
    # Base folder to download the files
    names = {
        "uk": ["uk", "Uk", "UK", None],
        "ieee24": ["ieee24", "Ieee24", "IEEE24", None],
        "ieee39": ["ieee39", "Ieee39", "IEEE39", None],
        "ieee118": ["ieee118", "Ieee118", "IEEE118", None],
            }

    def __init__(self, root, name, datatype='node', transform=None, pre_transform=None, pre_filter=None):
        
        self.datatype = datatype.lower()
        self.name = name.lower()
        self.raw_path = os.path.join(root, 'raw')
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        
        assert self.name in self.names.keys()
        super(PowerGrid, self).__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0]) 

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        if self.datatype.lower() == 'node':
            return osp.join(self.root, self.name, "processed_node")
        if self.datatype.lower() == 'nodeopf':
            return osp.join(self.root, self.name, "processed_nodeopf")

    @property
    def raw_file_names(self):
        # List of the raw files
        return ['X.mat',
                'Y_polar.mat',
                'edge_index.mat',
                'edge_attr.mat',
                'edge_index_opf.mat',
                'edge_attr_opf.mat',
                'Xopf.mat',
                'Y_polar_opf.mat',
                ]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download the file specified in self.url and store
        # it in self.raw_dir
        path = self.raw_path

    def process(self):
        # function that deletes row
        def th_delete(tensor, indices):
            mask = torch.ones(tensor.size(), dtype=torch.bool)
            mask[indices] = False
            return tensor[mask]

        if self.datatype.lower() == 'nodeopf':
            path = os.path.join(self.raw_dir, 'edge_index_opf.mat')
            edge_index = scipy.io.loadmat(path)
            path = os.path.join(self.raw_dir, 'edge_attr_opf.mat')
            edge_attr = scipy.io.loadmat(path)
            path = os.path.join(self.raw_dir, 'Xopf.mat')
            X = mat73.loadmat(path)
            path = os.path.join(self.raw_dir, f'Y_polar_opf.mat')
            Y = mat73.loadmat(path)
            edge_order = torch.tensor(edge_index['edge_index'] - 1, dtype=torch.long).t().contiguous().to(device)
            edge_attr = torch.tensor(edge_attr['edge_attr'], dtype=torch.float)
            edge_attr = torch.nn.functional.normalize(edge_attr, dim=1)
            data_list = []
            fullX = []
            fullY= []

            for i in range(int(len(X['X']))):
                fullX.append(torch.tensor(X['X'][i][:, [0, 1, 3]], dtype=torch.float, device=device))
                fullY.append(torch.tensor(Y['Y_polar'][i], dtype=torch.float, device=device))
            fullXcat = torch.cat(fullX, dim=0)
            fullYcat = torch.cat(fullY, dim=0)
            maxsX, _ = torch.max(torch.abs(fullXcat), dim=0)
            maxsY, _ = torch.max(torch.abs(fullYcat), dim=0)
            
            for i in range(int(len(X['X']))):
                N = fullX[i]
                mask = fullY[i] != 0
                N_norm = N/maxsX
                edge_attr = torch.nn.functional.normalize(edge_attr, dim=0)
                Y_o = fullY[i]
                Y_norm = Y_o /maxsY
                data = Data(x=N_norm, edge_index=edge_order.to(device), y=Y_norm, edge_attr=edge_attr, maxs=maxsY, mask=mask).to(device)
                data_list.append(data)

        else:
            path = os.path.join(self.raw_dir, 'edge_index.mat')
            edge_index = scipy.io.loadmat(path)
            path = os.path.join(self.raw_dir, 'edge_attr.mat')
            edge_attr = scipy.io.loadmat(path)
            path = os.path.join(self.raw_dir, 'X.mat')
            X = mat73.loadmat(path)
            path = os.path.join(self.raw_dir, f'Y_polar.mat')
            Y = mat73.loadmat(path)
            edge_order = torch.tensor(edge_index['edge_index'] - 1, dtype=torch.long).t().contiguous().to(device)
            edge_attr = torch.tensor(edge_attr['edge_attr'], dtype=torch.float)
            edge_attr = torch.nn.functional.normalize(edge_attr, dim=1)
            data_list = []

            fullX = []
            fullY= []
            for i in range(int(len(X['Xpf']))):
                fullX.append(torch.tensor(X['Xpf'][i], dtype=torch.float, device=device)) #remove input angle as always unknown or zero
                fullY.append(torch.tensor(Y['Y_polarpf'][i], dtype=torch.float, device=device))

            fullXcat = torch.cat(fullX, dim=0)
            fullYcat = torch.cat(fullY, dim=0)
            maxsX, _ = torch.max(torch.abs(fullXcat), dim=0)
            maxsY, _ = torch.max(torch.abs(fullYcat), dim=0)

            for i in range(int(0.5*len(X['Xpf']))):
                N = fullX[i]
                mask = fullY[i] != 0
                N_norm = N/maxsX
                edge_attr = torch.nn.functional.normalize(edge_attr, dim=0)
                Y_o = fullY[i]
                Y_norm = Y_o /maxsY
                data = Data(x=N_norm, edge_index=edge_order.to(device), y=Y_norm, edge_attr=edge_attr, maxs=maxsY, mask=mask).to(device)
                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


