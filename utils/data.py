import os

import torch
from torch_geometric.data import Dataset


class IsoAbundanceDataset(Dataset):
    """
    A custom PyTorch Geometric dataset for isoform abundance prediction.

    This dataset is designed to handle data related to isoform abundance,
    loading it from a specified file and providing access to individual
    data samples. The data is expected to be preprocessed and stored as
    a list of PyTorch tensors or data objects in a file.
    """
    def __init__(self, root, file_name, transform=None, pre_transform=None):
        super(IsoAbundanceDataset, self).__init__(root, transform, pre_transform)
        self.file_path = os.path.join(root, file_name)
        self.data_list = torch.load(self.file_path)

    @property
    def processed_file_names(self):
        return [self.file_path]

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
    