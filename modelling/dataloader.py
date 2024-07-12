import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import random

class ProteinLigandDataset(Dataset):
    def __init__(self, h5_file_path, transform=None):
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.group_names = list(self.h5_file.keys())
        self.transform = transform

    def __len__(self):
        return len(self.group_names)

    def __getitem__(self, idx):
        group_name = self.group_names[idx]
        group = self.h5_file[group_name]

        representation = group['representation'][:]  # Load representation
        p_binding_affinity = group.attrs["p_binding_affinity"] 

        if self.transform:
            representation = self.transform(representation)

        # Convert to PyTorch tensors and return as float 32

        return torch.from_numpy(representation).float(), torch.tensor(p_binding_affinity).float()
    
    def head(self):
        return self.group_names[:5]
    
    def show_random_sample(self):
        idx = random.randint(0, len(self.group_names))
        pdb_code = self.group_names[idx]
        return self.__getitem__(idx), pdb_code

    def close(self):
        self.h5_file.close()

# test the dataloader
path="../representation/representations.h5"
dataset = ProteinLigandDataset(path)
print(dataset.head())
#print(dataset.show_random_sample())
print(len(dataset))
print(dataset[0])
dataset.close()