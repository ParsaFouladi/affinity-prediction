import h5py
import torch
from torch.utils.data import Dataset, DataLoader,SubsetRandomSampler,Subset
import random
import numpy as np

class ProteinLigandTrain(Dataset):
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
        else:
            # Ensure the representation is in the format (channels, height, width)
            if representation.shape[-1] == 3:
                representation = np.transpose(representation, (2, 0, 1))

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
    
    def get_train_val_split(self, val_split=0.1, seed=42):
        """Splits the dataset into training and validation sets with shuffling."""
        dataset_size = len(self)
        indices = list(range(dataset_size))
        np.random.seed(seed)  # Set random seed for reproducibility
        np.random.shuffle(indices)
        split = int(np.floor(val_split * dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)  # Create a sampler for validation as well
        return train_sampler, val_sampler

# Example Usage (Slightly modified)

### Usage Example
# dataset = ProteinLigandDataset(h5_file_path)

# # Split dataset
# train_sampler, val_sampler = dataset.get_train_val_split(dataset)

# # Create DataLoaders with samplers for both training and validation
# train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
# val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)  # Shuffle validation set

# # test the dataloader
# path="../representation/representations.h5"
# dataset = ProteinLigandDataset(path)
# print(dataset.head())
# #print(dataset.show_random_sample())
# print(len(dataset))
# print(dataset[0])
# print(dataset[0][0][3,1,0])
# dataset.close()


class ProteinLigandTest(Dataset):
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
        else:
            # Ensure the representation is in the format (channels, height, width)
            if representation.shape[-1] == 3:
                representation = np.transpose(representation, (2, 0, 1))

        # Convert to PyTorch tensors and return as float 32

        return torch.from_numpy(representation).float(), torch.tensor(p_binding_affinity).float(), group_name
    
    def head(self):
        return self.group_names[:5]
    
    def show_random_sample(self):
        idx = random.randint(0, len(self.group_names))
        pdb_code = self.group_names[idx]
        return self.__getitem__(idx), pdb_code

    def close(self):
        self.h5_file.close()
