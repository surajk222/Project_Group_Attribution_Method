import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import pandas as pd

class DryBean():
    def __init__(self, train, **kwargs):
        dry_bean = _DryBean(**kwargs)

        train_size = int(0.8*len(dry_bean))
        test_size = len(dry_bean)-train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dry_bean, train_size, test_size)

        if train:
            return train_dataset
        else: 
            return test_dataset


class _DryBean(Dataset):
    """Dry Bean dataset."""

    def __init__(self, csv_file, transform=None, shuffle=True):
        """
        Args:
            csv_file (string): Path to the csv file.
            transform (callable, optional): Optionale transform to be applied.
        """
        self.data_frame = pd.read_excel(io=csv_file)
        self.transform = transform
        self.categories = {
            "BARBUNYA": 0,
            "BOMBAY": 1,
            "CALI": 2,
            "DERMASON": 3,
            "HOROZ": 4,
            "SEKER": 5,
            "SIRA": 6
        }

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        """
        Arguemnts: 
            idx (int): Index of the datapoint requested.
            
        Returns:
            A tuple containing the requested datapoint, label. 

            datapoint (np.array with size (16,))
            label (np.array with size (7,))
            """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datapoint = self.data_frame.iloc[idx,0:-1]
        datapoint = np.array(datapoint)
        if self.transform:
            datapoint=self.transform(datapoint)

        category = self.categories[self.data_frame.iloc[idx,-1]]
        label = np.zeros((7,))
        label[category] = 1

        return datapoint, label