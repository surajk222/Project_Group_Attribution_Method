import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from data.util.utils import DatasetMode


class DryBean(Dataset):
    """Dry Bean dataset."""

    def __init__(
        self,
        mode: DatasetMode,
        transform: callable=None
        )->None:
        """
        Args:
            mode (DatasetMode): Train, Validation or Test-Dataset.
            transform (callable, optional): Transform to be applied.
        """
        file_path = {
            DatasetMode.TRAIN: r"data\files\splitted\Dry_Bean_Dataset_Train.csv",
            DatasetMode.VALIDATION: r"data\files\splitted\Dry_Bean_Dataset_Validation.csv",
            DatasetMode.TEST: r"data\files\splitted\Dry_Bean_Dataset_Test.csv"
        }
        
        data_frame = pd.read_csv(file_path[mode],sep=";")

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
        self.datapoints, self.labels = self._normalize_and_split_into_data_and_categories(data_frame)

    def __len__(self):
        return len(self.datapoints)
    
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

        datapoint = self.datapoints[idx]
        if self.transform:
            datapoint=self.transform(datapoint)

        label = self.labels[idx]

        return datapoint, label
    
    def _normalize_and_split_into_data_and_categories(
        self,
        data_df: pd.DataFrame
        )->tuple[torch.Tensor, torch.Tensor]:
        """
        Normalizes the Dataset. Each feature is minmax-scaled between [0,1].

        Args:
            data_df (pd.DataFrame): Dataframe that must be normalized.

        Returns:
            Tuple of datapoints_tensor, categories_tensor

            datapoints_tensor (torch.Tensor): data_df[:,0:-1] normalized.
            categories_tensor (torch.Tensor): data_df[:,-1], so the labels.
        """
        datapoints = data_df.iloc[:,0:-1]


        datapoints = (datapoints - datapoints.min()) / (datapoints.max() - datapoints.min())
        datapoints_np = datapoints.to_numpy(dtype=np.float32)
        datapoints_tensor = torch.from_numpy(datapoints_np)

        categories_df = data_df.iloc[:,-1]
        categories_np = pd.get_dummies(categories_df).to_numpy(dtype=np.float32)
        categories_tensor = torch.from_numpy(categories_np)

        return datapoints_tensor, categories_tensor

    

