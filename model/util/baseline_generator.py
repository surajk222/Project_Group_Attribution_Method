from data.datasets import DryBean
from data.util.utils import DatasetMode
import numpy as np
import torch

def generate_uniform_baseline(n=50):

    dataset = DryBean(DatasetMode.TRAIN)

    train_inputs = dataset[:][0]
    train_labels = dataset[:][1]

    print(train_inputs.shape)
    print(train_labels.shape)

    categories_dict = dataset.categories

    mean_of_categories = torch.zeros((7,train_inputs.shape[1]))

    for category_idx in categories_dict.values():
        mask_of_category = train_labels.argmax(dim=1)==category_idx
        datapoints_of_category = train_inputs[mask_of_category,:]

        #Anzahl der Datenpunkte je Kategorie
        #print(len(datapoints_of_category))

        #print(datapoints_of_category.shape)

        np.random.seed(42)

        sample_indexes = np.random.choice(len(datapoints_of_category),n,replace=False)

        sample_datapoints = np.take(datapoints_of_category, sample_indexes, axis=0)

        mean_of_category = torch.mean(a=sample_datapoints, axis=0)

        mean_of_categories[category_idx,:] = mean_of_category

    return torch.mean(mean_of_categories, axis=0)

