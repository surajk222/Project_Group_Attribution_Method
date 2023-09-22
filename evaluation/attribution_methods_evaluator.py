import torch
import numpy as np

from typing import Callable

from tqdm import tqdm

from model.models import NeuralNetwork
from data.datasets import DryBean
from data.util.utils import DatasetMode
from model.attribution_methods.integrated_gradients import IntegratedGradients
from evaluation.utils.visualisation import _visualize_log_odds

import copy



class AttributionMethodsEvaluator():

    """
    This class serves the evaluation and comparison of the three attribution methods Integrated Gradients, Lime and KernelShap.
    """

    def __init__(self, model: NeuralNetwork):
        self.model = model
        self.dataset = DryBean(DatasetMode.TEST)
        self.ig = IntegratedGradients(self.model.get_gradients_with_respect_to_inputs)

    def get_log_odds_of_datapoint(
            self,
            x: torch.Tensor,
            attribute,
            apply_log: bool = True,
            masking_baseline = torch.zeros(16),
            **kwargs) -> np.ndarray:
        """
        Calculates the log odds of one datapoint. Designed generically so it can work with different attribution methods. To work with an attribution method,
            it needs to implement the attribute callable (see args below).

        Args:
            x (torch.Tensor): Input for which the Log Odds are being calculated.
            attribute (Callable): Calculates the attribution scores for an input and target label index.
                Args:
                    x (torch.Tensor): Input for which the attribution scores are calculated
                    target_label_index (int): Index of the output feature for which the attribution scores are 
                        calculated. If None, the Index of the max output feauture is selected.
                    **kwargs: additional, attribution method specific arguments

                Returns:
                    attribution_scores (torch.Tensor)
                    target_label_index (int): Equals the input argument, except the input argument is None. In this case, the
                        index of the maximum output feature is returned.
            apply_log (bool, optional): If the log should be applied: log(certainty / 1-certainty).
            masking_baseline (torch.Tensor): baseline to mask the features with, zero baseline as default
            **kwargs: additional arguments for specific attribution methods, e.g. baseline for integrated gradients. **kwargs get passed
                to the attribute callable.

        Returns:
            certainties or log_odds_of_datapoint(np.ndarray): Depending on apply_log
        """
        x = torch.clone(x) #to avoid manipulation the dataset

        target_label_index = self.model.predict(x).argmax().item()

        attribution_scores, _ = attribute(x,target_label_index=None, **kwargs)
        masking_order = torch.argsort(attribution_scores, descending=True)
        masking_order = masking_order.numpy()

        predictions_with_mask = np.zeros((16))

        for i in range(len(predictions_with_mask)):
            x[masking_order[0:i]] = masking_baseline[masking_order[0:i]]
            prediction = self.model.predict(x)
            predictions_with_mask[i] = prediction[target_label_index]

        if apply_log:
            counter_propability = 1 - predictions_with_mask
            log_odds = np.log(predictions_with_mask + 1**(-16) / counter_propability)
            return log_odds
        else:
            return predictions_with_mask
    
    def get_random_references_of_datapoint(
            self,
            x: torch.Tensor,
            apply_log : bool = True,
            masking_baseline = torch.zeros(16),
            **kwargs
            ) -> np.ndarray:
        """
        Calculates a random reference of a datapoint by randomly choosing the masking order.

        Args:
            x (torch.Tensor): datapoint for which the reference gets calculated.
            apply_log (bool, optional): If the log should be applied: log(certainty / 1-certainty).
            masking_baseline (torch.Tensor): baseline to mask the features with, zero baseline as default

        Returns:
            certainties or log_odds
        """
        x = torch.clone(x) #to avoid manipulation the dataset
        
        target_label_index = self.model.predict(x).argmax().item()
        # print("Model prediction: " + str(self.model.predict(x)))
        random_masking_order = np.random.choice(a=16, size=16, replace=False)
        
        predictions_with_random_mask = np.zeros((16))
        # print("initial x: " + str(x))

        for i in range(len(predictions_with_random_mask)):
            x[random_masking_order[0:i]] = masking_baseline[random_masking_order[0:i]]
            # print("Masked input: " + str(x))
            prediction = self.model.predict(x)
            # print("Target-label-index: " + str(target_label_index))
            predictions_with_random_mask[i] = prediction[target_label_index]
            # print("Prediction: " + str(prediction))

        if apply_log:
            counter_propability = 1 - predictions_with_random_mask
            log_odds = np.log(predictions_with_random_mask + 1**(-16) / counter_propability)
            return log_odds
        
        else:
            return predictions_with_random_mask

    
    def get_log_odds_of_dataset(
            self,
            dataset: torch.utils.data.Dataset,
            attribute,
            apply_log: bool = True,
            masking_baseline = torch.zeros(16),
            **kwargs
        ) -> tuple[np.ndarray, #log-odds
              np.ndarray, #mean of log_odds
              np.ndarray, #max of log_odds
              np.ndarray]:  #min of log_odds

        """
        Calculates the log odds of each datapoint in a dataset and mean, max and min of them. 
            Designed generically so it can work with different attribution methods. To work with an attribution method,
            it needs to implement the attribute callable (see args below).

        Args:
            dataset (torch.utils.data.Dataset): Dataset for which the log odds get calculated.
            attribute (Callable): Calculates the attribution scores for an input and target label index.
                Args:
                    x (torch.Tensor): Input for which the attribution scores are calculated
                    target_label_index (int): Index of the output feature for which the attribution scores are 
                        calculated. If None, the Index of the max output feauture is selected.
                    **kwargs: additional, attribution method specific arguments
                    
                Returns:
                    attribution_scores (torch.Tensor)
                    target_label_index (int): Equals the input argument, except the input argument is None. In this case, the
                        index of the maximum output feature is returned.
            apply_log (bool, optional): If the log should be applied: log(certainty / 1-certainty).
            masking_baseline (torch.Tensor): baseline to mask the features with, zero baseline as default
            **kwargs: additional arguments for specific attribution methods, e.g. baseline for integrated gradients. **kwargs get passed
                to the attribute callable.
        """
        log_odds = np.zeros((len(dataset),16))

        for i in tqdm(range(len(log_odds))):
            log_odds[i] = self.get_log_odds_of_datapoint(dataset[i][0],attribute=attribute,apply_log=apply_log, masking_baseline=masking_baseline, **kwargs)

        #mean, max and min calculation
        mean = log_odds.mean(axis=0)

        log_odds_sums = log_odds.sum(axis=1)

        max_index = log_odds_sums.argmax()
        max = log_odds[max_index]

        min_index = log_odds_sums.argmin()
        min = log_odds[min_index]



        return log_odds, mean, max, min
    
    def get_random_references_of_dataset(
            self,
            dataset: torch.utils.data.Dataset,
            apply_log : bool = True,
            masking_baseline = torch.zeros(16),
            **kwargs
        ) -> tuple[np.ndarray, np.ndarray] :
        """
        Calculates a random reference of each datapoint in a dataset and the mean of them. The masking order is choosen ramdomly for each datapoint. 
            Designed generically so it can work with different attribution methods. To work with an attribution method,
            it needs to implement the attribute callable (see args below).

        Args:
            dataset (torch.utils.data.Dataset): Dataset for which the baseline gets calculated.
            apply_log (bool, optional): If the log should be applied: log(certainty / 1-certainty).
            masking_baseline (torch.Tensor): baseline to mask the features with, zero baseline as default

        Returns:
            random references (np.ndarray)
            mean (np.ndarray): mean of references
        """

        random_references = np.zeros((len(dataset), 16))

        for i in tqdm(range(len(random_references))):
            random_references[i] = self.get_random_references_of_datapoint(dataset[i][0], apply_log=apply_log, masking_baseline=masking_baseline, **kwargs)
            

        mean = random_references.mean(axis=0)

        return random_references, mean

    def visualize_log_odds_of_dataset(
            self,
            dataset: torch.utils.data.Dataset,
            attribute,
            title, 
            apply_log: bool = True,
            masking_baseline = torch.zeros(16),
            **kwargs
        ) -> None:
        """
        Calculates the mean, max and min log odds of each datapoint in a dataset and plots them. 
            Designed generically so it can work with different attribution methods. To work with an attribution method,
            it needs to implement the attribute callable (see args below).

        Args:
            dataset (torch.utils.data.Dataset): Dataset for which the log odds get calculated.
            attribute (Callable): Calculates the attribution scores for an input and target label index.
                Args:
                    x (torch.Tensor): Input for which the attribution scores are calculated
                    target_label_index (int): Index of the output feature for which the attribution scores are 
                        calculated. If None, the Index of the max output feauture is selected.
                    **kwargs: additional, attribution method specific arguments
            
                Returns:
                    attribution_scores (torch.Tensor)
                    target_label_index (int): Equals the input argument, except the input argument is None. In this case, the
                        index of the maximum output feature is returned.
            title (String): title of plot visualizing the log odds
            apply_log (bool, optional): If the log should be applied: log(certainty / 1-certainty).
            masking_baseline (torch.Tensor): baseline to mask the features with, zero baseline as default
            **kwargs: additional arguments for specific attribution methods, e.g. baseline for integrated gradients. **kwargs get passed
                to the attribute callable.
        """

        dataset_copy = copy.deepcopy(dataset)
        log_odds, mean, max, min = self.get_log_odds_of_dataset(dataset_copy,attribute,apply_log,masking_baseline,**kwargs)

        dataset_copy = copy.deepcopy(dataset)
        random_references, random_references_mean = self.get_random_references_of_dataset(dataset=dataset_copy,apply_log=apply_log,masking_baseline=masking_baseline, **kwargs)

        _visualize_log_odds(title, log_odds, mean, max, min, random_references_mean,apply_log)