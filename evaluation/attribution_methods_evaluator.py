import torch
import numpy as np

from typing import Callable

from tqdm import tqdm

from model.models import NeuralNetwork
from data.datasets import DryBean
from data.util.utils import DatasetMode
from model.attribution_methods.integrated_gradients import IntegratedGradients
from evaluation.utils.visualisation import _visualize_log_odds



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
            baseline (torch.Tensor): baseline for the Integrated Gradient Method. Does only need to be given if this method is used.
            **kwargs: additional arguments for specific attribution methods, e.g. baseline for integrated gradients. **kwargs get passed
                to the attribute callable.

        Returns:
            log_odds_of_datapoint(np.ndarray): Log odds of the given input.
        """

        attribution_scores, target_label_index = attribute(x,target_label_index=None, **kwargs)
        masking_order = torch.argsort(attribution_scores, descending=True)
        masking_order = masking_order.numpy()

        predictions_with_mask = np.zeros((16))

        for i in range(len(predictions_with_mask)):
            x[masking_order[0:i]] = 0
            prediction = self.model.predict(x)
            predictions_with_mask[i] = prediction[target_label_index]

        
        counter_propability = 1 - predictions_with_mask
        log_odds = np.log(predictions_with_mask + 1**(-16) / counter_propability)
        return log_odds
    
    def get_log_odds_of_dataset(
            self,
            dataset: torch.utils.data.Dataset,
            attribute,
            **kwargs
        ) -> [np.ndarray, #log-odds
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
            baseline (torch.Tensor): baseline for the Integrated Gradient Method. Does only need to be given if this method is used.
            **kwargs: additional arguments for specific attribution methods, e.g. baseline for integrated gradients. **kwargs get passed
                to the attribute callable.
        """
        log_odds = np.zeros((len(dataset),16))

        for i in tqdm(range(len(log_odds))):
            log_odds[i] = self.get_log_odds_of_datapoint(dataset[i][0],attribute=attribute,**kwargs)

        #mean, max and min calculation
        mean = log_odds.mean(axis=0)

        log_odds_sums = log_odds.sum(axis=1)

        max_index = log_odds_sums.argmax()
        max = log_odds[max_index]

        min_index = log_odds_sums.argmin()
        min = log_odds[min_index]



        return log_odds, mean, max, min

    def visualize_log_odds_of_dataset(
            self,
            dataset: torch.utils.data.Dataset,
            attribute,
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
            baseline (torch.Tensor): baseline for the Integrated Gradient Method. Does only need to be given if this method is used.
            **kwargs: additional arguments for specific attribution methods, e.g. baseline for integrated gradients. **kwargs get passed
                to the attribute callable.
        """

        log_odds, mean, max, min = self.get_log_odds_of_dataset(dataset,attribute,**kwargs)

        _visualize_log_odds(log_odds, mean, max, min)