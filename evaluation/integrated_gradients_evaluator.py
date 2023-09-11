import torch
from data.datasets import DryBean
from data.util.utils import DatasetMode
from model.attribution_methods.integrated_gradients import IntegratedGradients
from model.models import NeuralNetwork
from tqdm import tqdm
import numpy as np

class IntegratedGradientsEvaluator():
    """
    This class evaluates the Integrated Gradient Attribution Method.

    It contains several methods for evaluation and axoim testing.

    """

    def __init__(self, model: NeuralNetwork):
        self.model = model
        self.dataset = DryBean(mode=DatasetMode.TEST)
        self.ig = IntegratedGradients(self.model.get_gradients_with_respect_to_inputs)


    
    def completeness_deltas(
            self,
            baseline: torch.Tensor = None,
            n_steps: int = 50
        ) -> np.ndarray[float]:
        """
        Calculates for each datapoint x in self.dataset the absolute difference between:
            sum of integrated gradients attribution scores and
            f(x) - f(baseline)

        Args:
            baseline (optional, torch.Tensor): The baseline input. If None, the baseline is set to a zero-vector. Must have the same shape as x.
                Defaults to None.
            n_steps (optional, int): Number of steps for the integral approximation. Defaults to 50.
        
        Returns:
            completeness_delta_abs (np.ndarray): NumPy array for the absolute deltas of each datapoint.
        """
        if baseline == None:
            baseline = 0*self.dataset[0][0]

        completeness_deltas = []

        for i,(x,y) in tqdm(enumerate(self.dataset)):

            f_x, target_label_index = self.model.get_max_feature(x)
            f_baseline = self.model.predict(baseline)[target_label_index]

            ig_scores = self.ig.attribute(x=x,target_label_index=target_label_index, baseline=baseline, n_steps=n_steps)[0]
            ig_scores_sum = ig_scores.sum()

            delta = (f_x - f_baseline - ig_scores_sum)
            delta = delta.item()

            completeness_deltas.append(delta)


        completeness_deltas_np = np.array(completeness_deltas)

        completeness_deltas_abs = completeness_deltas_np

        return completeness_deltas_abs
    
    def completeness_deltas_statistics(
            self,
            baseline: torch.Tensor = None,
            n_steps: int = 50,
            return_results = False
        ) -> tuple[np.ndarray[float], float,float,float]:

        """
        Calculates the mean, max and min of the absolute completeness deltas and prints the results.

        Args:
            baseline (optional, torch.Tensor): The baseline input. If None, the baseline is set to a zero-vector. Must have the same shape as x.
                Defaults to None.
            n_steps (optional, int): Number of steps for the integral approximation. Defaults to 50.

        Returns:
            np.ndarray[float]: absolute varray of the completeness delta for each datapoint
            float: absolute
        """
        
        completeness_deltas_abs = self.completeness_deltas(baseline=baseline,n_steps=n_steps)

        #mean
        mean = completeness_deltas_abs.mean()

        #max
        max = completeness_deltas_abs.max()

        #min
        min = completeness_deltas_abs.min()

        print(f"Mittlere absolute Abweichung: {mean : .2e}")
        print(f"Maximum der betragsmäßigen Abweichung; {max : .2e}")
        print(f"Minimum der betragsmäßigen Abweichung: {min : .2e}")

        if return_results:
            return completeness_deltas_abs, mean, max, min

