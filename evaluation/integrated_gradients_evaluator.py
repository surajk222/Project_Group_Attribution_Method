import torch
from data.datasets import DryBean
from data.util.utils import DatasetMode
from model.attribution_methods.integrated_gradients import IntegratedGradients
from model.models import NeuralNetwork

class IntegratedGradientsEvaluator():
    """
    This class evaluates the Integrated Gradient Attribution Method.

    It contains several methods for evaluation and axoim testing.

    """

    def __init__(self, model: NeuralNetwork):
        self.model = model
        self.dataset = DryBean(mode=DatasetMode.TEST)
        self.ig = IntegratedGradients(self.model.get_gradients_with_respect_to_inputs)

    def mse_delta(
            self,
            n_steps: int = 50,
            baseline: torch.Tensor = None
    ) -> float:
        """
        This method evaluates the delta criterium, also called completeness axiom.
        It calculates the MSE of the difference between the sum of the attribution scores
        and f(x)-f(x') over the test dataset.

        Args:
            n_steps (optional, int): Number of steps for the integral approximation. Default is 50.
            baseline (optional, torch.Tensor): Baseline for the Integrated Gradients method. If no baseline is provided,
                the zero-vector is chosen as a baseline.

        Returns:
            mse_delta (float)
        """


        mse_delta = 0
        if baseline == None:
            baseline = torch.zeros_like(self.dataset[0][0])

        for i,(x,y) in enumerate(self.dataset):
            print(i)
            attributions = self.ig.attribute(x,baseline=baseline,n_steps=n_steps)

            model_prediction = self.model(x)
            model_prediction = model_prediction.max()

            delta = self.model.get_max_feature(x) - self.model.get_max_feature(baseline)
            delta = (delta.item() - attributions.sum().item())**2
            mse_delta += delta

        mse_delta /= len(self.dataset)
        return mse_delta
