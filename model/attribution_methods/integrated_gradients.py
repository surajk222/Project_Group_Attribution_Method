from typing import Callable
import torch

class IntegratedGradients():
    """
    Implementation of the Integragrated Gradients method from Sundarajan, Taly and Yan.
    See https://arxiv.org/abs/1703.01365 for the original paper.
    """

    def __init__(
            self,
            get_gradients: Callable[[torch.Tensor, int],torch.Tensor]
            ) -> None:
        """
        Args:
            get_gradients (Callable): Calculates the gradients of the models output with respect to the input.
                Args:
                    x (torch.Tensor): Input for which the gradient should be calculated.
                    target_label_index (int): index of the output feature for which the gradient gets calculated.
        """
        self.get_gradients = get_gradients

    def attribute(
            self,
            x: torch.Tensor,
            target_label_index: int = None,
            baseline: torch.Tensor = None,
            n_steps: int = 50
            ) -> torch.Tensor:
        """
        Performs the calculation of the attribution scores with the Integrated Gradients method.

        Args:
            x (torch.Tensor): Input for which the integrated gradients must be calculated.
            target_label_index (optional, int): Index of the output feature for which the integrated gradients must be calculated.
                If the target_label_index is not specified, the Integrated Gradients of the output feature with the highest value
                gets calculated.
            baseline (optional, torch.Tensor): The baseline input. If None, the baseline is set to a zero-vector. Must have the same shape as x.
                Defaults to None.
            n_steps (optional, int): Number of steps for the integral approximation. Defaults to 50.
        
        Returns:
            integrated_gradients (torch.Tensor): The Integrated Gradients of the model output for the provided output feature to the input.
        """
        if baseline == None:
            baseline = x*0
        assert(baseline.shape == x.shape)

        straightline_path = torch.vstack([baseline + float(i)/n_steps * (x-baseline) for i in range(1, n_steps + 1)])
        gradients = self.get_gradients(straightline_path, target_label_index)

        avg_gradients = gradients.mean(dim=0)
        integrated_gradients = (x-baseline) * avg_gradients

        return integrated_gradients