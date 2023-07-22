from torch import nn
import torch

class NeuralNetwork(nn.Module):
    """
    Neural Network for the dry beans dataset.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,7)
            #Since we use nn.CrossEntropyLoss() as our loss function, there is no need for a softmax-activation function.
        )

    def forward(self,x):
        return self.model(x)
    
    def evaluate(self,x):
        self.eval()
        with torch.no_grad():
            y = self.forward(x)
            y = nn.functional.softmax(y, dim=0)
        self.train()
        return y
    
    def get_gradients_with_respect_to_inputs(
            self,
            x: torch.Tensor,
            target_label_idx: int = None,
            ) -> torch.Tensor:

        """
        This method is meant to be a callable for the integrated gradients class. It
        computes the model output with respect to the model input.

        Args:
            x (torch.Tensor): A batch of inputs at which the gradients of the model output are calculated.
                x must have the dimensions num_batches x input_dim
            target_label_idx (optional, int): index of the output feature for which the gradient should be calculated. This is
                usually the index of the output feature with the highest score I.
                If the target_label_index is not specified, the index of the output feature with the highest value
                will be selected.

        Returns:
            gradients (torch.Tensor): Gradients of the model output with respect to the input x.
        """
        if target_label_idx==None:
            original_x = x[-1]
            target_label_idx = self(original_x).argmax().sum().item()

        gradients = torch.zeros_like(x)

        for i,input in enumerate(x):
            input.requires_grad = True
            self.eval()
            model_prediction = self(input)[target_label_idx]
            model_prediction.backward(gradient=torch.ones_like(model_prediction))
            gradients[i,:] = input.grad
        
        return gradients

        
