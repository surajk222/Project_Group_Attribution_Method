from torch import nn
import torch

class NeuralNetwork(nn.Module):
    """
    Neural Network for the dry beans dataset.
    """

    def __init__(self, hidden_layers: list[int]):
        """
        Args:
            hidden_layers(list[int]): list of the neuron-numbers of the hidden layers.
        """
        super().__init__()
        activation_fn = nn.ReLU()
        
        #add layers to the layer_list
        layer_list = []
        for i in range(len(hidden_layers)):
            #input-layer
            if i == 0:
                layer_list.append(nn.Linear(16,hidden_layers[i]))
                layer_list.append(activation_fn)
            else:
                layer_list.append(nn.Linear(hidden_layers[i-1],hidden_layers[i]))
                layer_list.append(activation_fn)
        
        #output-layer
        layer_list.append(nn.Linear(hidden_layers[i],7)) #Since we use nn.CrossEntropyLoss() as our loss function, there is no need for a softmax-activation function.

        self.model = nn.Sequential(*layer_list)

    def forward(self,x):
        return self.model(x)
    
    def predict(self,
        x: torch.Tensor,
        detach: bool = True) -> torch.Tensor:
        """
        Evaluates the model and applies the softmax function to it.
        Args:
            x (torch.Tensor): Datapoint for which the model gets predicted.
            detach (bool, optional = False): If the model output should be detached from the graph.)

        Returns:
            y (torch.Tensor): Model output with softmax-function applied to it.
        """

        self.eval()

        y = self(x)

        if detach:
            return nn.functional.softmax(y).detach()
        
        else:
            return nn.functional.softmax(y)

        

    
    def get_max_feature(
            self,
            x: torch.Tensor,
            )->tuple[torch.Tensor, torch.LongTensor]:
        """
        Evaluates the model for a given input vector and returns the maximum value and its index.

        Args:
            x (torch.Tensor): Input feature fpr which the maximum should be calculated.
        
        Returns:
            y (torch.Tensor): Maximum of the model output for the given input.
            y_index (torch.LongTensor): Argmax of the model output for the given input.

        """
        self.eval()
        with torch.no_grad():
            y = self.predict(x)
            y_index = y.argmax()
            y = y.max()
        return y, y_index
    
    def get_gradients_with_respect_to_inputs(
            self,
            x: torch.Tensor,
            target_label_idx: int = None,
            ) -> (torch.Tensor, int):

        """
        This method is meant to be a callable for the integrated gradients class. It
        computes the gradient of the model output with respect to the model input.

        Args:
            x (torch.Tensor): A batch of inputs at which the gradients of the model output are calculated.
                x must have the dimensions num_batches x input_dim
            target_label_idx (optional, int): index of the output feature for which the gradient should be calculated. This is
                usually the index of the output feature with the highest score I.
                If the target_label_index is not specified, the index of the output feature with the highest value
                will be selected.

        Returns:
            gradients (torch.Tensor): Gradients of the model output with respect to the input x.
            target_label_index (int): Equals the input argument, except the input argument is None. In this case, the
                index of the maximum output feature is returned.
        """
        if target_label_idx==None:
            original_x = x[-1]
            target_label_idx = self(original_x).argmax().sum().item()

        gradients = torch.zeros_like(x)

        for i,input in enumerate(x):
            input.requires_grad = True
            self.eval()
            model_prediction = self.predict(input, detach=False)[target_label_idx]
            model_prediction.backward(gradient=torch.ones_like(model_prediction))
            gradients[i,:] = input.grad
        
        return gradients, target_label_idx