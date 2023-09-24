from torch import nn
import torch
import copy

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
            y_index = y.argmax().item()
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
    

class AutoBaselineNetwork(nn.Module):
    """
    Neurol Network for autobaseline calculation. This is one part of the Network to calculate the Uniform Output Baseline (see CombinedBaselineNetwork).
    """

    def __init__(self, initial_baseline : torch.Tensor):
        """
        Args:
            initial_baseline (torch.Tensor): Baseline to start from.    
        """
        super().__init__()

        self.model = nn.Linear(1,16,bias=False)
        initial_baseline_with_grad = copy.deepcopy(initial_baseline).requires_grad_(True)
        with torch.no_grad():
            self.model.weight[:,0] = initial_baseline_with_grad
    def forward(self,x):
        return self.model(x)
    
class CombinedBaselineNetwork(nn.Module):
    """
    Neural Network to determine a Uniform Output Baseline. In this network we are only training the Parameters of the autobaseline_model.
    """

    def __init__(self, dry_beans_model : NeuralNetwork, initial_baseline : torch.Tensor):
        super().__init__()

        self.autobaseline_model = AutoBaselineNetwork(initial_baseline=initial_baseline)
        self.dry_beans_model = copy.deepcopy(dry_beans_model)
        self.dry_beans_model.requires_grad_(False)

    def forward(self,x):
        autobaseline_model_output = self.autobaseline_model(x)
        dry_beans_model_output = self.dry_beans_model.predict(autobaseline_model_output, detach=False)

        return autobaseline_model_output,dry_beans_model_output
    
    def get_autobaseline(self):
        return self.autobaseline_model(torch.ones((1)))
    
def combined_model_loss_fn(
        autobaseline : torch.Tensor, 
        initial_baseline : torch.Tensor, 
        actual_model_output : torch.Tensor, 
        target_model_output : torch.Tensor, 
        baseline_error_weight : float):
    
    """
        Loss Function for the CombindedBaselineNetwork.

        Args:
            autobaseline (torch.Tensor): Baseline computed from the AutoBaselineNetwork.
            initial_baseline (torch.Tensor): Initial baseline set before training.
            actual_model_output (torch.Tensor): Model Output of the autobaseline.
            target_model_output (torch.Tensor): Target Model Output of the Baseline. In our case the Uniform distribution.
            baseline_error_weight (float): weight of the baseline error.
    """
    
    l_baseline = torch.nn.functional.l1_loss(autobaseline,initial_baseline)
    
    #l_model_output = torch.max(torch.abs(actual_model_output - target_model_output))
    l_model_output = torch.nn.functional.l1_loss(actual_model_output, target_model_output)

    return baseline_error_weight * l_baseline + (1-baseline_error_weight) * l_model_output
