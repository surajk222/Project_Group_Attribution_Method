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