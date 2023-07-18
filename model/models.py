from torch import nn

class NeuralNetwork(nn.Module):
    """
    Neural Network for the dry beans dataset.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 32),
            nn.Relu,
            nn.Linear(32,16),
            nn.Relu,
            nn.Linear(16,7)
            #Since we use nn.CrossEntropyLoss() as our loss function, there is no need for a softmax-activation function.
        )

    def forward(self,x):
        return self.model(x)