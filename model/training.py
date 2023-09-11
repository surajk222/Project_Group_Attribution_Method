from model.models import NeuralNetwork
import torch
from torch.utils.data import DataLoader
from data.datasets import DryBean
from data.util.utils import DatasetMode
from model.util.visualisation import visualise_loss_and_accuracy
from tqdm import tqdm

def train_model(
        hidden_layers: list[int],
        num_epochs: int = 8,
        lr: float = 0.001
         )->tuple[
             torch.nn.Module,
             list[float],
             list[float],
             list[float],
             list[float]
         ]:
    """
    This method instantiates a NeuralNetwork with the given hidden_layers and trains it on the Dry Beans Dataset. 
    It returns the model and four performance metrics. The metrics are calculated before each epoch and after the final epoch.

    Args:
        hidden_layers (list[int]): list of the neuron-numbers of the hidden layers.
        num_epochs (int): Number of epochs the network should be trained.
        lr (int): The learning rate.

    returns:
        model (torch.nn.Module): The trained network on the Dry Beans Dataset.
        test_loss_array (list[float]): The average test-loss per epoch.
        test_accuracy_array (list[float]): The average test-accuracy per epoch.
        train_loss_array (list[float]): The average train-loss per epoch.
        train_accuracy_array (list[float]): The average train-accuracy per epoch.
    
    """


    train_dataset = DryBean(mode=DatasetMode.TRAIN)
    test_dataset = DryBean(mode=DatasetMode.TEST)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    torch.manual_seed(42)
    model = NeuralNetwork(hidden_layers=hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss_array, test_accuracy_array, train_loss_array, train_accuracy_array = [], [], [], []

    def train_and_test_one_epoch(train: bool):
        #test-loss and accuracy
        model.eval()
        test_loss, test_accuracy = 0,0
        with torch.no_grad():
            for test_inputs, test_labels in test_dataloader:
                test_predictions = model(test_inputs)
                test_loss += loss_fn(test_predictions, test_labels).item()
                test_accuracy += (test_predictions.argmax(1) == test_labels.argmax(1)).type(torch.float).sum().item()

        test_loss = test_loss / len(test_dataloader)
        test_accuracy = test_accuracy / len(test_dataloader.dataset)

        #training-loss and accuracy and training
        train_loss, train_accuracy = 0,0
        for train_inputs, train_labels in train_dataloader:
            #train error
            model.eval()
            with torch.no_grad():
                train_predictions = model(train_inputs)
                train_loss += loss_fn(train_predictions, train_labels).item()
                train_accuracy += (train_predictions.argmax(1) == train_labels.argmax(1)).type(torch.float).sum().item()

            if train:
                #training
                model.train()
                optimizer.zero_grad()
                train_predictions = model(train_inputs)
                loss = loss_fn(train_predictions, train_labels)
                loss.backward()
                optimizer.step()
            
        train_loss = train_loss / len(train_dataloader)
        train_accuracy = train_accuracy / len(train_dataloader.dataset)
        test_loss_array.append(test_loss)
        test_accuracy_array.append(test_accuracy)
        train_loss_array.append(train_loss)
        train_accuracy_array.append(train_accuracy)
        #print("Test-Loss: " + str(test_loss))
        #print("Test-Accuracy: " + str(test_accuracy))
        #print("Train-Loss: " + str(train_loss))
        #print("Train-Accuracy: " + str(train_accuracy))

    for epoch in tqdm(range(num_epochs)):
        train_and_test_one_epoch(train=True)
        

    #to get the train and test errors as well as accuracy after the last train epoch
    train_and_test_one_epoch(train=False)
    print("Final metrics: ")
    print("Test-Loss: " + str(test_loss_array[-1]))
    print("Test-Accuracy: " + str(test_accuracy_array[-1]))
    print("Train-Loss: " + str(train_loss_array[-1]))
    print("Train-Accuracy: " + str(train_accuracy_array[-1]))
    print(sum(param.numel() for param in model.parameters()))
    print(model)

    return model, test_loss_array, test_accuracy_array, train_loss_array, train_accuracy_array

def train_model_and_visualize(
        hidden_layers: list[int],
        num_epochs: int = 8,
        lr: float = 0.001
        ) -> torch.nn.Module:
    model, test_loss_array, test_accuracy_array, train_loss_array, train_accuracy_array = train_model(
        hidden_layers=hidden_layers,
        num_epochs=num_epochs,
        lr=lr)
    
    visualise_loss_and_accuracy(
        train_accuracy=train_accuracy_array,
        train_loss=train_loss_array,
        validation_accuracy=test_accuracy_array,
        validation_loss=test_loss_array
    )

    return model