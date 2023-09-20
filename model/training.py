from model.models import NeuralNetwork, CombinedBaselineNetwork, combined_model_loss_fn
import torch
from torch.utils.data import DataLoader
from data.datasets import DryBean
from data.util.utils import DatasetMode
from model.util.visualisation import visualise_loss_and_accuracy
from tqdm import tqdm
import torchmetrics
import matplotlib.pyplot as plt

def train_model(
        hidden_layers: list[int],
        num_epochs: int = 10,
        lr: float = 0.001,
        batch_size= 8
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
    test_dataset = DryBean(mode=DatasetMode.VALIDATION)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=7,average="macro")
    test_accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=7,average="macro")
    test_accuracy_per_class = torchmetrics.Accuracy(task="multiclass",num_classes=7,average=None)




    torch.manual_seed(42)
    model = NeuralNetwork(hidden_layers=hidden_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss_array, test_accuracy_array, train_loss_array, train_accuracy_array = [], [], [], []

    def train_and_test_one_epoch(train: bool):
        #test-loss and accuracy
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for test_inputs, test_labels in test_dataloader:
                test_predictions = model(test_inputs)
                test_loss += loss_fn(test_predictions, test_labels).item()
                test_accuracy.update(test_predictions, test_labels.argmax(dim=1))
                test_accuracy_per_class.update(test_predictions,  test_labels.argmax(dim=1))

        test_loss = test_loss / len(test_dataloader)

        #training-loss and accuracy and training
        train_loss = 0
        for train_inputs, train_labels in train_dataloader:
            #train error
            model.eval()
            with torch.no_grad():
                train_predictions = model(train_inputs)
                train_loss += loss_fn(train_predictions, train_labels).item()
                train_accuracy.update(train_predictions, train_labels.argmax(dim=1))
        for train_inputs, train_labels in train_dataloader:
            if train:
                #training
                model.train()
                optimizer.zero_grad()
                train_predictions = model(train_inputs)
                loss = loss_fn(train_predictions, train_labels)
                loss.backward()
                optimizer.step()
            
        train_loss = train_loss / len(train_dataloader)

        test_loss_array.append(test_loss)
        test_accuracy_array.append(test_accuracy.compute().item())

        train_loss_array.append(train_loss)
        train_accuracy_array.append(train_accuracy.compute().item())
        #test_accuracy_per_class.plot()
        plt.show()

    for epoch in tqdm(range(num_epochs)):
        train_and_test_one_epoch(train=True)
        

    #to get the train and test errors as well as accuracy after the last train epoch
    train_and_test_one_epoch(train=False)
    print("Final metrics: ")
    print(f"Validation-Loss: {test_loss_array[-1]: 0.3f}")
    print(f"Validation-Accuracy: {test_accuracy_array[-1]: 0.1%}")
    print(f"train-Loss: {train_loss_array[-1]: 0.3f}")
    print(f"train-Accuracy: {train_accuracy_array[-1]: 0.1%}")
    total_params = sum(param.numel() for param in model.parameters())
    print("# Parameters: " + str(total_params))
    print(model)

    return model, test_loss_array, test_accuracy_array, train_loss_array, train_accuracy_array

def train_model_and_visualize(
        hidden_layers: list[int],
        num_epochs: int = 10,
        lr: float = 0.001,
        batch_size= 8
        ) -> torch.nn.Module:
    model, test_loss_array, test_accuracy_array, train_loss_array, train_accuracy_array = train_model(
        hidden_layers=hidden_layers,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size)


    visualise_loss_and_accuracy(
        train_accuracy=train_accuracy_array,
        train_loss=train_loss_array,
        validation_accuracy=test_accuracy_array,
        validation_loss=test_loss_array
    )

    return model




def train_autobaseline(
        dry_beans_model : NeuralNetwork, 
        initial_baseline : torch.Tensor = None, 
        num_epochs : int = 300,
        baseline_error_weight = 0.4):
    
    if initial_baseline == None:
        torch.manual_seed(48)
        initial_baseline = torch.FloatTensor(16).uniform_(0.005,0.01)

    dataset_len = 1000
    x = torch.ones((dataset_len, 1)) #dataset_len x 1
    y_baseline = torch.unsqueeze(initial_baseline,0).repeat(dataset_len,1) #dataset_len x len(initial_baseline)
    y_model_output = torch.ones((1000,7)) * (1/7)

    dataset = torch.utils.data.TensorDataset(x,y_baseline,y_model_output)

    dataloader = DataLoader(dataset=dataset,batch_size=32)

    combined_baseline_model = CombinedBaselineNetwork(dry_beans_model=dry_beans_model,initial_baseline=initial_baseline)

    optimizer = torch.optim.Adam(params=combined_baseline_model.autobaseline_model.parameters(), lr=0.001)


    
    for epoch in tqdm(range(num_epochs)):
        for x, y_baseline,y_model_output in dataloader:
            optimizer.zero_grad()
            autobaseline, model_output = combined_baseline_model(x)

            loss = combined_model_loss_fn(
                autobaseline=autobaseline,
                initial_baseline=y_baseline,
                actual_model_output=model_output,
                target_model_output=y_model_output,
                baseline_error_weight=baseline_error_weight
            )

            loss.backward()
            optimizer.step()

    print("autobaseline: " + str(combined_baseline_model.get_autobaseline()))
    print("prediction: " + str(dry_beans_model.predict(combined_baseline_model.get_autobaseline())))

    return combined_baseline_model.get_autobaseline().detach()