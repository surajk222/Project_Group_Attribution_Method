from model.models import NeuralNetwork
import torch
from torch.utils.data import DataLoader
from data.datasets import DryBean

def train_model():
    train_dataset = DryBean(train=True)
    test_dataset = DryBean(train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    model = NeuralNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    num_epochs = 8
    test_loss_array, test_accuracy_array, train_loss_array, train_accuracy_array = [], [], [], []

    for epoch in range(num_epochs):
        model.eval()
        test_loss, test_accuracy = 0,0
        with torch.no_grad():
            for test_inputs, test_labels in test_dataloader:
                test_predictions = model(test_inputs)
                test_loss += loss_fn(test_predictions, test_labels).item()
                test_accuracy += (test_predictions.argmax(1) == test_labels.argmax(1)).type(torch.float).sum().item()

        test_loss = test_loss / len(test_dataloader)
        test_accuracy = test_accuracy / len(test_dataloader.dataset)

        #training-error and training
        train_loss, train_accuracy = 0,0
        for train_inputs, train_labels in train_dataloader:
            #training
            model.train()
            optimizer.zero_grad()
            train_predictions = model(train_inputs)
            loss = loss_fn(train_predictions, train_labels)
            loss.backward()
            optimizer.step()

            #train error
            model.eval()
            with torch.no_grad():
                train_loss += loss_fn(train_predictions, train_labels).item()
                train_accuracy += (train_predictions.argmax(1) == train_labels.argmax(1)).type(torch.float).sum().item()
            
        train_loss = train_loss / len(train_dataloader)
        train_accuracy = train_accuracy / len(train_dataloader.dataset)
        test_loss_array.append(test_loss)
        test_accuracy_array.append(test_accuracy)
        train_loss_array.append(train_loss)
        train_accuracy_array.append(train_accuracy)
        print("Test-Loss: " + str(test_loss))
        print("Test-Accuracy: " + str(test_accuracy))
        print("Train-Loss: " + str(train_loss))
        print("Train-Accuracy: " + str(train_accuracy))

    return model, test_loss_array, test_accuracy_array, train_loss_array, train_accuracy_array

    

def train_one_epoch(model : torch.nn.Module, train_dataloader, test_dataloader, loss_fn, optimizer):
    """
    Performs evaluation on test data and training on training dataset. Both test and training error and accuracy are recorded.
    """

    #test-error
    model.eval()
    test_loss, test_accuracy = 0,0
    with torch.no_grad():
        for i,test_data in enumerate(test_dataloader):
            test_inputs, test_labels = test_data
            test_predictions = model(test_inputs)
            test_loss += loss_fn(test_predictions, test_labels)
            test_accuracy += (test_predictions.argmax(1) == test_labels.argmax(1)).type(torch.float).sum().item()

    test_loss = test_loss / len(test_dataloader)
    test_accuracy = test_accuracy / len(test_dataloader.dataset)

    #training-error and training
    train_loss, train_accuracy = 0,0
    for i, train_data in enumerate(train_dataloader):
        #training
        model.train()
        train_inputs, train_labels = train_data
        optimizer.zero_grad()
        train_predictions = model(train_inputs)
        train_loss = loss_fn(train_predictions, train_labels)
        train_loss.backward()

        optimizer.step()

        #train error
        model.eval()
        with torch.no_grad():
            train_loss += loss_fn(train_predictions, train_labels)
            train_accuracy += (train_predictions.argmax(1) == train_labels.argmax(1)).type(torch.float).sum().item()
        
    train_loss = train_loss / len(train_dataloader)
    train_accuracy = train_accuracy / len(train_dataloader.dataset)

    return test_loss, test_accuracy, train_loss, train_accuracy, model

