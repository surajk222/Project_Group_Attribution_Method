import models
import torch

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
            test_loss = loss_fn(test_predictions, test_labels)
            test_accuracy += (test_predictions.argmax(1) == test_labels).type(torch.float).sum().item()

    test_loss = test_loss / len(test_dataloader)
    test_accuracy = test_accuracy / len(test_dataloader.dataset)

    #training-error and training
    model.train()
    train_loss_array = []
    for i, train_data in enumerate(train_dataloader):
        #training
        train_inputs, train_labels = train_data
        optimizer.zero_grad()
        train_predictions = model(train_inputs)
        train_loss = loss_fn(train_predictions, train_labels)
        train_loss.backward()

        optimizer.step()

        #train error
        with torch.no_grad:
            train_loss = loss_fn(train_predictions, train_loss)
            train_loss_array.append(train_loss)
            train_accuracy += (train_predictions.argmax(1) == train_labels).type(torch.float).sum().item()
        
    train_accuracy = train_accuracy / len(train_dataloader.dataset)

    return test_loss, test_accuracy, train_loss_array, train_accuracy