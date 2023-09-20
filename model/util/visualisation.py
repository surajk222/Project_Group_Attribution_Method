from matplotlib import pyplot as plt
import numpy as np

def visualise_loss_and_accuracy(
        train_loss: list[float],
        validation_loss: list[float],
        train_accuracy: list[float],
        validation_accuracy: list[float]
    ) -> None:
    """
    Visualises train- and validation loss- and accuracy during training.

    Args:
        train_loss(list[float]): List of the train losses.
        validation_loss(list[float]): List of the validation losses.
        train_accuracy(list[float]): List of train accuracies.
        validation_accuracy(list[float]): List of validation accuracies.
    """
    x = np.linspace(0, len(train_loss),num=len(train_loss), endpoint=False)

    fig, (ax1,ax2) = plt.subplots(2)

    
    ax1.plot(x,train_loss, label="Train")
    ax1.plot(x,validation_loss, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()

    ax2.plot(x, train_accuracy, label="Train")
    ax2.plot(x,validation_accuracy, label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()
