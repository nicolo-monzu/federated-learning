import os
import matplotlib.pyplot as plt
import numpy as np


def read(filepath):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1, ndmin=2)
    return data[:, 1], data[:, 2], data[:, 3]

def plot_training(run_name, logs_dir, save_dir=None):
    train_loss, val_loss, val_acc = read(f"{logs_dir}/{run_name}_log.csv")
    epochs = range(1, len(train_loss) + 1)


    # Loss
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{run_name}_loss_plot")
    plt.show()

    # Accuracy
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/{run_name}_acc_plot")
    plt.show()


if __name__ == '__main__':
    plot_training("", "centralized_model/logs/", "centralized_model/plots/")
