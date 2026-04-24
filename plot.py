import os
import matplotlib.pyplot as plt
import numpy as np


def read(filepath):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1, ndmin=2)
    return data[:, 1], data[:, 2], data[:, 3], data[:, 4]

def plot_training(run_name, logs_dir, save_dir=None):
    train_loss, val_loss, train_acc, val_acc = read(f"{logs_dir}/{run_name}_log.csv")
    epochs = range(1, len(train_loss) + 1)

    _, (loss_plt, acc_plt) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    loss_plt.plot(epochs, train_loss, label='Train Loss')
    loss_plt.plot(epochs, val_loss, label='Val Loss')
    loss_plt.set_xlabel('Epochs')
    loss_plt.set_ylabel('Loss')
    loss_plt.set_title('Loss')
    loss_plt.legend()

    # Accuracy
    acc_plt.plot(epochs, train_acc, label='Train Accuracy')
    acc_plt.plot(epochs, val_acc, label='Val Accuracy')
    acc_plt.set_xlabel('Epochs')
    acc_plt.set_ylabel('Accuracy')
    acc_plt.set_title('Accuracy')
    acc_plt.legend()

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{run_name}_plot")

    plt.show()


if __name__ == '__main__':
    plot_training("debug_20260424_185552", "centralized_model/logs/", "../centralized_model/plots/")
