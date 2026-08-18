import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def read(filename):
    train_loss = []
    val_loss = []
    val_acc = []

    with open(filename, newline='') as f:
        reader = csv.DictReader(f)

        for row in reader:
            train_loss.append(float(row['train_loss']))

            # Empty validation cells become NaN
            val_loss.append(
                float(row['val_loss']) if row['val_loss'].strip()
                else np.nan
            )

            val_acc.append(
                float(row['val_acc']) if row['val_acc'].strip()
                else np.nan
            )

    return train_loss, val_loss, val_acc

def plot_training(run_name, logs_dir, save_dir=None, federated=False):
    train_loss, val_loss, val_acc = read(f"{logs_dir}/{run_name}_log.csv")
    epochs = list(range(1, len(train_loss) + 1))
    x_label = 'Rounds' if federated else 'Epochs'

    # Loss
    plt.figure()
    if federated:
        plt.plot(epochs, train_loss, label='Train Loss')
    else:
        plt.plot(epochs, train_loss, marker='o', markersize=4, label='Train Loss')

    valid = ~np.isnan(val_acc)
    plt.plot(np.array(epochs)[valid], np.array(val_loss)[valid], marker='o', markersize=4, label='Val Loss')

    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.title('Loss')
    if not federated:
        ticks = list(range(len(epochs), 0, -max(1, len(epochs) // 10)))[::-1]
        plt.xticks(ticks)
    plt.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{run_name}_loss_plot")
    else:
        plt.show()

    plt.close()

    # Accuracy
    plt.figure()

    valid = ~np.isnan(val_acc)
    plt.plot(np.array(epochs)[valid], np.array(val_acc)[valid], marker='o', markersize=4,label='Val Accuracy')

    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    if not federated:
        plt.xticks(ticks)
    plt.legend()
    plt.tight_layout()

    if save_dir:
        plt.savefig(f"{save_dir}/{run_name}_acc_plot")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    plot_training("", "centralized_model/logs/", "centralized_model/plots/")
