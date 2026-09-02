# Advanced Machine Learning Project: Federated Learning Under the Lens of Task Arithmetic

This project investigates fine-tuning of **DINO ViT-S/16** on **CIFAR-100**, with a focus on federated learning and task arithmetic.

## Training Scripts

The project provides three training scripts:

- `train.py`: standard fine-tuning.
- `train_federated.py`: fine-tuning in a simulated federated learning environment.
- `train_federated_sparse.py`: sparse fine-tuning in a simulated federated learning environment.

## Starting a Run

Instructions to start a new training run:

```bash
python train.py start --help
python train_federated.py start --help
python train_federated_sparse.py start --help
```

The `--help` option lists all the available configuration parameters.

Each run produces:

- a checkpoint (`.pth`);
- a `<run_name>_log.csv` file containing the training logs;
- a `<run_name>.json` file containing the run configuration and details;
- a plot of the training loss;
- a plot of the training accuracy.

## Resuming a Run

Instructions to resume an existing training run:

```bash
python train.py resume --help
python train_federated.py resume --help
python train_federated_sparse.py resume --help
```

A valid checkpoint must exist in the configured `checkpoint_dir` in order to resume a run.

When resuming a run, the original configuration is retained, **except for the number of training epochs and the patience**. These two parameters can be overridden through `config.yaml` or via the command line.

## Evaluation
```bash
python eval.py <checkpoint>
```
eval.py returns the test loss and accuracy of the model state corresponding to the best epoch/round in terms of validation accuracy

## Configuration

Training configuration can be specified either through `config.yaml` or via command-line arguments.

Command-line arguments can be used to override the corresponding values in the configuration file.

## Experiments

The `notebook.ipynb` notebook contains instructions for running all the experiments and reproducing the project results.
