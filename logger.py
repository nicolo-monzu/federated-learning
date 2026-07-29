import json
import csv
import os

class Logger:
    def __init__(self, directory, run_name):
        os.makedirs(directory, exist_ok=True)
        self.log_path = f'{directory}/{run_name}_log.csv'
        self.det_path = f'{directory}/{run_name}.json'
        self.run = None
        self.logs = None

    def get_run(self):
        return self.run

    def get_best_accuracy(self):
        return self.run['best_accuracy']

    def start(self, run):
        self.run = run
        self.logs = [['epoch', 'train_loss', 'val_loss', 'val_acc', 'lr']]

        # Create json file
        with open(self.det_path, 'w') as f:
            json.dump(self.run, f, indent=4, ensure_ascii=False)

        # Create log file
        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(self.logs[0])


    def state_dict(self):
        return {
            'run': self.run,
            'log': self.logs,
        }

    def resume(self, state_dict):
        self.run = state_dict['run']
        self.logs = state_dict['log']

        # Restore log file
        with open(self.log_path + '.tmp', 'w', newline='') as f:
            csv.writer(f).writerows(self.logs)
        os.replace(self.log_path + '.tmp', self.log_path)

        # Restore json file
        with open(self.det_path + '.tmp', 'w') as f:
            json.dump(self.run, f, indent=4, ensure_ascii=False)
        os.replace(self.det_path + '.tmp', self.det_path)


    def log(self, epoch, train_loss, val_loss, val_acc, lr):
        self.logs.append([epoch, train_loss, val_loss, val_acc, lr])

        self.run['total_epochs'] = epoch
        if val_acc > self.run['best_accuracy']:
            self.run['best_epoch'] = epoch
            self.run['best_accuracy'] = val_acc

        # Update log file
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow(self.logs[-1])

        # Update json file
        with open(self.det_path + '.tmp', 'w') as f:
            json.dump(self.run, f, indent=4, ensure_ascii=False)
        os.replace(self.det_path + '.tmp', self.det_path)

