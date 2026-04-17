import json
import os


def truncate_file(filepath, n):
    with open(filepath, 'r') as f:
        lines = [next(f) for _ in range(n)]

    with open(filepath+'.tmp', 'w') as f:
        f.writelines(lines)
    os.replace(filepath+'.tmp', filepath)

class Logger:
    def __init__(self, log_dir, run_name):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = f'{log_dir}/{run_name}_log.csv'
        self.det_path = f'{log_dir}/{run_name}.json'

    def get_run(self):
        return self.run

    def start(self, run):
        self.run = run
        # Create json
        with open(self.det_path, 'w') as f:
            json.dump(self.run, f, indent=4, ensure_ascii=False)
        # Create log
        with open(self.log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,train_acc,val_acc\n')


    def resume(self, epoch, best_acc):
        with open(self.det_path, 'r') as f:
            self.run = json.load(f)
        # Restore json
        self.update_best_acc(best_acc)
        # Restore log
        truncate_file(self.log_path, epoch+1)


    def update_best_acc(self, best_acc):
        self.run['best_accuracy'] = best_acc
        # Update json
        with open(self.det_path+'.tmp', 'w') as f:
            json.dump(self.run, f, indent=4, ensure_ascii=False)
        os.replace(self.det_path+'.tmp', self.det_path)


    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(self.log_path, 'a') as f:
            f.write(f'{epoch},{train_loss},{val_loss},{train_acc},{val_acc}\n')
