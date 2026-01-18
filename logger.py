import json
import os


class Logger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = f'{log_dir}/log.csv'
        self.det_path = f'{log_dir}/settings.json'

    def start(self, exp):
        with open(self.det_path, 'w') as f:
            json.dump(exp, f, indent=4, ensure_ascii=False)
        with open(self.log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,train_acc,val_acc\n')

    def resume(self, exp):
        #todo
        pass

    def update_best_acc(self, best_acc):
        #todo
        pass

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(self.log_path, 'a') as f:
            f.write(f'{epoch},{train_loss},{val_loss},{train_acc},{val_acc}\n')