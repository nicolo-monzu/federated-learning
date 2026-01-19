import json
import os


class Logger:
    def __init__(self, log_dir, run_name):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = f'{log_dir}/log_{run_name}.csv'
        self.det_path = f'{log_dir}/{run_name}.json'


    def start(self, exp):
        with open(self.det_path, 'w') as f:
            json.dump(exp, f, indent=4, ensure_ascii=False)
        with open(self.log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,train_acc,val_acc\n')


    def resume(self, exp):
        #todo
        pass


    def update_best_acc(self, best_acc):
        with open(self.det_path, 'r') as f:
            exp = json.load(f)

        exp['best_accuracy'] = best_acc

        with open(self.det_path+'.tmp', 'w') as f:
            json.dump(exp, f, indent=4, ensure_ascii=False)
        os.replace(self.det_path+'.tmp', self.det_path)


    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(self.log_path, 'a') as f:
            f.write(f'{epoch},{train_loss},{val_loss},{train_acc},{val_acc}\n')