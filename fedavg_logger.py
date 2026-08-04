import csv
from os import mkdir

class FEDAVGLogger:
    def __init__(self, iid_flag, rounds, local_steps, scheduler, nc=None):
        self.path = f"fedavg_model/logs/{"IID" if iid_flag else "NIID"}_r{rounds}_j{local_steps}{"_nc" + str(nc) if not iid_flag else ""}.csv"
        mkdir(self.path)
        self.hyperp = Hyperparameters(iid_flag, rounds, local_steps, nc)
        self.round_results = [RoundResults(i, self.path) for i in range(rounds)]
        self.accuracy = 0
        self.scheduler = scheduler
        self.eval_accuracy = 0
        self.eval_loss = 0
        self.weights = None
        print(f"fedavg_model/logs/{"IID" if iid_flag else "NIID"}_r{rounds}_j{local_steps}{"_nc" + str(nc) if not iid_flag else ""}.csv created")
    def get_hyperparameters(self):
        return self.hyperp
    def get_accuracy(self):
        return self.accuracy
    def get_scheduler(self):
        return self.scheduler
    def add_eval_results(self, accuracy, loss):
        self.eval_accuracy = accuracy
        self.eval_loss = loss
    def add_weights(self, weights):
        self.weights = weights
    def get_round_results(self, idx):
        return self.round_results[idx]
    def get_hyperparameters_list(self):
        hp = self.get_hyperparameters()
        return [hp.get_iid_flag(), hp.get_rounds(), hp.get_local_steps(), hp.get_nc(), 10]
    def save_file(self):
        with open(self.path, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                            [
                                ["IID", "rounds", "local steps", "classes_per_client", "selected_clients", "accuracy", "loss"],
                                self.get_hyperparameters_list() + [self.eval_accuracy, self.eval_loss]
                            ]
            )


class Hyperparameters:
    def __init__(self, iid_flag, rounds, local_steps, nc=0):
        self.iid_flag = iid_flag
        self.rounds = rounds
        self.local_steps = local_steps
        self.nc = nc
    def get_iid_flag(self):
        return self.iid_flag
    def get_nc(self):
        return self.nc
    def get_local_steps(self):
        return self.local_steps
    def get_rounds(self):
        return self.rounds


class RoundResults:
    def __init__(self, round_id, base_path):
        self.round_path = f"{base_path}/round_{round_id}"
        mkdir(self.round_path)
        self.round_id = round_id
        self.client_results = []
    def add_client(self, client_id):
        self.client_results.append(ClientResults(client_id, self.round_path))
    def get_client_results(self):
        return self.client_results

class ClientResults:    # train results of a specific client in a specific round
    def __init__(self, client_id, base_path):
        self.path = f"{base_path}/client_{client_id}.csv"
        self.client_id = client_id
        self.train_loss = []# train loss at each epoch
    def add_client_results(self, train_loss):
        self.train_loss.append(train_loss)
    def save_file(self):
        with open(self.path, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                            [
                                ["round", "training_loss"],
                                [[i, a] for i, a in enumerate(self.train_loss)]
                            ]
            )