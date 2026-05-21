import random
from pprint import pprint

from train import start

def random_search(num_trials=20, num_epochs=15):
    best_accuracy = 0
    best_trial = None

    for trial in range(1, num_trials + 1):
        batch_size = random.choice([64, 128, 256])
        # Logarithmic sampling from 1e-4 to 1
        max_lr = 10 ** random.uniform(-4, 0)
        # Linear sampling
        decay_rate = random.uniform(0.65, 0.9)
        # Logarithmic sampling from 1e-6 to 1e-2
        weight_decay = 10 ** random.uniform(-6, -2)

        print(f"Trial: {trial}/{num_trials}")
        pprint({'batch_size': batch_size, 'max_lr': max_lr, 'decay_rate': decay_rate, 'weight_decay': weight_decay})

        trial = start(num_epochs, batch_size, max_lr, decay_rate, weight_decay)

        accuracy = trial['best_accuracy']
        print(f"Best accuracy: {accuracy:.2f}\n")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_trial = trial

    print("RANDOM SEARCH COMPLETED")
    print("Best trial:")
    pprint(best_trial)


if __name__ == "__main__":
    random_search()