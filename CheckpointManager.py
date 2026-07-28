import random
import numpy as np
import torch
import os

def save(checkpoint, filename):
    torch.save(checkpoint, filename+'.tmp')
    os.replace(filename+'.tmp', filename)

class CheckpointManager:
    def __init__(self, directory, run_name):
        os.makedirs(directory, exist_ok=True)
        self.last_path = f'{directory}/{run_name}_last.pth'
        self.best_path = f'{directory}/{run_name}_best.pth'
        self.best_acc = -1 # Accuracy of the best checkpoint (-1 if it doesn't exist)
        self.loaded = None

    def _save_if_best(self, checkpoint, accuracy):
        if accuracy > self.best_acc:
            # Update best accuracy
            self.best_acc = accuracy
            # Save best checkpoint
            save(checkpoint, self.best_path)

    def save(self, epoch, model, optimizer, scheduler, val_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy': val_acc,
            # RNG states
            'torch_rng_state': torch.get_rng_state(),
            'python_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        # Save last checkpoint
        save(checkpoint, self.last_path)

        # Conditionally save best checkpoint
        self._save_if_best(checkpoint, val_acc)


    def resume(self):
        # Delete temp files
        for path in [self.last_path, self.best_path]:
            try:
                os.remove(path + '.tmp')
            except FileNotFoundError:
                pass

        # Load last checkpoint
        last = torch.load(self.last_path)
        self.loaded = last

        # Load best checkpoint's accuracy if it exists
        try:
            best = torch.load(self.best_path)
            self.best_acc = best['accuracy']
        except FileNotFoundError:
            pass

        # Evaluate if the resumed 'last' epoch is a new best
        self._save_if_best(last, last['accuracy'])

        return last['epoch'], self.best_acc


    def restore_state(self, model, optimizer, scheduler):
        if self.loaded is None:
            raise RuntimeError("No checkpoint loaded to restore state from.")
        model.load_state_dict(self.loaded['model_state_dict'])
        optimizer.load_state_dict(self.loaded['optimizer_state_dict'])
        scheduler.load_state_dict(self.loaded['scheduler_state_dict'])
        torch.set_rng_state(self.loaded["torch_rng_state"])
        random.setstate(self.loaded["python_rng_state"])
        np.random.set_state(self.loaded["numpy_rng_state"])
        if self.loaded["cuda_rng_state"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.loaded["cuda_rng_state"])

        self.loaded = None # Free memory