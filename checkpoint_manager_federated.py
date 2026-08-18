import torch
import os

def save(checkpoint, filename):
    torch.save(checkpoint, filename+'.tmp')
    os.replace(filename+'.tmp', filename)

class CheckpointManagerFederated:
    def __init__(self, directory, run_name):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory
        self.path = f'{directory}/{run_name}.pth'
        self.best_acc = -1.0
        self.best_model_state_dict = None
        self.loaded = None

    def set_run_name(self, run_name):
        self.path = f'{self.dir}/{run_name}.pth'

    def save(self, round, model, scale, scheduler, logger, val_acc):
        # Update if best model
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_model_state_dict = {
                k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()
            }

        checkpoint = {
            'training_type': 'federated',
            # Last
            'round': round,
            'model_state_dict': model.state_dict(),
            'scale': scale,
            'scheduler_state_dict': scheduler.state_dict(),
            'logger_state_dict': logger.state_dict(),
            'accuracy': val_acc,
            # Best
            'best_model_state_dict': self.best_model_state_dict,
            'best_accuracy': self.best_acc
        }
        save(checkpoint, self.path)


    def resume(self):
        # Delete temp file if exist
        try:
            os.remove(self.path + '.tmp')
        except FileNotFoundError:
            pass

        # Load checkpoint
        checkpoint = torch.load(self.path)
        self.loaded = checkpoint

        # Check training type
        if checkpoint['training_type'] != 'federated':
            raise RuntimeError(
                f"Checkpoint training type mismatch: expected "
                f"'federated', got '{checkpoint['training_type']}'."
            )

        self.best_acc = checkpoint['best_accuracy']
        self.best_model_state_dict = checkpoint['best_model_state_dict']

        return checkpoint['logger_state_dict'], checkpoint['round']


    def restore_state(self, model, clients, scheduler):
        if self.loaded is None:
            raise RuntimeError("No checkpoint loaded to restore state from.")
        model.load_state_dict(self.loaded['model_state_dict'])

        scale = self.loaded['scale']

        scheduler.load_state_dict(self.loaded['scheduler_state_dict'])

        self.loaded = None # Free memory
        return scale
