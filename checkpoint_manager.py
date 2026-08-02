import torch
import os

def save(checkpoint, filename):
    torch.save(checkpoint, filename+'.tmp')
    os.replace(filename+'.tmp', filename)

class CheckpointManager:
    def __init__(self, directory, run_name):
        os.makedirs(directory, exist_ok=True)
        self.path = f'{directory}/{run_name}.pth'
        self.best_acc = -1.0
        self.best_model_state_dict = None
        self.loaded = None

    def save(self, epoch, model, optimizer, scaler, scheduler, logger, val_acc):
        # Update if best model
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_model_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        checkpoint = {
            # Last
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
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

        self.best_acc = checkpoint['best_accuracy']
        self.best_model_state_dict = checkpoint['best_model_state_dict']

        return checkpoint['logger_state_dict'], checkpoint['epoch']


    def restore_state(self, model, optimizer, scaler, scheduler):
        if self.loaded is None:
            raise RuntimeError("No checkpoint loaded to restore state from.")
        model.load_state_dict(self.loaded['model_state_dict'])
        optimizer.load_state_dict(self.loaded['optimizer_state_dict'])
        scaler.load_state_dict(self.loaded['scaler_state_dict'])
        scheduler.load_state_dict(self.loaded['scheduler_state_dict'])
        self.loaded = None # Free memory
