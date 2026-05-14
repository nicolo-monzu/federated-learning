from torch.utils.data import DataLoader
from torch.utils.data import Subset


def create_client_dataloaders(batch_size):
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader