from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

B = 5   # minibatch size (batch size of each client)

def create_client_dataloaders(clients: list[Subset]):
    dataloaders = []
    for c in clients:
        subset, _ = train_test_split(list(range(len(c))), test_size=0.1, random_state=1234, stratify=c.targets)
        train_loader = DataLoader(subset, batch_size=B, shuffle=True, num_workers=2)
        _, subset = train_test_split(list(range(len(c))), test_size=0.1, random_state=1234, stratify=c.targets)
        val_loader = DataLoader(c, batch_size=B, shuffle=True)
        dataloaders.append((train_loader, val_loader))
    return dataloaders