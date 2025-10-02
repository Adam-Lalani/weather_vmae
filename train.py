import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, device, epoch, scheduler=None):
    """
    Runs a single training epoch for the VideoMAE model.

    Args:
        model (torch.nn.Module): The VideoMAE model to be trained.
        dataloader (DataLoader): The DataLoader providing training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating weights.
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').
        epoch (int): The current epoch number, for display purposes.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    
    total_loss = 0.0
  
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):

        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(pixel_values=batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    if scheduler:
        scheduler.step()
        
    avg_epoch_loss = total_loss / len(dataloader)
    return avg_epoch_loss