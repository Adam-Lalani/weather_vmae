import torch
from tqdm import tqdm
import numpy as np

def train_one_epoch(model, dataloader, optimizer, device, epoch, scheduler=None, mask_ratio=0.75):
    """
    Runs a single training epoch for the VideoMAE model.

    Args:
        model (torch.nn.Module): The VideoMAE model to be trained.
        dataloader (DataLoader): The DataLoader providing training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating weights.
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').
        epoch (int): The current epoch number, for display purposes.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        mask_ratio (float): The ratio of patches to mask (default: 0.75 or 75%).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    
    total_loss = 0.0
  
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):

        batch = batch.to(device)
        batch_size = batch.shape[0]
        
  
        config = model.module.config if hasattr(model, 'module') else model.config
        
 
        tubelet_size = getattr(config, 'tubelet_size', 2)
        num_frames_after_tubelet = config.num_frames // tubelet_size
        num_spatial_patches = (config.image_size // config.patch_size) ** 2
        num_patches = num_spatial_patches * num_frames_after_tubelet

        num_masked = int(mask_ratio * num_patches)
        bool_masked_pos = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            mask_indices = torch.randperm(num_patches, device=device)[:num_masked]
            bool_masked_pos[i, mask_indices] = True
        
        optimizer.zero_grad()
        
        outputs = model(pixel_values=batch, bool_masked_pos=bool_masked_pos)
        loss = outputs.loss
        
        if loss.dim() > 0:
            loss = loss.mean()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    if scheduler:
        scheduler.step()
        
    avg_epoch_loss = total_loss / len(dataloader)
    return avg_epoch_loss