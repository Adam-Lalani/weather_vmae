import torch
from tqdm import tqdm
import numpy as np

def train_one_epoch(model, dataloader, optimizer, device, epoch, scheduler=None, context_ratio=0.85):
    """
    Runs a single training epoch for the V-JEPA 2 model.

    Args:
        model (torch.nn.Module): The V-JEPA 2 model to be trained.
        dataloader (DataLoader): The DataLoader providing training data.
        optimizer (torch.optim.Optimizer): The optimizer for updating weights.
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').
        epoch (int): The current epoch number, for display purposes.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        context_ratio (float): The ratio of patches to use as context (default: 0.85).

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
        num_frames_after_tubelet = config.frames_per_clip // tubelet_size
        num_spatial_patches = (config.crop_size // config.patch_size) ** 2
        num_patches = num_spatial_patches * num_frames_after_tubelet

        num_context = int(context_ratio * num_patches)
        context_mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            context_indices = torch.randperm(num_patches, device=device)[:num_context]
            context_mask[i, context_indices] = True
        
        optimizer.zero_grad()
        
        pixel_values_videos = batch.permute(0, 2, 1, 3, 4)
        
        outputs = model(
            pixel_values_videos=pixel_values_videos,
            context_mask=context_mask,
            return_dict=True
        )
        
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
