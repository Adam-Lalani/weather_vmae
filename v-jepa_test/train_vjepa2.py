import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, device, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        outputs = model(pixel_values_videos=batch, return_dict=True)
        encoder_repr = outputs.last_hidden_state
        predictor_repr = outputs.predictor_output.last_hidden_state
        target_repr = encoder_repr[:, :predictor_repr.shape[1], :]
        loss = torch.nn.functional.mse_loss(predictor_repr, target_repr)
        
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if scheduler:
        scheduler.step()
        
    return total_loss / len(dataloader)
