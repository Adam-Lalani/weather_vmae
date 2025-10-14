import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import time
import wandb
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_dataloaders
from model_vjepa2 import create_vjepa2_model
from train_vjepa2 import train_one_epoch

def main(config):
    """
    Main function to orchestrate the training and evaluation process.
    """

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config.seed)
    
    run_name = f"vjepa2_pretrain_ep{config.epochs}_bs{config.batch_size}_lr{config.lr}"
    wandb.init(project="era5-vjepa2-pretraining", config=config, name=run_name, entity="adam_lalani-brown-university")
    
    print(f"Using device: {DEVICE}")

    print("Initializing dataloaders...")
    train_loader, val_loader, train_mean, train_std, lat, lon = get_dataloaders(
        batch_size=config.batch_size,
        clip_length=config.num_frames,
        image_size=config.image_size,
        num_workers=config.num_workers,
        date_start=config.date_start,
        date_end=config.date_end
    )
    
    
    print("Initializing V-JEPA 2 model...")
    model_args = {
        'image_size': config.image_size, 'patch_size': config.patch_size,
        'num_frames': config.num_frames, 'num_channels': config.num_channels,
        'hidden_size': config.hidden_size, 'num_hidden_layers': config.num_hidden_layers,
        'num_attention_heads': config.num_attention_heads, 'pred_hidden_size': config.pred_hidden_size,
        'pred_num_attention_heads': config.pred_num_attention_heads, 'pred_num_hidden_layers': config.pred_num_hidden_layers,
        'pred_num_mask_tokens': config.pred_num_mask_tokens,
    }
    vjepa2_model = create_vjepa2_model(**model_args)

    if DEVICE == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        vjepa2_model = nn.DataParallel(vjepa2_model)
    vjepa2_model.to(DEVICE)

    model_to_optimize = vjepa2_model.module if isinstance(vjepa2_model, nn.DataParallel) else vjepa2_model
    optimizer = torch.optim.AdamW(
        model_to_optimize.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay, 
        betas=(0.9, 0.95)
    )
    
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=config.warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=config.min_lr)
        ], 
        milestones=[config.warmup_epochs]
    )

    print("\nStarting V-JEPA 2 Pre-training...")
    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*30} Epoch {epoch}/{config.epochs} {'='*30}")
        
        avg_loss = train_one_epoch(
            vjepa2_model, train_loader, optimizer, DEVICE, epoch, scheduler
        )
        
        wandb.log({
            "epoch": epoch,
            "loss": avg_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        if epoch % config.log_interval == 0:
            print(f"--- Epoch {epoch}: Saving checkpoint ---")
            
            # Save checkpoint
            checkpoint_path = f"vjepa2_era5_epoch_{epoch}.pth"
            model_to_save = vjepa2_model.module if isinstance(vjepa2_model, nn.DataParallel) else vjepa2_model
            torch.save(model_to_save.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

    print(f"\n{'='*60}")
    print("Pre-training completed!")
    print(f"Total Experiment Time: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"{'='*60}")

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a V-JEPA 2 model on ERA5 data.")

    # Training config
    parser.add_argument('--epochs', type=int, default=400, help='Total training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Peak learning rate.')
    parser.add_argument('--warmup_epochs', type=int, default=40, help='Epochs for learning rate warmup.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for cosine decay.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='AdamW weight decay.')
    parser.add_argument('--log_interval', type=int, default=5, help='Epoch interval for logging viz and checkpoints.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    
    # Model config
    parser.add_argument('--image_size', type=int, default=224, help='Spatial size of each frame.')
    parser.add_argument('--patch_size', type=int, default=16, help='Size of each patch.')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames per clip.')
    parser.add_argument('--num_channels', type=int, default=3, help='Number of data channels (variables).')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Encoder hidden size.')
    parser.add_argument('--num_hidden_layers', type=int, default=24, help='Number of encoder layers.')
    parser.add_argument('--num_attention_heads', type=int, default=16, help='Number of encoder attention heads.')
    parser.add_argument('--pred_hidden_size', type=int, default=384, help='Predictor hidden size.')
    parser.add_argument('--pred_num_attention_heads', type=int, default=12, help='Number of predictor attention heads.')
    parser.add_argument('--pred_num_hidden_layers', type=int, default=12, help='Number of predictor layers.')
    parser.add_argument('--pred_num_mask_tokens', type=int, default=10, help='Number of mask tokens.')
    
    # V-JEPA specific
    parser.add_argument('--context_ratio', type=float, default=0.85, help='Ratio of patches to use as context.')
    
    # Data parameters
    parser.add_argument('--date_start', type=str, default='2021-01-01', help='Start date for data (YYYY-MM-DD).')
    parser.add_argument('--date_end', type=str, default='2021-12-31', help='End date for data (YYYY-MM-DD).')

    args = parser.parse_args()
    
    main(args)
