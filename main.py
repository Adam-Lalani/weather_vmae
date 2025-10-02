import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import time
import wandb
import argparse
import os

from data import get_dataloaders
from model import create_videomae_model
from train import train_one_epoch
from visualize import log_reconstruction_gif

def main(config):
    """
    Main function to orchestrate the training and evaluation process.
    """

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config.seed)
    
    run_name = f"vmae_pretrain_ep{config.epochs}_bs{config.batch_size}_lr{config.lr}"
    wandb.init(project="era5-videomae-pretraining", config=config, name=run_name, entity="adam_lalani-brown-university")
    
    print(f"Using device: {DEVICE}")

    print("Initializing dataloaders...")
    train_loader, val_loader, train_mean, train_std, lat, lon = get_dataloaders(
        batch_size=config.batch_size,
        clip_length=config.num_frames,
        image_size=config.image_size,
        num_workers=config.num_workers
    )
    
    # fixed clip for reconstruction gif
    vis_clip = next(iter(val_loader))[0]
    
    print("Initializing model...")
    model_args = {
        'image_size': config.image_size, 'patch_size': config.patch_size,
        'num_frames': config.num_frames, 'num_channels': config.num_channels,
        'embed_dim': config.embed_dim, 'encoder_depth': config.encoder_depth,
        'encoder_heads': config.encoder_heads, 'decoder_embed_dim': config.decoder_embed_dim,
        'decoder_depth': config.decoder_depth, 'decoder_heads': config.decoder_heads,
    }
    videomae_model = create_videomae_model(**model_args)

    if DEVICE == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        videomae_model = nn.DataParallel(videomae_model)
    videomae_model.to(DEVICE)

    model_to_optimize = videomae_model.module if isinstance(videomae_model, nn.DataParallel) else videomae_model
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

    print("\nStarting VideoMAE Pre-training...")
    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*30} Epoch {epoch}/{config.epochs} {'='*30}")
        
        avg_mae_loss = train_one_epoch(videomae_model, train_loader, optimizer, DEVICE, epoch, scheduler)
        
        wandb.log({
            "epoch": epoch,
            "mae_loss": avg_mae_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        if epoch % config.log_interval == 0:
            print(f"--- Epoch {epoch}: Logging reconstruction GIF and saving checkpoint ---")
            
            # Call the visualization function from the visualize script
            log_reconstruction_gif(
                videomae_model, vis_clip, train_mean.tolist(), train_std.tolist(), 
                lat, lon, DEVICE, epoch
            )
            
            # Save checkpoint
            checkpoint_path = f"videomae_era5_epoch_{epoch}.pth"
            model_to_save = videomae_model.module if isinstance(videomae_model, nn.DataParallel) else videomae_model
            torch.save(model_to_save.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

    print(f"\n{'='*60}")
    print("Pre-training completed!")
    print(f"Total Experiment Time: {(time.time() - start_time) / 3600:.2f} hours")
    print(f"{'='*60}")

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VideoMAE model on ERA5 data.")

    # Training config
    parser.add_argument('--epochs', type=int, default=2, help='Total training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='Peak learning rate.')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Epochs for learning rate warmup.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for cosine decay.')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='AdamW weight decay.')
    parser.add_argument('--log_interval', type=int, default=10, help='Epoch interval for logging viz and checkpoints.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    
    # Model config
    parser.add_argument('--image_size', type=int, default=224, help='Spatial size of each frame.')
    parser.add_argument('--patch_size', type=int, default=16, help='Size of each patch.')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames per clip.')
    parser.add_argument('--num_channels', type=int, default=3, help='Number of data channels (variables).')
    parser.add_argument('--embed_dim', type=int, default=768, help='Encoder embedding dimension.')
    parser.add_argument('--encoder_depth', type=int, default=12, help='Number of encoder layers.')
    parser.add_argument('--encoder_heads', type=int, default=12, help='Number of encoder attention heads.')
    parser.add_argument('--decoder_embed_dim', type=int, default=384, help='Decoder embedding dimension.')
    parser.add_argument('--decoder_depth', type=int, default=4, help='Number of decoder layers.')
    parser.add_argument('--decoder_heads', type=int, default=6, help='Number of decoder attention heads.')

    args = parser.parse_args()
    
    main(args)