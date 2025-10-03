import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import io

def denormalize_clip(clip, mean, std):
    """
    Denormalizes a tensor clip (T, C, H, W) with mean and standard deviation.
    """
    clip = clip.clone()
    # Reshape mean and std to (1, C, 1, 1) for broadcasting over the clip
    mean = torch.tensor(mean, device=clip.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=clip.device).view(1, -1, 1, 1)
    clip.mul_(std).add_(mean)
    return clip

def log_reconstruction_gif(model, clip, mean, std, lat, lon, device, epoch):
    """
    Generates a reconstruction GIF of a single clip, plots it on a map, 
    and logs it to W&B.
    """
    model.eval()
    with torch.no_grad():
        # Add a batch dimension and send to device
        clip_batch = clip.unsqueeze(0).to(device)

        # Handle DataParallel wrapper
        model_to_use = model.module if hasattr(model, 'module') else model
        
        # Generate a random mask for visualization (same as training)
        config = model_to_use.config
        tubelet_size = getattr(config, 'tubelet_size', 2)
        num_frames_after_tubelet = config.num_frames // tubelet_size
        num_spatial_patches = (config.image_size // config.patch_size) ** 2
        num_patches = num_spatial_patches * num_frames_after_tubelet
        
        # Create a mask with 75% masking ratio
        mask_ratio = 0.75
        num_masked = int(mask_ratio * num_patches)
        bool_masked_pos = torch.zeros((1, num_patches), dtype=torch.bool, device=device)
        mask_indices = torch.randperm(num_patches, device=device)[:num_masked]
        bool_masked_pos[0, mask_indices] = True
        
        # 1. Get model outputs with reconstruction
        outputs = model(pixel_values=clip_batch, bool_masked_pos=bool_masked_pos)
        
        # 2. Simple visualization: show original and darken masked regions
        original_vis = denormalize_clip(clip_batch.cpu().squeeze(0), mean, std)
        
        # Create masked version by darkening masked patches  
        masked_vis = original_vis.clone()
        patch_size = config.patch_size
        tubelet_size = getattr(config, 'tubelet_size', 2)
        num_patches_side = config.image_size // patch_size
        
        # Simple spatial masking visualization (approximation)
        for idx in range(num_patches):
            if bool_masked_pos[0, idx]:
                # Map patch index to spatial location (simplified)
                spatial_idx = idx % (num_patches_side ** 2)
                i = spatial_idx // num_patches_side
                j = spatial_idx % num_patches_side
                # Darken this spatial region across all frames
                masked_vis[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] *= 0.3
        
        # 3. Create a proper reconstruction by combining visible and predicted patches
        # The model's logits only contain predictions for MASKED patches, not all patches
        hybrid_vis = original_vis.clone()
        
        # Get the model's predictions for masked patches only
        pred_patches = outputs.logits.detach().cpu()  # (B, num_masked_patches, patch_dim)
        
        # Debug: print some statistics about predictions
        print(f"Prediction stats - min: {pred_patches.min():.3f}, max: {pred_patches.max():.3f}, mean: {pred_patches.mean():.3f}")
        print(f"Original stats - min: {original_vis.min():.3f}, max: {original_vis.max():.3f}, mean: {original_vis.mean():.3f}")
        
        # Track which masked patch we're currently processing
        masked_patch_idx = 0
        pred_values = []  # Store all predicted values for debugging
        
        # For each patch, check if it's masked and use prediction if so
        for idx in range(num_patches):
            if bool_masked_pos[0, idx]:
                spatial_idx = idx % (num_patches_side ** 2)
                i = spatial_idx // num_patches_side
                j = spatial_idx % num_patches_side
                
                # Get the predicted patch values for this masked patch
                pred_patch = pred_patches[0, masked_patch_idx]  # (patch_dim,)
                
                # The predictions are in normalized space, so we need to denormalize them
                # For simplicity, we'll denormalize the mean of the patch
                pred_value_normalized = pred_patch.mean().item()
                
                # Denormalize using the same mean/std as the original data
                # pred_value = pred_value_normalized * std + mean
                # For the first channel (temperature), use the first element of mean/std
                pred_value = pred_value_normalized * std[0] + mean[0]
                pred_values.append(pred_value)
                
                # Fill the masked region with the denormalized predicted value
                hybrid_vis[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = pred_value
                
                # Move to next masked patch
                masked_patch_idx += 1
        
        # Debug: print statistics about denormalized predictions
        if pred_values:
            pred_values = torch.tensor(pred_values)
            print(f"Denormalized prediction stats - min: {pred_values.min():.3f}, max: {pred_values.max():.3f}, mean: {pred_values.mean():.3f}")
            print(f"Number of unique values: {len(torch.unique(pred_values))}")
        
        print(f"Reconstruction stats - min: {hybrid_vis.min():.3f}, max: {hybrid_vis.max():.3f}, mean: {hybrid_vis.mean():.3f}")

        # --- GIF Generation ---
        gif_frames = []
        num_frames = clip.shape[0]
        
        # Determine a consistent color range for the temperature variable (channel 0)
        vmin = original_vis[:, 0, :, :].min()
        vmax = original_vis[:, 0, :, :].max()
        
        print(f"\nGenerating reconstruction GIF for epoch {epoch}...")
        for t in range(num_frames):
            fig, axes = plt.subplots(
                1, 3, 
                figsize=(24, 7),
                subplot_kw={'projection': ccrs.PlateCarree()}
            )
            fig.suptitle(f'VideoMAE Reconstruction (Epoch {epoch}, Frame {t+1}/{num_frames})', fontsize=20)

            titles = ['Original', 'Masked Input', 'Reconstruction']
            clips_to_plot = [original_vis, masked_vis, hybrid_vis]

            for i, ax in enumerate(axes):
                # We'll visualize the first channel (temperature)
                frame_data = clips_to_plot[i][t, 0, :, :]
                
                im = ax.imshow(frame_data, origin='upper', extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                               transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax)
                
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.set_title(titles[i], fontsize=16)
            
            # Add a shared colorbar
            fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, label='Temperature (K)')

            # Save frame to a temporary buffer and append to list
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            gif_frames.append(imageio.imread(buf))
            plt.close(fig)

        # 5. Save the GIF and log to W&B
        gif_path = f"reconstruction_epoch_{epoch}.gif"
        imageio.mimsave(gif_path, gif_frames, fps=2)
        
        # Use wandb.Video to log the gif
        wandb.log({"Reconstruction GIF": wandb.Video(gif_path, fps=2, format="gif")})
        print("GIF logged to W&B.")

    model.train()