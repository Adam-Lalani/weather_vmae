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

        # 1. Get model outputs (loss, logits, mask)
        outputs = model(pixel_values=clip_batch)
        mask = outputs.mask.detach()  # Shape: (B, NumPatches)
        
        # Patchify the original clip to work with the mask
        original_patches = model.patchify(clip_batch) # (B, NumPatches, PatchDim)

        # 2. Create the masked input for visualization
        mask_expanded = mask.unsqueeze(-1)
        masked_patches = original_patches * (1 - mask_expanded) # Zero out masked patches
        masked_clip_tensor = model.unpatchify(masked_patches)

        # 3. Create the hybrid reconstruction (most accurate view)
        # Use original visible patches + reconstructed masked patches
        pred_patches = outputs.logits.detach()
        hybrid_patches = original_patches * (1 - mask_expanded) + pred_patches * mask_expanded
        hybrid_reconstruction_tensor = model.unpatchify(hybrid_patches)

        # 4. Denormalize clips for visualization
        original_vis = denormalize_clip(clip_batch.cpu().squeeze(0), mean, std)
        masked_vis = denormalize_clip(masked_clip_tensor.cpu().squeeze(0), mean, std)
        hybrid_vis = denormalize_clip(hybrid_reconstruction_tensor.cpu().squeeze(0), mean, std)

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