import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import wandb
import os

# Channel names for the new data structure (4 variables × 5 pressure levels)
CHANNEL_NAMES = [
    'U-wind 1000hPa', 'U-wind 850hPa', 'U-wind 700hPa', 'U-wind 500hPa', 'U-wind 300hPa',
    'V-wind 1000hPa', 'V-wind 850hPa', 'V-wind 700hPa', 'V-wind 500hPa', 'V-wind 300hPa',
    'Temp 1000hPa', 'Temp 850hPa', 'Temp 700hPa', 'Temp 500hPa', 'Temp 300hPa',
    'SpecHum 1000hPa', 'SpecHum 850hPa', 'SpecHum 700hPa', 'SpecHum 500hPa', 'SpecHum 300hPa'
]
CHANNEL_UNITS = ['m/s', 'm/s', 'm/s', 'm/s', 'm/s', 'm/s', 'm/s', 'm/s', 'm/s', 'm/s', 
                 'K', 'K', 'K', 'K', 'K', 'kg/kg', 'kg/kg', 'kg/kg', 'kg/kg', 'kg/kg']

def denorm(clip, mean, std):
    clip = clip.clone()
    mean_t = torch.as_tensor(mean, device=clip.device).view(1, -1, 1, 1)
    std_t = torch.as_tensor(std, device=clip.device).view(1, -1, 1, 1)
    return clip * std_t + mean_t

def _infer_origin(lat):
    lat = np.asarray(lat)
    return "upper" if lat[0] > lat[-1] else "lower"

def log_reconstruction_gif(model, clip, mean, std, lat, lon, device, epoch, mask_ratio=0.75, channel=0, fps=3, run_name=None):
    model.eval()
    with torch.no_grad():
        T, C, H, W = clip.shape
        # Handle DataParallel wrapper
        model_to_use = model.module if hasattr(model, 'module') else model
        cfg = getattr(model_to_use, "config")
        ps = int(getattr(cfg, "patch_size", 16))
        tube = int(getattr(cfg, "tubelet_size", 2))

        assert H % ps == 0 and W % ps == 0
        assert T % tube == 0

        nt_h, nt_w = H // ps, W // ps
        n_spatial = nt_h * nt_w
        n_tgroups = T // tube
        n_total = n_spatial * n_tgroups

        mask = torch.zeros((1, n_total), dtype=torch.bool, device=device)
        num_masked = int(mask_ratio * n_total)
        ids = torch.randperm(n_total, device=device)[:num_masked]
        mask[0, ids] = True

        px = clip.unsqueeze(0).to(device)
        out = model(pixel_values=px, bool_masked_pos=mask)

        pred_flat = None
        for k in ("reconstruction", "pred_pixel_values", "logits"):
            if hasattr(out, k):
                pred_flat = getattr(out, k)
                break
        if pred_flat is None:
            raise ValueError("Model output lacks reconstruction/logits.")
        pred_flat = pred_flat.detach().cpu()

        expected_dim = tube * C * ps * ps
        if pred_flat.shape[-1] != expected_dim:
            raise ValueError(f"pred dim {pred_flat.shape[-1]} != tube*C*ps*ps={expected_dim}")

        masked_ids = torch.nonzero(mask[0], as_tuple=False).squeeze(1).cpu().tolist()
        if len(masked_ids) != pred_flat.shape[1]:
            raise ValueError("masked index count != number of predicted patches")

        original = denorm(px.squeeze(0).cpu(), mean, std)
        masked = original.clone()
        recon = original.clone()

        mean_t = torch.as_tensor(mean).view(1, C, 1, 1)
        std_t = torch.as_tensor(std).view(1, C, 1, 1)

        m = 0
        for idx in masked_ids:
            tg, s = divmod(idx, n_spatial)
            i, j = divmod(s, nt_w)
            t0, t1 = tg * tube, (tg + 1) * tube

            masked[t0:t1, :, i*ps:(i+1)*ps, j*ps:(j+1)*ps] *= 0.3

            patch = pred_flat[0, m].view(tube, ps, ps, C).permute(0, 3, 1, 2).contiguous()
            patch = patch * std_t + mean_t
            recon[t0:t1, :, i*ps:(i+1)*ps, j*ps:(j+1)*ps] = patch
            m += 1

        vmin = float(original[:, channel].min())
        vmax = float(original[:, channel].max())
        origin = _infer_origin(lat)

        frames = []
        for t in range(T):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": ccrs.PlateCarree()})
            fig.suptitle(f"VideoMAE Reconstruction — Epoch {epoch}, Frame {t+1}/{T}")

            for title, src, ax in zip(("Original", "Masked", "Reconstruction"), (original, masked, recon), axes):
                im = ax.imshow(
                    src[t, channel].numpy(),
                    origin=origin,
                    extent=[float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))],
                    transform=ccrs.PlateCarree(),
                    cmap="coolwarm",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS, linestyle=":")
                ax.set_title(title)

            channel_name = CHANNEL_NAMES[channel] if channel < len(CHANNEL_NAMES) else f"Channel {channel}"
            channel_unit = CHANNEL_UNITS[channel] if channel < len(CHANNEL_UNITS) else "units"
            fig.colorbar(im, ax=axes, orientation="horizontal", pad=0.08, label=f"{channel_name} ({channel_unit})")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            frames.append(imageio.imread(buf))
            buf.close()
            plt.close(fig)

        # Downscale frames and save as MP4 (smaller than GIF)
        scale = 0.5  # downscale factor
        max_frames = min(T, 8)
        ds_frames = []
        for idx, fr in enumerate(frames[:max_frames]):
            # simple stride-based downscale to avoid extra deps
            ds = fr[:: int(1/scale) if scale < 1 else 1, :: int(1/scale) if scale < 1 else 1]
            ds_frames.append(ds)

        # Create run-specific folder for outputs (outside the code directory)
        if run_name:
            output_dir = f"../outputs/{run_name}"
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = "../outputs"
            os.makedirs(output_dir, exist_ok=True)
        
        mp4_path = os.path.join(output_dir, f"reconstruction_epoch_{epoch}.mp4")
        try:
            with imageio.get_writer(mp4_path, fps=max(1, fps), codec="libx264", quality=7) as w:
                for fr in ds_frames:
                    w.append_data(fr)
            wandb.log({"Reconstruction Video": wandb.Video(mp4_path, fps=max(1, fps))}, step=epoch)
        except Exception:
            # fallback: still save GIF if mp4 writing fails
            gif_path = os.path.join(output_dir, f"reconstruction_epoch_{epoch}.gif")
            imageio.mimsave(gif_path, ds_frames, fps=max(1, fps))
            wandb.log({"Reconstruction GIF": wandb.Video(gif_path, fps=max(1, fps), format="gif")}, step=epoch)

    model.train()
  