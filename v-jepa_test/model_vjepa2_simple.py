import torch
from transformers import VJEPA2Config, VJEPA2Model

def create_vjepa2_model(
    image_size=224,
    patch_size=16,
    num_frames=8,
    num_channels=3,
    # Encoder settings
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    # Predictor settings
    pred_hidden_size=384,
    pred_num_attention_heads=12,
    pred_num_hidden_layers=12,
    pred_num_mask_tokens=10,
):
    """
    Creates a V-JEPA 2 model using Hugging Face transformers.
    
    Args:
        image_size (int): The spatial size (height and width) of the input frames.
        patch_size (int): The size of the patches to split each frame into.
        num_frames (int): The number of frames (time steps) in each video clip.
        num_channels (int): The number of channels for each frame (e.g., weather variables).
        hidden_size (int): Dimensionality of the encoder layers.
        num_hidden_layers (int): The number of hidden layers in the encoder.
        num_attention_heads (int): Number of attention heads for each attention layer.
        pred_hidden_size (int): Dimensionality of the predictor layers.
        pred_num_attention_heads (int): Number of attention heads in the predictor.
        pred_num_hidden_layers (int): Number of hidden layers in the predictor.
        pred_num_mask_tokens (int): Number of mask tokens for prediction.

    Returns:
        A PyTorch model instance (VJEPA2Model) ready for training.
    """
    print("Creating a randomly initialized V-JEPA 2 model...")

    config = VJEPA2Config(
        patch_size=patch_size,
        crop_size=image_size,
        frames_per_clip=num_frames,
        tubelet_size=2,
        hidden_size=hidden_size,
        in_chans=num_channels,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        pred_hidden_size=pred_hidden_size,
        pred_num_attention_heads=pred_num_attention_heads,
        pred_num_hidden_layers=pred_num_hidden_layers,
        pred_num_mask_tokens=pred_num_mask_tokens,
        pred_zero_init_mask_tokens=True,
    )

    model = VJEPA2Model(config)

    return model
