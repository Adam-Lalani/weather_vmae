import torch
from transformers import VideoMAEConfig, VideoMAEForPreTraining

def create_videomae_model(
    image_size=224,
    patch_size=16,
    num_frames=8,
    num_channels=3,
    # Encoder settings
    embed_dim=768,
    encoder_depth=12,
    encoder_heads=12,
    # Decoder settings
    decoder_embed_dim=384,
    decoder_depth=4,
    decoder_heads=6,
    mlp_ratio=4.0,
):
    """
    Creates a Video Masked Autoencoder (VideoMAE) model with a Vision Transformer
    (ViT) backbone

    Args:
        image_size (int): The spatial size (height and width) of the input frames.
        patch_size (int): The size of the patches to split each frame into.
        num_frames (int): The number of frames (time steps) in each video clip.
        num_channels (int): The number of channels for each frame (e.g., weather variables).
        embed_dim (int): The embedding dimension of the encoder.
        encoder_depth (int): The number of layers in the encoder.
        encoder_heads (int): The number of attention heads in the encoder.
        decoder_embed_dim (int): The embedding dimension of the decoder.
        decoder_depth (int): The number of layers in the decoder.
        decoder_heads (int): The number of attention heads in the decoder.
        mlp_ratio (float): The ratio for the MLP (feed-forward) layers' hidden size.

    Returns:
        A PyTorch model instance (VideoMAEForPreTraining) ready for training.
    """
    print("Creating a randomly initialized VideoMAE model...")

    config = VideoMAEConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_frames=num_frames,
        num_channels=num_channels,
        hidden_size=embed_dim,
        num_hidden_layers=encoder_depth,
        num_attention_heads=encoder_heads,
        intermediate_size=int(embed_dim * mlp_ratio),
        decoder_hidden_size=decoder_embed_dim,
        decoder_num_hidden_layers=decoder_depth,
        decoder_num_attention_heads=decoder_heads,
        decoder_intermediate_size=int(decoder_embed_dim * mlp_ratio),
        norm_pix_loss=True  
    )

    model = VideoMAEForPreTraining(config)

    return model