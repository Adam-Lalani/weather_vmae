# Weather VideoMAE: Self-Supervised Learning on ERA5 Weather Data

This repository implements a Video Masked Autoencoder (VideoMAE) for self-supervised pre-training on ERA5 weather data. The model learns to reconstruct masked patches in weather video sequences, enabling downstream applications in weather prediction and analysis.

## 🎯 Project Overview

This project applies the VideoMAE architecture to weather data, specifically:
- **Data**: ERA5 reanalysis data (temperature, wind components) over Canada
- **Model**: VideoMAE with 75% masking ratio for self-supervised pre-training
- **Goal**: Learn rich weather representations for downstream tasks

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/aeolus-earth/toy_weather_mae_adam.git
cd weather_mae
```

2. **Create and activate environment:**
```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate weather_mae

# Or using pip
python -m venv weather_mae
source weather_mae/bin/activate  # On Windows: weather_mae\Scripts\activate
pip install -r requirements.txt
```

### Data Setup

The code automatically downloads ERA5 data from Google Cloud Storage. No additional setup is required for data access.

## 🏃‍♂️ Usage

### Basic Training

```bash
python main.py --epochs 1000 --batch_size 8 --lr 1.5e-4
```

### Advanced Configuration

```bash
python main.py \
    --epochs 600 \
    --batch_size 8 \
    --lr 1.5e-4 \
    --warmup_epochs 20 \
    --image_size 224 \
    --num_frames 8 \
    --embed_dim 768 \
    --encoder_depth 12 \
    --date_start 2020-01-01 \
    --date_end 2020-12-31
```

### Cluster Training (SLURM)

```bash
sbatch run.slurm
```

## 📊 Model Architecture

- **Backbone**: Vision Transformer (ViT)
- **Input**: 8-frame weather video clips (224×224 pixels)
- **Variables**: Temperature, U-wind, V-wind components
- **Masking**: 75% of patches randomly masked
- **Output**: Reconstructed weather frames

## 🔧 Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 600 | Total training epochs |
| `--batch_size` | 4 | Batch size per GPU |
| `--lr` | 1.5e-4 | Learning rate |
| `--image_size` | 224 | Spatial resolution |
| `--num_frames` | 8 | Temporal sequence length |
| `--embed_dim` | 768 | Encoder embedding dimension |
| `--date_start` | 2021-01-01 | Start date for data (YYYY-MM-DD) |
| `--date_end` | 2021-12-31 | End date for data (YYYY-MM-DD) |

### Model Architecture

- **Encoder**: 12 layers, 12 attention heads, 768 embedding dim
- **Decoder**: 4 layers, 6 attention heads, 384 embedding dim
- **Patches**: 16×16 pixel patches
- **Masking**: 75% random masking ratio

## 📈 Monitoring

The training process includes:
- **Weights & Biases**: Automatic experiment tracking
- **Visualization**: Reconstruction GIFs every 10 epochs
- **Checkpoints**: Model state saved periodically

## 🗂️ File Structure

```
weather_mae/
├── main.py              # Main training script
├── model.py             # VideoMAE model definition
├── data.py              # ERA5 data loading utilities
├── train.py             # Training loop implementation
├── visualize.py         # Visualization and logging
├── requirements.txt     # Python dependencies
├── run_oscar.slurm      # SLURM job script
└── README.md           # This file
```
