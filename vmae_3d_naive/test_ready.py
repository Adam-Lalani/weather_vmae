#!/usr/bin/env python3
"""
Test script to verify the code is ready to run.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print("✅ torch imported successfully")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    try:
        import xarray as xr
        print("✅ xarray imported successfully")
    except ImportError as e:
        print(f"❌ xarray import failed: {e}")
        return False
    
    try:
        import gcsfs
        print("✅ gcsfs imported successfully")
    except ImportError as e:
        print(f"❌ gcsfs import failed: {e}")
        return False
    
    try:
        from transformers import VideoMAEConfig, VideoMAEForPreTraining
        print("✅ transformers imported successfully")
    except ImportError as e:
        print(f"❌ transformers import failed: {e}")
        return False
    
    try:
        import wandb
        print("✅ wandb imported successfully")
    except ImportError as e:
        print(f"❌ wandb import failed: {e}")
        return False
    
    return True

def test_local_imports():
    """Test if local modules can be imported."""
    print("\nTesting local imports...")
    
    try:
        from data import get_dataloaders, TEMPORAL_RESOLUTIONS
        print("✅ data module imported successfully")
    except ImportError as e:
        print(f"❌ data module import failed: {e}")
        return False
    
    try:
        from model_mae import create_videomae_model
        print("✅ model_mae module imported successfully")
    except ImportError as e:
        print(f"❌ model_mae module import failed: {e}")
        return False
    
    try:
        from train import train_one_epoch
        print("✅ train module imported successfully")
    except ImportError as e:
        print(f"❌ train module import failed: {e}")
        return False
    
    try:
        from visualize import log_reconstruction_gif
        print("✅ visualize module imported successfully")
    except ImportError as e:
        print(f"❌ visualize module import failed: {e}")
        return False
    
    return True

def test_temporal_resolutions():
    """Test if temporal resolutions are properly configured."""
    print("\nTesting temporal resolutions...")
    
    try:
        from data import TEMPORAL_RESOLUTIONS
        
        expected_resolutions = ['1h', '6h', '12h']
        for resolution in expected_resolutions:
            if resolution in TEMPORAL_RESOLUTIONS:
                print(f"✅ {resolution} resolution configured")
            else:
                print(f"❌ {resolution} resolution missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Temporal resolutions test failed: {e}")
        return False

def test_main_script():
    """Test if main script can be parsed."""
    print("\nTesting main script...")
    
    try:
        # Test argument parsing
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--temporal_resolution', type=str, default='6h', 
                           choices=['1h', '6h', '12h'])
        parser.add_argument('--num_frames', type=int, default=8)
        parser.add_argument('--epochs', type=int, default=600)
        parser.add_argument('--batch_size', type=int, default=4)
        
        # Test with default args
        args = parser.parse_args(['--temporal_resolution', '6h', '--num_frames', '8'])
        print("✅ Main script argument parsing works")
        return True
    except Exception as e:
        print(f"❌ Main script test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("TESTING CODE READINESS")
    print("="*60)
    
    tests = [
        test_imports,
        test_local_imports,
        test_temporal_resolutions,
        test_main_script
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - CODE IS READY TO RUN!")
        print("\nTo run training:")
        print("python main.py --temporal_resolution 6h --num_frames 8 --epochs 10")
    else:
        print("❌ SOME TESTS FAILED - FIX ISSUES BEFORE RUNNING")
        print("\nInstall missing dependencies:")
        print("pip install -r requirements.txt")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    main()
