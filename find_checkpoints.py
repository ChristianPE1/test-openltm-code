"""
Quick script to find all checkpoints and show their details

Usage: python find_checkpoints.py
"""

import os
import glob
from datetime import datetime

def find_all_checkpoints():
    """Find all checkpoint.pth files in the checkpoints directory"""
    
    print("\n" + "="*80)
    print("🔍 Searching for Checkpoints")
    print("="*80 + "\n")
    
    # Search patterns
    patterns = [
        "checkpoints/**/checkpoint.pth",
        "checkpoints/*/checkpoint.pth",
        "checkpoints/*/*/checkpoint.pth"
    ]
    
    all_checkpoints = []
    for pattern in patterns:
        checkpoints = glob.glob(pattern, recursive=True)
        all_checkpoints.extend(checkpoints)
    
    # Remove duplicates
    all_checkpoints = list(set(all_checkpoints))
    
    if not all_checkpoints:
        print("❌ No checkpoints found!")
        print("\n📁 Looking in: ./checkpoints/")
        print("\n💡 Checkpoints are saved during training as 'checkpoint.pth'")
        print("   They should be in directories like:")
        print("   - checkpoints/classification_peru_rainfall_small_*/checkpoint.pth")
        print("   - checkpoints/classification_peru_rainfall_timerxl_*/checkpoint.pth")
        return
    
    # Sort by modification time (newest first)
    all_checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    print(f"✅ Found {len(all_checkpoints)} checkpoint(s)\n")
    print("="*80)
    
    for i, ckpt_path in enumerate(all_checkpoints, 1):
        # Get directory name (contains model info)
        ckpt_dir = os.path.dirname(ckpt_path)
        model_name = os.path.basename(ckpt_dir)
        
        # Get file stats
        stat = os.stat(ckpt_path)
        size_mb = stat.st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        # Identify model type
        if "small" in model_name.lower():
            model_type = "Small Model (4 layers, 512 dim)"
            emoji = "🟢"
        elif "timerxl" in model_name.lower():
            model_type = "Transfer Learning (8 layers, 1024 dim, pretrained)"
            emoji = "🔵"
        elif "scratch" in model_name.lower():
            model_type = "Big Model from Scratch (8 layers, 1024 dim)"
            emoji = "🟠"
        else:
            model_type = "Unknown Model"
            emoji = "⚪"
        
        print(f"\n{emoji} Checkpoint #{i}: {model_type}")
        print(f"   📁 Directory: {model_name}")
        print(f"   📄 Path: {ckpt_path}")
        print(f"   📦 Size: {size_mb:.2f} MB")
        print(f"   🕐 Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   📊 To test: python test_checkpoint_standalone.py --checkpoint_path \"{ckpt_path}\"")
        print("-" * 80)
    
    # Recommend latest checkpoint
    print(f"\n💡 Recommendation:")
    print(f"   Test the most recent checkpoint (#{1}):")
    print(f"   python test_checkpoint_standalone.py --checkpoint_path \"{all_checkpoints[0]}\"")
    print("\n   Or use automatic detection:")
    print("   python test_checkpoint_standalone.py --find_latest")
    
    print("\n" + "="*80 + "\n")


def compare_checkpoint_sizes():
    """Compare sizes of different model checkpoints"""
    
    patterns = {
        "Small Model": "checkpoints/*small*/checkpoint.pth",
        "Transfer Learning": "checkpoints/*timerxl*/checkpoint.pth",
        "Big Model Scratch": "checkpoints/*scratch*/checkpoint.pth"
    }
    
    print("\n📊 Checkpoint Size Comparison")
    print("="*80)
    
    for model_name, pattern in patterns.items():
        checkpoints = glob.glob(pattern, recursive=True)
        if checkpoints:
            size_mb = os.path.getsize(checkpoints[0]) / (1024 * 1024)
            print(f"   {model_name:25s}: {size_mb:6.2f} MB")
        else:
            print(f"   {model_name:25s}: Not found")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    find_all_checkpoints()
    compare_checkpoint_sizes()
