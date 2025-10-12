"""
Standalone script to test a saved checkpoint and get full metrics
(Precision, Recall, F1, Confusion Matrix)

Usage:
    python test_checkpoint_standalone.py --checkpoint_path "path/to/checkpoint.pth"

Example:
    python test_checkpoint_standalone.py --checkpoint_path "checkpoints/classification_peru_rainfall_timerxl_*/checkpoint.pth"
"""

import argparse
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import glob
import sys

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your existing modules
from exp.exp_forecast import Exp_Forecast

def test_checkpoint(checkpoint_path):
    """Load checkpoint and run testing"""
    
    print(f"\n{'='*80}")
    print(f"🔍 Testing Checkpoint")
    print(f"{'='*80}\n")
    print(f"📁 Checkpoint: {checkpoint_path}\n")
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path)
    
    # Extract settings from checkpoint path
    checkpoint_dir = os.path.dirname(checkpoint_path)
    setting = os.path.basename(checkpoint_dir)
    
    print(f"📊 Model Setting: {setting}\n")
    
    # Parse settings from directory name
    # Example: classification_peru_rainfall_timerxl_timer_xl_classifier_PeruRainfall_sl1440_it96_ot96_lr1e-05_bt16_wd0_el8_dm1024_dff2048_nh8_cosTrue_Peru_Rainfall_Transfer_Learning_Cleaned_0
    
    parts = setting.split('_')
    
    # Create args object with default values
    class Args:
        def __init__(self):
            self.task_name = 'classification'
            self.is_training = 0
            self.model_id = 'peru_rainfall_test'
            self.model = 'timer_xl_classifier'
            
            # Data settings
            self.data = 'PeruRainfall'
            self.root_path = './datasets/processed/'
            self.data_path = 'peru_rainfall_cleaned.csv'
            self.features = 'MS'
            self.target = 'rain'
            self.freq = 'h'
            self.checkpoints = './checkpoints/'
            
            # Forecasting task
            self.seq_len = 1440  # Default, will be extracted
            self.label_len = 48
            self.pred_len = 96
            
            # Test-specific parameters (usually same as train)
            self.test_seq_len = 1440  # Will match seq_len
            self.test_pred_len = 96  # Will match pred_len
            
            # Timer-XL specific tokens
            self.input_token_len = 96  # Default, will be extracted
            self.output_token_len = 96  # Default, will be extracted
            
            # Model architecture
            self.enc_in = 27
            self.dec_in = 27
            self.c_out = 27
            self.d_model = 1024  # Default, will be extracted
            self.n_heads = 8
            self.e_layers = 8  # Default, will be extracted
            self.d_layers = 1
            self.d_ff = 2048  # Default, will be extracted
            self.moving_avg = 25
            self.factor = 1
            self.distil = True
            self.dropout = 0.1
            self.embed = 'timeF'
            self.activation = 'gelu'
            self.output_attention = False
            
            # Optimization
            self.num_workers = 1  # Set to 1 to allow persistent_workers
            self.itr = 1
            self.train_epochs = 20
            self.batch_size = 16  # Default, will be extracted
            self.patience = 10
            self.learning_rate = 1e-5  # Default, will be extracted
            self.des = 'test'
            self.loss = 'CrossEntropyLoss'
            self.lradj = 'type1'
            self.use_amp = False
            
            # GPU
            self.use_gpu = True
            self.gpu = 0
            self.use_multi_gpu = False
            self.devices = '0'
            self.device_ids = [0]
            self.ddp = False  # Distributed Data Parallel
            self.dp = False  # Data Parallel
            self.local_rank = 0
            
            # Adaptation/Transfer Learning
            self.adaptation = False
            
            # Visualization
            self.visualize = False
            
            # Classification specific
            self.num_class = 2
            
            # Timer-XL specific
            self.patch_len = 16
            self.stride = 8
            self.prompt_num = 10
            self.use_norm = 1
            self.cos = True  # Default, will be extracted
            self.covariate = False  # Whether using covariate variables
            self.flash_attention = False  # Whether to use flash attention
            
            # Data loading parameters
            self.nonautoregressive = False
            self.test_flag = 'test'
            self.subset_rand_ratio = 1.0
            
            # Transfer learning
            self.pretrained_weight = './checkpoints/TimerXL_pretrained.pth'
            self.freeze_encoder = False
    
    args = Args()
    
    # Set test_dir IMMEDIATELY after Args creation (needed by exp_forecast.py)
    args.test_dir = setting
    args.test_file_name = 'checkpoint.pth'
    
    # Parse settings from directory name
    try:
        for i, part in enumerate(parts):
            if part.startswith('sl') and part[2:].isdigit():
                args.seq_len = int(part[2:])
                args.test_seq_len = args.seq_len  # Sync test with train
            elif part.startswith('bt') and part[2:].isdigit():
                args.batch_size = int(part[2:])
            elif part.startswith('lr'):
                # Handle scientific notation: lr1e-05 -> 1e-05
                lr_str = part[2:]
                try:
                    args.learning_rate = float(lr_str)
                except ValueError:
                    pass  # Keep default
            elif part.startswith('el') and part[2:].isdigit():
                args.e_layers = int(part[2:])
            elif part.startswith('dm') and part[2:].isdigit():
                args.d_model = int(part[2:])
            elif part.startswith('dff') and part[2:].isdigit():
                args.d_ff = int(part[2:])
            elif part.startswith('it') and part[2:].isdigit():
                args.input_token_len = int(part[2:])
            elif part.startswith('ot') and part[2:].isdigit():
                args.output_token_len = int(part[2:])
                args.test_pred_len = args.output_token_len  # Sync test with train
            elif part.startswith('cosTrue'):
                args.cos = True
            elif part.startswith('cosFalse'):
                args.cos = False
    except Exception as e:
        print(f"⚠️ Warning: Could not parse all settings from directory name: {e}")
        print(f"   Using default values for missing parameters\n")
    
    print(f"🔧 Model Configuration:")
    print(f"   - Sequence Length: {args.seq_len}")
    print(f"   - Input Token Length: {args.input_token_len}")
    print(f"   - Output Token Length: {args.output_token_len}")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   - Learning Rate: {args.learning_rate}")
    print(f"   - Encoder Layers: {args.e_layers}")
    print(f"   - Model Dimension: {args.d_model}")
    print(f"   - Feed-forward Dimension: {args.d_ff}")
    print(f"   - Cosine Scheduler: {args.cos}")
    print()
    
    # Create experiment instance (this builds the model)
    print(f"🔧 Building model architecture...")
    exp = Exp_Forecast(args)
    
    # Load model weights from checkpoint
    print(f"📥 Loading checkpoint weights...")
    try:
        exp.model.load_state_dict(checkpoint, strict=False)
        exp.model.eval()
        print(f"✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"⚠️  Warning during checkpoint loading: {e}")
        print(f"   Trying to load with strict=False (some weights may be missing)\n")
        exp.model.eval()
    
    print(f"{'='*80}")
    print(f"🧪 Running Test Evaluation...")
    print(f"{'='*80}\n")
    
    # Run testing (same way as run.py does)
    # test=0 means use the already loaded model (don't try to reload from disk)
    try:
        exp.test(setting, test=0)
        print(f"\n{'='*80}")
        print(f"✅ Testing Complete!")
        print(f"{'='*80}")
        print(f"📄 Results saved to: result_classification_{args.model_id}.txt\n")
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ Error during testing: {e}")
        print(f"{'='*80}\n")
        import traceback
        traceback.print_exc()
        
        # Additional debugging
        print(f"\n🔍 Debug Information:")
        print(f"   args.visualize: {hasattr(args, 'visualize')} = {getattr(args, 'visualize', 'NOT SET')}")
        print(f"   args.test_seq_len: {hasattr(args, 'test_seq_len')} = {getattr(args, 'test_seq_len', 'NOT SET')}")
        print(f"   args.test_pred_len: {hasattr(args, 'test_pred_len')} = {getattr(args, 'test_pred_len', 'NOT SET')}")
        print(f"   args.input_token_len: {hasattr(args, 'input_token_len')} = {getattr(args, 'input_token_len', 'NOT SET')}")
        print(f"   args.output_token_len: {hasattr(args, 'output_token_len')} = {getattr(args, 'output_token_len', 'NOT SET')}")


def find_latest_checkpoint(pattern="checkpoints/classification_peru_rainfall_*/**/checkpoint.pth"):
    """Find the most recent checkpoint matching the pattern"""
    checkpoints = glob.glob(pattern, recursive=True)
    
    if not checkpoints:
        print(f"❌ No checkpoints found matching pattern: {pattern}")
        return None
    
    # Sort by modification time
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    print(f"\n📦 Found {len(checkpoints)} checkpoint(s):\n")
    for i, ckpt in enumerate(checkpoints, 1):
        mtime = os.path.getmtime(ckpt)
        from datetime import datetime
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"   {i}. {os.path.basename(os.path.dirname(ckpt))}")
        print(f"      Modified: {mtime_str}")
        print(f"      Path: {ckpt}\n")
    
    return checkpoints[0]


def main():
    parser = argparse.ArgumentParser(description='Test a saved checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint.pth file')
    parser.add_argument('--find_latest', action='store_true',
                        help='Automatically find and test the latest checkpoint')
    parser.add_argument('--pattern', type=str, 
                        default='checkpoints/classification_peru_rainfall_*/**/checkpoint.pth',
                        help='Glob pattern to search for checkpoints')
    
    args = parser.parse_args()
    
    if args.find_latest or args.checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(args.pattern)
        if checkpoint_path is None:
            return
    else:
        checkpoint_path = args.checkpoint_path
    
    test_checkpoint(checkpoint_path)


if __name__ == '__main__':
    main()
