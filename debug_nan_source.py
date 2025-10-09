# Debug script para identificar dónde aparece NaN en el forward pass

import torch
import torch.nn as nn
import sys
sys.path.append('/content/test-openltm-code')

from models.timer_xl_classifier import Model
from utils.tools import dotdict

# Crear configuración
configs = dotdict({
    'input_token_len': 96,
    'output_token_len': 96,
    'output_attention': False,
    'use_norm': False,  # IMPORTANTE: Sin normalización para debug
    'n_classes': 2,
    'd_model': 1024,
    'n_heads': 8,
    'e_layers': 8,
    'd_ff': 2048,
    'dropout': 0.1,
    'activation': 'relu',
    'covariate': False,
    'flash_attention': False
})

print("1. Creating model...")
model = Model(configs).cuda()

print("2. Loading pretrained weights...")
checkpoint_path = '/content/test-openltm-code/checkpoints/timer_xl/checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location='cuda:0')

# Load compatible weights
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in checkpoint.items() 
                   if k in model_dict and v.shape == model_dict[k].shape
                   and 'classifier' not in k}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

print(f"✅ Loaded {len(pretrained_dict)} pretrained parameters")

# Crear datos de prueba
print("\n3. Creating test data...")
batch_size = 2
seq_len = 1440
n_features = 27

x = torch.randn(batch_size, seq_len, n_features).cuda()
x_mark = torch.zeros(batch_size, seq_len, 1).cuda()
y_mark = torch.zeros(batch_size, 1, 1).cuda()

print(f"Input shape: {x.shape}")
print(f"Input stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
print(f"Has NaN in input: {torch.isnan(x).any()}")
print(f"Has Inf in input: {torch.isinf(x).any()}")

# Hook para capturar outputs intermedios
def add_hooks(model):
    activations = {}
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                activations[name] = {
                    'shape': output.shape,
                    'min': output.min().item() if not torch.isnan(output).any() else float('nan'),
                    'max': output.max().item() if not torch.isnan(output).any() else float('nan'),
                    'mean': output.mean().item() if not torch.isnan(output).any() else float('nan'),
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item(),
                }
        return hook
    
    model.embedding.register_forward_hook(get_activation('embedding'))
    model.blocks.register_forward_hook(get_activation('blocks'))
    model.head.register_forward_hook(get_activation('head'))
    model.classifier.register_forward_hook(get_activation('classifier'))
    
    return activations

print("\n4. Running forward pass with hooks...")
activations = add_hooks(model)

model.eval()
with torch.no_grad():
    try:
        outputs = model(x, x_mark, y_mark)
        print(f"\n✅ Forward pass completed!")
        print(f"Output shape: {outputs.shape}")
        print(f"Output stats: min={outputs.min():.4f}, max={outputs.max():.4f}")
        print(f"Has NaN: {torch.isnan(outputs).any()}")
        print(f"Has Inf: {torch.isinf(outputs).any()}")
    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")

print("\n5. Intermediate activations:")
for name, stats in activations.items():
    print(f"\n{name}:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    if stats['has_nan']:
        print(f"  ⚠️ NaN DETECTED IN {name.upper()}!")
        break

print("\n6. Checking embedding weights...")
print(f"Embedding weight: min={model.embedding.weight.min():.4f}, max={model.embedding.weight.max():.4f}")
print(f"Embedding has NaN: {torch.isnan(model.embedding.weight).any()}")
print(f"Embedding has Inf: {torch.isinf(model.embedding.weight).any()}")

print("\n7. Checking head weights...")
print(f"Head weight: min={model.head.weight.min():.4f}, max={model.head.weight.max():.4f}")
print(f"Head has NaN: {torch.isnan(model.head.weight).any()}")
print(f"Head has Inf: {torch.isinf(model.head.weight).any()}")

print("\n8. Checking classifier weights...")
for i, module in enumerate(model.classifier.modules()):
    if isinstance(module, nn.Linear):
        print(f"Classifier layer {i}:")
        print(f"  Weight: min={module.weight.min():.4f}, max={module.weight.max():.4f}")
        print(f"  Has NaN: {torch.isnan(module.weight).any()}")
        print(f"  Has Inf: {torch.isinf(module.weight).any()}")
