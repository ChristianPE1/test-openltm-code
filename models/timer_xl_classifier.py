"""
Timer-XL adapted for binary rainfall classification
Modified from original Timer-XL for Peru rainfall prediction
"""

import torch
from torch import nn
from layers.Transformer_EncDec import TimerBlock, TimerLayer
from layers.SelfAttention_Family import AttentionLayer, TimeAttention


class Model(nn.Module):
    """
    Timer-XL adapted for binary classification (Rain / No Rain)
    
    Key modifications:
    1. Replace regression head with classification head
    2. Add pooling layer to aggregate temporal information
    3. Support for transfer learning from pre-trained weights
    
    Args:
        configs: Configuration object with attributes:
            - input_token_len: Length of input tokens
            - output_token_len: Length of output tokens (not used for classification)
            - d_model: Model dimension
            - n_heads: Number of attention heads
            - e_layers: Number of encoder layers
            - d_ff: Feed-forward dimension
            - dropout: Dropout rate
            - activation: Activation function
            - use_norm: Whether to use normalization
            - covariate: Whether using covariate variables
            - flash_attention: Whether to use flash attention
            - output_attention: Whether to output attention weights
            - n_classes: Number of output classes (default: 2 for binary)
    """
    
    def __init__(self, configs):
        super().__init__()
        
        self.input_token_len = configs.input_token_len
        self.output_token_len = configs.output_token_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.n_classes = getattr(configs, 'n_classes', 2)  # Binary classification
        
        # Embedding layer: projects input tokens to model dimension
        self.embedding = nn.Linear(self.input_token_len, configs.d_model)
        
        # Timer-XL encoder blocks (can load pre-trained weights)
        self.blocks = TimerBlock(
            [
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(
                            True, 
                            attention_dropout=configs.dropout,
                            output_attention=self.output_attention, 
                            d_model=configs.d_model, 
                            num_heads=configs.n_heads,
                            covariate=configs.covariate, 
                            flash_attention=configs.flash_attention
                        ),
                        configs.d_model, 
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Projection head (for compatibility with pre-trained weights)
        self.head = nn.Linear(configs.d_model, configs.output_token_len)
        
        # ========== CLASSIFICATION HEAD ==========
        # Pool temporal dimension and classify
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Pool over time
        
        self.classifier = nn.Sequential(
            nn.Linear(configs.output_token_len, 256),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(64, self.n_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier layers with Xavier uniform"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def load_pretrained_encoder(self, checkpoint_path, strict=False):
        """
        Load pre-trained weights for encoder and head
        
        Args:
            checkpoint_path: Path to pre-trained checkpoint
            strict: Whether to strictly enforce weight loading
        
        Returns:
            Number of loaded parameters
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model_dict = self.state_dict()
        pretrained_dict = {}
        
        # Load compatible layers (embedding, blocks, head)
        for k, v in checkpoint.items():
            # Skip classifier layers (they don't exist in pre-trained model)
            if 'classifier' in k or 'pooling' in k:
                continue
            
            # Load only if shapes match
            if k in model_dict and v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
        
        # Update model weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
        
        print(f"✅ Loaded {len(pretrained_dict)}/{len(checkpoint)} pre-trained weights")
        print(f"⚠️  Classifier initialized randomly (will be trained from scratch)")
        
        return len(pretrained_dict)
    
    def freeze_encoder(self):
        """Freeze encoder layers for feature extraction"""
        for name, param in self.named_parameters():
            if 'classifier' not in name and 'pooling' not in name:
                param.requires_grad = False
        
        print("🔒 Encoder frozen. Only classifier will be trained.")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder layers for fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
        
        print("🔓 Encoder unfrozen. Full model will be fine-tuned.")
    
    def forward(self, x, x_mark, y_mark):
        """
        Forward pass for classification
        
        Args:
            x: Input tensor [B, L, C]
                B: batch size
                L: sequence length
                C: number of features
            x_mark: Time marks (not used, for compatibility)
            y_mark: Future time marks (not used, for compatibility)
        
        Returns:
            logits: Classification logits [B, n_classes]
            attns: Attention weights (if output_attention=True)
        """
        
        # Instance normalization with safer epsilon
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-6
            )
            # Clamp to prevent division by very small numbers
            stdev = torch.clamp(stdev, min=1e-5)
            x /= stdev
        
        B, L, C = x.shape
        
        # Reshape to [B, C, L] for patching
        x = x.permute(0, 2, 1)
        
        # Create tokens: [B, C, N, P]
        # N: number of tokens per variable
        # P: token length (input_token_len)
        x = x.unfold(
            dimension=-1, 
            size=self.input_token_len, 
            step=self.input_token_len
        )
        N = x.shape[2]
        
        # Embed tokens: [B, C, N, D]
        embed_out = self.embedding(x)
        
        # Flatten to [B, C * N, D]
        embed_out = embed_out.reshape(B, C * N, -1)
        
        # Pass through Timer-XL encoder
        embed_out, attns = self.blocks(embed_out, n_vars=C, n_tokens=N)
        
        # Project: [B, C * N, output_token_len]
        dec_out = self.head(embed_out)
        
        # ========== CLASSIFICATION ==========
        # For classification, pool over the variable dimension
        # dec_out shape: [B, C * N, output_token_len]
        
        # Reshape to separate variables: [B, C * N, output_token_len] -> [B, C, N, output_token_len]
        dec_out = dec_out.reshape(B, C, N, self.output_token_len)
        
        # Pool over variables (C): [B, C, N, output_token_len] -> [B, N, output_token_len]
        dec_out_pooled = dec_out.mean(dim=1)
        
        # Pool over temporal tokens (N): [B, N, output_token_len] -> [B, output_token_len]
        dec_out_final = dec_out_pooled.mean(dim=1)
        
        # Safety check: replace NaN/Inf with zeros (shouldn't happen with fixed normalization)
        if torch.isnan(dec_out_final).any() or torch.isinf(dec_out_final).any():
            dec_out_final = torch.nan_to_num(dec_out_final, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Classify: [B, output_token_len] -> [B, n_classes]
        logits = self.classifier(dec_out_final)
        
        if self.output_attention:
            return logits, attns
        return logits


class ModelMultiRegion(nn.Module):
    """
    Timer-XL for multi-region rainfall classification
    Predicts binary label for each region independently
    """
    
    def __init__(self, configs):
        super().__init__()
        
        self.n_regions = getattr(configs, 'n_regions', 5)
        self.input_token_len = configs.input_token_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # Shared encoder
        self.embedding = nn.Linear(self.input_token_len, configs.d_model)
        
        self.blocks = TimerBlock(
            [
                TimerLayer(
                    AttentionLayer(
                        TimeAttention(
                            True,
                            attention_dropout=configs.dropout,
                            output_attention=self.output_attention,
                            d_model=configs.d_model,
                            num_heads=configs.n_heads,
                            covariate=configs.covariate,
                            flash_attention=configs.flash_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.head = nn.Linear(configs.d_model, configs.output_token_len)
        
        # Separate classifier per region
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.output_token_len, 128),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(32, 2)
            ) for _ in range(self.n_regions)
        ])
    
    def forward(self, x, x_mark, y_mark):
        """
        Forward pass for multi-region classification
        
        Args:
            x: [B, L, C * n_regions] (concatenated features)
        
        Returns:
            logits: [B, n_regions, 2] (binary logits per region)
        """
        
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x /= stdev
        
        B, L, C_total = x.shape
        C = C_total // self.n_regions  # Features per region
        
        x = x.permute(0, 2, 1)
        x = x.unfold(dimension=-1, size=self.input_token_len, step=self.input_token_len)
        N = x.shape[2]
        
        embed_out = self.embedding(x)
        embed_out = embed_out.reshape(B, C_total * N, -1)
        
        embed_out, attns = self.blocks(embed_out, n_vars=C_total, n_tokens=N)
        
        dec_out = self.head(embed_out)
        dec_out = dec_out.reshape(B, C_total, N, -1).reshape(B, C_total, -1)
        dec_out = dec_out.permute(0, 2, 1)
        
        if self.use_norm:
            dec_out = dec_out * stdev.permute(0, 2, 1) + means.permute(0, 2, 1)
        
        # Pool and classify per region
        logits_list = []
        for i in range(self.n_regions):
            # Extract features for this region
            region_start = i * C
            region_end = (i + 1) * C
            region_out = dec_out[:, :, region_start:region_end].mean(dim=(1, 2))
            
            # Classify
            region_logits = self.classifiers[i](region_out)
            logits_list.append(region_logits)
        
        # Stack: [B, n_regions, 2]
        logits = torch.stack(logits_list, dim=1)
        
        if self.output_attention:
            return logits, attns
        return logits
