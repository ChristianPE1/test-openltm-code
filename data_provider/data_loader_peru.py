"""
Custom DataLoader for Peru Rainfall Prediction
Handles ERA5 data with binary classification target (RainTomorrow)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class PeruRainfallDataset(Dataset):
    """
    Dataset for Peru rainfall binary classification using ERA5 data
    
    Args:
        root_path: Path to processed data directory
        flag: 'train', 'val', or 'test'
        size: [seq_len, input_token_len, output_token_len]
        data_path: CSV filename (default: 'peru_rainfall.csv')
        scale: Whether to apply standardization
        subset_rand_ratio: Ratio of data to use (for few-shot learning)
        target_horizon: Hours ahead to predict (default: 24)
        nonautoregressive: Not used for classification (compatibility)
        test_flag: Not used for classification (compatibility)
    """
    
    def __init__(self, root_path, flag='train', size=None, data_path='peru_rainfall.csv', 
                 scale=True, subset_rand_ratio=1.0, target_horizon=24, 
                 nonautoregressive=False, test_flag=0):
        
        # Configuration
        self.seq_len = size[0]  # Lookback window
        self.input_token_len = size[1]  # Token length for Timer-XL
        self.output_token_len = size[2]  # Not used for classification
        self.flag = flag
        assert flag in ['train', 'val', 'test']
        
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.subset_rand_ratio = subset_rand_ratio
        self.target_horizon = target_horizon
        
        # For few-shot learning
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        
        self.__read_data__()
    
    def __read_data__(self):
        """Load and preprocess ERA5 data"""
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        
        # Load data (CSV format expected)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
            
            # Expected columns:
            # ['timestamp', 'region', 'feature_1', ..., 'feature_N', 'precipitation', 'rain_24h']
            
            # Separate features and target
            if 'timestamp' in df_raw.columns:
                df_raw = df_raw.drop('timestamp', axis=1)
            if 'region' in df_raw.columns:
                df_raw = df_raw.drop('region', axis=1)
            
            # Target column (binary: 0=No Rain, 1=Rain)
            if 'rain_24h' in df_raw.columns:
                target_col = 'rain_24h'
            elif 'rain_tomorrow' in df_raw.columns:
                target_col = 'rain_tomorrow'
            else:
                raise ValueError("Target column 'rain_24h' or 'rain_tomorrow' not found")
            
            # Features (all except target)
            feature_cols = [col for col in df_raw.columns if col not in [target_col, 'precipitation']]
            
            # Separate features and labels
            features = df_raw[feature_cols].values
            labels = df_raw[target_col].values
            
        else:
            raise ValueError(f'Unsupported file format: {dataset_file_path}')
        
        # Define data splits (temporal split to avoid data leakage)
        data_len = len(features)
        num_train = int(data_len * 0.7)
        num_val = int(data_len * 0.15)
        num_test = data_len - num_train - num_val
        
        border1s = [0, num_train - self.seq_len, num_train + num_val - self.seq_len]
        border2s = [num_train, num_train + num_val, data_len]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Standardization
        if self.scale:
            train_data = features[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            features = self.scaler.transform(features)
        
        # Store data
        self.data_x = features[border1:border2]
        self.labels = labels[border1:border2]
        
        self.n_features = self.data_x.shape[-1]
        self.n_samples = len(self.data_x) - self.seq_len
        
        print(f"[{self.flag.upper()}] Data loaded: {len(self.data_x)} timesteps, {self.n_features} features")
        print(f"[{self.flag.upper()}] Available samples: {self.n_samples}")
        print(f"[{self.flag.upper()}] Class distribution: {np.bincount(self.labels.astype(int))}")
    
    def __len__(self):
        return self.n_samples // self.internal
    
    def __getitem__(self, index):
        """
        Returns:
            seq_x: Input sequence [seq_len, n_features]
            label: Binary target (0 or 1)
        """
        # Adjust index for few-shot learning
        index = index * self.internal
        
        # Extract sequence
        s_begin = index
        s_end = s_begin + self.seq_len
        
        # Features
        seq_x = self.data_x[s_begin:s_end, :]
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        
        # Target (label at the end of sequence)
        label = self.labels[s_end - 1]
        label = torch.tensor(label, dtype=torch.long)
        
        # Return dummy seq_x_mark and seq_y_mark for compatibility
        seq_x_mark = torch.zeros((self.seq_len, 1))
        seq_y_mark = torch.zeros((1, 1))
        
        return seq_x, label, seq_x_mark, seq_y_mark


class PeruRainfallMultiRegionDataset(Dataset):
    """
    Dataset for Peru rainfall with multiple regions
    Each sample contains data from all 5 regions
    
    Args:
        root_path: Path to processed data directory
        flag: 'train', 'val', or 'test'
        size: [seq_len, input_token_len, output_token_len]
        data_path: CSV filename (default: 'peru_rainfall_multiregion.csv')
        scale: Whether to apply standardization
        subset_rand_ratio: Ratio of data to use
        n_regions: Number of regions (default: 5)
        nonautoregressive: Not used for classification (compatibility)
        test_flag: Not used for classification (compatibility)
    """
    
    def __init__(self, root_path, flag='train', size=None, 
                 data_path='peru_rainfall_multiregion.csv',
                 scale=True, subset_rand_ratio=1.0, n_regions=5,
                 nonautoregressive=False, test_flag=0):
        
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        self.n_regions = n_regions
        
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.subset_rand_ratio = subset_rand_ratio
        
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        
        self.__read_data__()
    
    def __read_data__(self):
        """Load multi-region data"""
        self.scalers = [StandardScaler() for _ in range(self.n_regions)]
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        
        df_raw = pd.read_csv(dataset_file_path)
        
        # Expected structure: timestamp, region_0_feat_0, ..., region_0_rain, ..., region_4_rain
        # Or separate files per region
        
        # For simplicity, assume columns: timestamp, region, features..., target
        # Group by timestamp to align all regions
        if 'timestamp' in df_raw.columns:
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
            df_raw = df_raw.sort_values('timestamp')
        
        # Store data per region
        self.data_by_region = []
        self.labels_by_region = []
        
        for region_id in range(self.n_regions):
            region_data = df_raw[df_raw['region'] == region_id]
            
            # Extract features and target
            feature_cols = [col for col in region_data.columns 
                          if col not in ['timestamp', 'region', 'rain_24h', 'rain_tomorrow', 'precipitation']]
            
            features = region_data[feature_cols].values
            
            if 'rain_24h' in region_data.columns:
                labels = region_data['rain_24h'].values
            elif 'rain_tomorrow' in region_data.columns:
                labels = region_data['rain_tomorrow'].values
            else:
                raise ValueError("Target column not found")
            
            self.data_by_region.append(features)
            self.labels_by_region.append(labels)
        
        # Data splits
        data_len = len(self.data_by_region[0])
        num_train = int(data_len * 0.7)
        num_val = int(data_len * 0.15)
        
        border1s = [0, num_train - self.seq_len, num_train + num_val - self.seq_len]
        border2s = [num_train, num_train + num_val, data_len]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Standardization per region
        if self.scale:
            for i in range(self.n_regions):
                train_data = self.data_by_region[i][border1s[0]:border2s[0]]
                self.scalers[i].fit(train_data)
                self.data_by_region[i] = self.scalers[i].transform(self.data_by_region[i])
        
        # Slice data
        self.data_by_region = [data[border1:border2] for data in self.data_by_region]
        self.labels_by_region = [labels[border1:border2] for labels in self.labels_by_region]
        
        self.n_features = self.data_by_region[0].shape[-1]
        self.n_samples = len(self.data_by_region[0]) - self.seq_len
        
        print(f"[{self.flag.upper()}] Multi-region data loaded")
        print(f"[{self.flag.upper()}] {self.n_regions} regions, {self.n_features} features per region")
        print(f"[{self.flag.upper()}] Available samples: {self.n_samples}")
    
    def __len__(self):
        return self.n_samples // self.internal
    
    def __getitem__(self, index):
        """
        Returns:
            seq_x: [seq_len, n_regions * n_features] (concatenated)
            labels: [n_regions] (binary per region)
        """
        index = index * self.internal
        
        s_begin = index
        s_end = s_begin + self.seq_len
        
        # Concatenate all regions
        seq_x_list = []
        labels_list = []
        
        for i in range(self.n_regions):
            seq_x_region = self.data_by_region[i][s_begin:s_end, :]
            label_region = self.labels_by_region[i][s_end - 1]
            
            seq_x_list.append(seq_x_region)
            labels_list.append(label_region)
        
        # Stack regions horizontally
        seq_x = np.concatenate(seq_x_list, axis=1)
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        
        labels = torch.tensor(labels_list, dtype=torch.long)
        
        # Dummy marks
        seq_x_mark = torch.zeros((self.seq_len, 1))
        seq_y_mark = torch.zeros((self.n_regions, 1))
        
        return seq_x, labels, seq_x_mark, seq_y_mark
