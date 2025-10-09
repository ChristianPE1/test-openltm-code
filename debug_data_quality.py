"""
Debug script to check data quality issues causing NaN
"""

import pandas as pd
import numpy as np

# Load processed data
df = pd.read_csv('datasets/processed/peru_rainfall.csv')

print("=" * 80)
print("🔍 DATA QUALITY DIAGNOSTIC")
print("=" * 80)

print(f"\n1️⃣ BASIC INFO:")
print(f"   Shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

print(f"\n2️⃣ NaN VALUES:")
nan_counts = df.isnull().sum()
if nan_counts.sum() == 0:
    print("   ✅ No NaN values found")
else:
    print("   ❌ NaN values detected:")
    print(nan_counts[nan_counts > 0])

print(f"\n3️⃣ INFINITY VALUES:")
inf_counts = {}
for col in df.columns:
    if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count

if len(inf_counts) == 0:
    print("   ✅ No Inf values found")
else:
    print("   ❌ Inf values detected:")
    for col, count in inf_counts.items():
        print(f"      {col}: {count}")

print(f"\n4️⃣ EXTREME VALUES (potential issues):")
feature_cols = [col for col in df.columns if col not in ['rain_24h', 'timestamp', 'region']]

for col in feature_cols[:10]:  # Check first 10 features
    if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
        q01 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        min_val = df[col].min()
        max_val = df[col].max()
        
        # Check for extreme outliers (beyond 100x IQR)
        if max_val > 1e6 or min_val < -1e6:
            print(f"   ⚠️ {col}:")
            print(f"      Range: [{min_val:.2e}, {max_val:.2e}]")
            print(f"      1%-99%: [{q01:.2e}, {q99:.2e}]")

print(f"\n5️⃣ ZERO VARIANCE FEATURES:")
zero_var_cols = []
for col in feature_cols:
    if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
        if df[col].std() == 0:
            zero_var_cols.append(col)

if len(zero_var_cols) == 0:
    print("   ✅ All features have variance")
else:
    print("   ❌ Zero variance features (will cause NaN in normalization):")
    for col in zero_var_cols:
        print(f"      {col} = {df[col].iloc[0]}")

print(f"\n6️⃣ CLASS DISTRIBUTION:")
print(df['rain_24h'].value_counts())
print(f"   Balance: {df['rain_24h'].value_counts(normalize=True) * 100}")

print(f"\n7️⃣ CORRELATION WITH TARGET:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corrwith(df['rain_24h']).abs().sort_values(ascending=False)
print("   Top 5 most correlated features:")
for col, corr in correlations.head(5).items():
    if col != 'rain_24h':
        print(f"      {col}: {corr:.4f}")

print(f"\n8️⃣ POTENTIAL FIXES:")
issues = []

if nan_counts.sum() > 0:
    issues.append("- Fill NaN values with forward fill or interpolation")
if len(inf_counts) > 0:
    issues.append("- Replace Inf with max/min finite values")
if len(zero_var_cols) > 0:
    issues.append(f"- Remove zero variance features: {zero_var_cols}")

if len(issues) > 0:
    print("   REQUIRED FIXES:")
    for issue in issues:
        print(f"   {issue}")
else:
    print("   ✅ Data looks clean!")
    print("   → NaN issue must be in model initialization or forward pass")
    print("   → Check embedding layer weights for NaN")

print("\n" + "=" * 80)
print("💡 RECOMMENDATION:")
if len(issues) > 0:
    print("   Run preprocessing again with --fillna --clip_outliers flags")
else:
    print("   Data is clean. Problem is likely:")
    print("   1. Model initialization has NaN weights")
    print("   2. Gradient explosion in first forward pass")
    print("   3. Learning rate too high (try 1e-5 instead of 1e-3)")
print("=" * 80)
