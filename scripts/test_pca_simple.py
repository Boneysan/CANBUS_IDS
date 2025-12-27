#!/usr/bin/env python3
"""
Simple PCA performance test using pre-extracted features.

Compares ML inference speed with and without PCA.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import joblib
import psutil

def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

print("=" * 80)
print("SIMPLE PCA PERFORMANCE TEST")
print("=" * 80)
print("Testing: Baseline vs PCA-reduced ML inference")
print("=" * 80)

# Load raw CAN data and extract features
print("\n📊 Loading data...")
# Try different paths
possible_paths = [
    Path("test_data/attack-free-1.csv"),
    Path("../Vehicle_Models/data/raw/attack-free-1.csv"),
    Path.home() / "Documents" / "GitHub" / "Vehicle_Models" / "data" / "raw" / "attack-free-1.csv"
]

data_path = None
for path in possible_paths:
    if path.exists():
        data_path = path
        break

if data_path is None:
    print("❌ Could not find attack-free-1.csv in any expected location")
    sys.exit(1)

print(f"Loading from: {data_path}")
df_raw = pd.read_csv(data_path)
print(f"✅ Loaded {len(df_raw):,} raw samples")

# Use a subset for faster testing
sample_size = 50000
df_raw = df_raw.sample(n=min(sample_size, len(df_raw)), random_state=42)
print(f"Using {len(df_raw):,} samples for testing")

# Extract features using decision tree detector
print("\n🔧 Extracting features...")
from src.detection.decision_tree_detector import DecisionTreeDetector

detector = DecisionTreeDetector()
features_list = []

for idx, row in df_raw.iterrows():
    # Convert to message format
    message = {
        'can_id': int(row['arbitration_id'], 16) if isinstance(row['arbitration_id'], str) else int(row['arbitration_id']),
        'timestamp': float(row['timestamp']),
        'data': bytes.fromhex(row['data_field']) if isinstance(row['data_field'], str) else row['data_field'],
        'dlc': 8
    }
    features = detector.extract_features(message)
    features_list.append(features)
    
    if (idx + 1) % 10000 == 0:
        print(f"  Processed {idx + 1:,} messages...")

X = np.array(features_list)
y = df_raw['attack'].values if 'attack' in df_raw.columns else np.zeros(len(df_raw))

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]:,}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain: {len(X_train):,}")
print(f"Test:  {len(X_test):,}")

# ============================================================================
# BASELINE: Train without PCA
# ============================================================================
print("\n" + "=" * 80)
print("BASELINE (No PCA)")
print("=" * 80)

mem_before = get_memory_mb()

print("\n⏱️  Training Isolation Forest...")
start = time.time()

model_baseline = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)
model_baseline.fit(X_train)

train_time_baseline = time.time() - start
mem_after = get_memory_mb()
mem_baseline = mem_after - mem_before

print(f"✅ Training: {train_time_baseline:.2f}s")
print(f"   Memory:   {mem_baseline:.1f} MB")

# Test single inference
print("\n⚡ Single inference test...")
single_times = []
for i in range(100):
    start = time.time()
    _ = model_baseline.predict([X_test[i]])
    single_times.append((time.time() - start) * 1000)

single_baseline = np.mean(single_times)
print(f"   Average: {single_baseline:.3f} ms/msg")

# Test batch inference
print("\n🚀 Batch inference test...")
batch_size = 1000
start = time.time()
y_pred_baseline = model_baseline.predict(X_test[:batch_size])
batch_time = time.time() - start

per_msg_baseline = (batch_time / batch_size) * 1000
throughput_baseline = batch_size / batch_time

print(f"   Batch {batch_size}: {per_msg_baseline:.3f} ms/msg")
print(f"   Throughput:  {throughput_baseline:,.0f} msg/s")

# ============================================================================
# OPTIMIZED: Train with PCA
# ============================================================================
print("\n" + "=" * 80)
print("WITH PCA (Optimized)")
print("=" * 80)

mem_before = get_memory_mb()

# Apply PCA
n_components = min(5, X_train.shape[1] - 1)  # Adaptive based on available features
print(f"\n🔬 Applying PCA: {X_train.shape[1]} → {n_components} features...")
start = time.time()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

pca_time = time.time() - start
variance = np.sum(pca.explained_variance_ratio_)

print(f"✅ PCA: {pca_time:.2f}s")
print(f"   Variance: {variance*100:.1f}%")
print(f"   Features: {X_train.shape[1]} → {X_train_pca.shape[1]}")

# Train model
print("\n⏱️  Training Isolation Forest on reduced features...")
start = time.time()

model_pca = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)
model_pca.fit(X_train_pca)

train_time_pca = time.time() - start
total_time_pca = pca_time + train_time_pca
mem_after = get_memory_mb()
mem_pca = mem_after - mem_before

print(f"✅ Training: {train_time_pca:.2f}s")
print(f"   Total (PCA + training): {total_time_pca:.2f}s")
print(f"   Memory: {mem_pca:.1f} MB")

# Test single inference (with PCA transform)
print("\n⚡ Single inference test (with PCA)...")
single_times = []
for i in range(100):
    start = time.time()
    X_reduced = pca.transform(scaler.transform([X_test[i]]))
    _ = model_pca.predict(X_reduced)
    single_times.append((time.time() - start) * 1000)

single_pca = np.mean(single_times)
print(f"   Average: {single_pca:.3f} ms/msg")

# Test batch inference
print("\n🚀 Batch inference test...")
start = time.time()
y_pred_pca = model_pca.predict(X_test_pca[:batch_size])
batch_time = time.time() - start

per_msg_pca = (batch_time / batch_size) * 1000
throughput_pca = batch_size / batch_time

print(f"   Batch {batch_size}: {per_msg_pca:.3f} ms/msg")
print(f"   Throughput:  {throughput_pca:,.0f} msg/s")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

print("\n📊 Training:")
print(f"   Baseline:    {train_time_baseline:6.2f}s")
print(f"   With PCA:    {total_time_pca:6.2f}s")

print("\n💾 Memory:")
print(f"   Baseline:    {mem_baseline:6.1f} MB")
print(f"   With PCA:    {mem_pca:6.1f} MB")
savings = mem_baseline - mem_pca
print(f"   Savings:     {savings:6.1f} MB ({savings/mem_baseline*100:+.1f}%)")

print("\n⚡ Single Message Inference:")
print(f"   Baseline:    {single_baseline:6.3f} ms/msg")
print(f"   With PCA:    {single_pca:6.3f} ms/msg")
speedup_single = single_baseline / single_pca
print(f"   Speedup:     {speedup_single:.2f}x")

print(f"\n🚀 Batch Inference (size={batch_size}):")
print(f"   Baseline:    {per_msg_baseline:6.3f} ms/msg  ({throughput_baseline:8,.0f} msg/s)")
print(f"   With PCA:    {per_msg_pca:6.3f} ms/msg  ({throughput_pca:8,.0f} msg/s)")
speedup_batch = throughput_pca / throughput_baseline
print(f"   Speedup:     {speedup_batch:.2f}x")

print("\n🎯 Expected Raspberry Pi 4 Performance:")
print(f"   Current Pi 4 ML:         17.31 msg/s (57.7 ms/msg)")
pi4_estimate = (throughput_pca / throughput_baseline) * 17.31
print(f"   Estimated with PCA:      {pi4_estimate:.0f} msg/s ({1000/pi4_estimate:.1f} ms/msg)")
print(f"   Expected speedup on Pi:  {speedup_batch:.1f}x")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✅ Feature reduction: {X_train.shape[1]} → {n_components} features ({100*(1-n_components/X_train.shape[1]):.0f}% reduction)")
print(f"✅ Variance explained: {variance*100:.1f}%")
print(f"✅ Inference speedup: {speedup_batch:.1f}x on Ubuntu")
print(f"✅ Memory savings: {savings:.0f} MB")
print(f"\n🎯 Pi 4 estimate: 17 msg/s → {pi4_estimate:.0f} msg/s ({speedup_batch:.1f}x improvement)")

# Save models
output_dir = Path("data/models")
output_dir.mkdir(parents=True, exist_ok=True)

print("\n📦 Saving models...")
joblib.dump({'pca': pca, 'scaler': scaler}, output_dir / "feature_reducer_simple.joblib")
joblib.dump(model_pca, output_dir / "model_with_pca_simple.joblib")
print(f"✅ Saved to {output_dir}/")

print("\n✅ Test complete!")
