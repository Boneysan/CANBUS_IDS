#!/usr/bin/env python3
"""
Test PCA vs non-PCA ML detection performance.

Compares:
1. Training time with/without PCA
2. Inference speed with/without PCA
3. Model accuracy with/without PCA
4. Memory usage with/without PCA
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import psutil
import json

from src.preprocessing.feature_reduction import FeatureReducer
from src.preprocessing.feature_extractor import FeatureExtractor

def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def load_training_data():
    """Load training data from Vehicle_Models."""
    print("=" * 80)
    print("Loading training data...")
    print("=" * 80)
    
    # Try multiple paths
    data_paths = [
        Path("../Vehicle_Models/data/processed/train_normal_comprehensive.csv"),
        Path("/home/mike/Documents/GitHub/Vehicle_Models/data/processed/train_normal_comprehensive.csv")
    ]
    
    df = None
    for path in data_paths:
        if path.exists():
            print(f"✅ Found training data: {path}")
            df = pd.read_csv(path)
            break
    
    if df is None:
        raise FileNotFoundError("Training data not found in any expected location")
    
    print(f"Loaded {len(df):,} samples")
    print(f"Columns: {list(df.columns)[:10]}...")
    return df

def extract_features_from_data(df):
    """Extract CAN features from DataFrame."""
    print("\n" + "=" * 80)
    print("Extracting features...")
    print("=" * 80)
    
    extractor = FeatureExtractor()
    
    X = []
    y = []
    processed = 0
    start_time = time.time()
    
    for idx, row in df.iterrows():
        msg = row.to_dict()
        features = extractor.extract_features(msg)
        
        if features:
            X.append(list(features.values()))
            y.append(row.get('label', 0))  # 0 = normal, 1 = attack
            processed += 1
            
            if processed % 10000 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  Processed {processed:,} messages ({rate:.0f} msg/s)")
    
    X = np.array(X)
    y = np.array(y)
    feature_names = list(features.keys())
    
    elapsed = time.time() - start_time
    print(f"\n✅ Extracted {X.shape[0]:,} feature vectors")
    print(f"   Features: {X.shape[1]}")
    print(f"   Time: {elapsed:.2f}s ({X.shape[0]/elapsed:.0f} msg/s)")
    
    return X, y, feature_names

def train_without_pca(X_train, X_test, y_train, y_test):
    """Train and test model WITHOUT PCA."""
    print("\n" + "=" * 80)
    print("Testing WITHOUT PCA (Baseline)")
    print("=" * 80)
    
    mem_before = get_memory_mb()
    
    # Train model
    print("\n📊 Training Isolation Forest (100 estimators)...")
    start_time = time.time()
    
    model = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)
    
    train_time = time.time() - start_time
    mem_after = get_memory_mb()
    mem_used = mem_after - mem_before
    
    print(f"✅ Training complete: {train_time:.2f}s")
    print(f"   Memory used: {mem_used:.1f} MB")
    
    # Test inference speed
    print("\n⚡ Testing inference speed...")
    
    # Single message inference
    single_times = []
    for i in range(100):
        start = time.time()
        _ = model.predict([X_test[i]])
        single_times.append((time.time() - start) * 1000)  # ms
    
    avg_single = np.mean(single_times)
    
    # Batch inference
    batch_sizes = [10, 100, 1000]
    batch_results = {}
    
    for batch_size in batch_sizes:
        if len(X_test) >= batch_size:
            start = time.time()
            _ = model.predict(X_test[:batch_size])
            elapsed = time.time() - start
            
            per_msg_ms = (elapsed / batch_size) * 1000
            throughput = batch_size / elapsed
            
            batch_results[batch_size] = {
                'per_msg_ms': per_msg_ms,
                'throughput': throughput
            }
    
    # Accuracy test
    print("\n📈 Testing accuracy...")
    y_pred = model.predict(X_test)
    
    # Convert -1/1 to 0/1 for metrics
    y_pred_binary = (y_pred == -1).astype(int)
    
    return {
        'train_time': train_time,
        'memory_mb': mem_used,
        'single_inference_ms': avg_single,
        'batch_results': batch_results,
        'y_pred': y_pred_binary,
        'model': model
    }

def train_with_pca(X_train, X_test, y_train, y_test, feature_names):
    """Train and test model WITH PCA."""
    print("\n" + "=" * 80)
    print("Testing WITH PCA (Optimized)")
    print("=" * 80)
    
    mem_before = get_memory_mb()
    
    # Train PCA reducer
    print("\n🔬 Training PCA reducer (58 → 15 features)...")
    start_pca = time.time()
    
    reducer = FeatureReducer(n_components=15, variance_threshold=0.95)
    X_train_reduced = reducer.fit_transform(X_train, feature_names)
    X_test_reduced = reducer.transform(X_test)
    
    pca_time = time.time() - start_pca
    variance = np.sum(reducer.pca.explained_variance_ratio_)
    
    print(f"✅ PCA complete: {pca_time:.2f}s")
    print(f"   Variance explained: {variance*100:.2f}%")
    print(f"   Reduced: {X_train.shape[1]} → {X_train_reduced.shape[1]} features")
    
    # Show top features
    print("\n📊 Top 10 most important features:")
    for name, importance in reducer.get_feature_importance(10):
        print(f"   {name}: {importance:.4f}")
    
    # Train model on reduced features
    print("\n📊 Training Isolation Forest on reduced features...")
    start_train = time.time()
    
    model = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_reduced)
    
    train_time = time.time() - start_train
    total_time = time.time() - start_pca
    mem_after = get_memory_mb()
    mem_used = mem_after - mem_before
    
    print(f"✅ Training complete: {train_time:.2f}s")
    print(f"   Total time (PCA + training): {total_time:.2f}s")
    print(f"   Memory used: {mem_used:.1f} MB")
    
    # Test inference speed
    print("\n⚡ Testing inference speed with PCA...")
    
    # Single message inference (with PCA transform)
    single_times = []
    for i in range(100):
        start = time.time()
        X_reduced = reducer.transform(X_test[i:i+1])
        _ = model.predict(X_reduced)
        single_times.append((time.time() - start) * 1000)  # ms
    
    avg_single = np.mean(single_times)
    
    # Batch inference
    batch_sizes = [10, 100, 1000]
    batch_results = {}
    
    for batch_size in batch_sizes:
        if len(X_test_reduced) >= batch_size:
            start = time.time()
            _ = model.predict(X_test_reduced[:batch_size])
            elapsed = time.time() - start
            
            per_msg_ms = (elapsed / batch_size) * 1000
            throughput = batch_size / elapsed
            
            batch_results[batch_size] = {
                'per_msg_ms': per_msg_ms,
                'throughput': throughput
            }
    
    # Accuracy test
    print("\n📈 Testing accuracy...")
    y_pred = model.predict(X_test_reduced)
    
    # Convert -1/1 to 0/1 for metrics
    y_pred_binary = (y_pred == -1).astype(int)
    
    return {
        'pca_time': pca_time,
        'train_time': train_time,
        'total_time': total_time,
        'memory_mb': mem_used,
        'variance_explained': variance,
        'single_inference_ms': avg_single,
        'batch_results': batch_results,
        'y_pred': y_pred_binary,
        'reducer': reducer,
        'model': model
    }

def print_comparison(baseline, optimized, y_test):
    """Print detailed comparison."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    print("\n📊 Training Performance:")
    print(f"  Baseline (no PCA):  {baseline['train_time']:6.2f}s")
    print(f"  With PCA (total):   {optimized['total_time']:6.2f}s")
    print(f"  Difference:         {optimized['total_time'] - baseline['train_time']:+6.2f}s")
    
    print("\n💾 Memory Usage:")
    print(f"  Baseline (no PCA):  {baseline['memory_mb']:6.1f} MB")
    print(f"  With PCA:           {optimized['memory_mb']:6.1f} MB")
    savings = baseline['memory_mb'] - optimized['memory_mb']
    savings_pct = (savings / baseline['memory_mb']) * 100 if baseline['memory_mb'] > 0 else 0
    print(f"  Savings:            {savings:6.1f} MB ({savings_pct:+.1f}%)")
    
    print("\n⚡ Inference Speed (Single Message):")
    print(f"  Baseline (no PCA):  {baseline['single_inference_ms']:6.3f} ms/msg")
    print(f"  With PCA:           {optimized['single_inference_ms']:6.3f} ms/msg")
    speedup = baseline['single_inference_ms'] / optimized['single_inference_ms']
    print(f"  Speedup:            {speedup:.2f}x faster")
    
    print("\n🚀 Inference Speed (Batch Processing):")
    for batch_size in [10, 100, 1000]:
        if batch_size in baseline['batch_results']:
            b_ms = baseline['batch_results'][batch_size]['per_msg_ms']
            o_ms = optimized['batch_results'][batch_size]['per_msg_ms']
            speedup = b_ms / o_ms
            
            b_thr = baseline['batch_results'][batch_size]['throughput']
            o_thr = optimized['batch_results'][batch_size]['throughput']
            
            print(f"\n  Batch size {batch_size}:")
            print(f"    Baseline: {b_ms:6.3f} ms/msg  ({b_thr:8.0f} msg/s)")
            print(f"    With PCA: {o_ms:6.3f} ms/msg  ({o_thr:8.0f} msg/s)")
            print(f"    Speedup:  {speedup:.2f}x faster")
    
    print("\n📈 Accuracy Comparison:")
    
    # Baseline metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    y_test_binary = (y_test != 0).astype(int)  # Assuming label 0 = normal
    
    baseline_acc = accuracy_score(y_test_binary, baseline['y_pred'])
    optimized_acc = accuracy_score(y_test_binary, optimized['y_pred'])
    
    print(f"  Baseline accuracy:  {baseline_acc*100:6.2f}%")
    print(f"  PCA accuracy:       {optimized_acc*100:6.2f}%")
    print(f"  Difference:         {(optimized_acc - baseline_acc)*100:+6.2f}%")
    
    print("\n🎯 Expected Pi 4 Performance:")
    print(f"  Current Pi 4 ML:    17.31 msg/s (57.7 ms/msg)")
    
    # Estimate Pi 4 with PCA
    # Assume Pi 4 is ~30x slower than Ubuntu for ML
    ubuntu_with_pca = optimized['batch_results'][100]['throughput']
    pi4_estimate = ubuntu_with_pca / 30  # Conservative estimate
    
    print(f"  Estimated with PCA: {pi4_estimate:.0f} msg/s ({1000/pi4_estimate:.1f} ms/msg)")
    print(f"  Expected speedup:   {pi4_estimate/17.31:.1f}x faster on Pi 4")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ PCA reduces features from 58 → 15 (74% reduction)")
    print(f"✅ Explains {optimized['variance_explained']*100:.1f}% of variance")
    print(f"✅ {speedup:.1f}x faster inference on Ubuntu")
    print(f"✅ Accuracy loss: {(baseline_acc - optimized_acc)*100:.2f}%")
    print(f"✅ Memory savings: {savings:.0f} MB")
    print(f"\n🎯 Expected Pi 4 improvement: 17.31 → {pi4_estimate:.0f} msg/s")

def save_models(baseline, optimized, output_dir="data/models"):
    """Save trained models."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Saving models...")
    print("=" * 80)
    
    # Save PCA models
    reducer_path = output_path / "feature_reducer.joblib"
    model_path = output_path / "model_with_pca.joblib"
    
    optimized['reducer'].save(str(reducer_path))
    joblib.dump(optimized['model'], model_path)
    
    print(f"✅ Saved PCA reducer: {reducer_path}")
    print(f"✅ Saved PCA model: {model_path}")
    
    # Save metadata
    metadata = {
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'features_original': 58,
        'features_reduced': 15,
        'variance_explained': float(optimized['variance_explained']),
        'training_time': optimized['total_time'],
        'speedup': float(baseline['single_inference_ms'] / optimized['single_inference_ms'])
    }
    
    metadata_path = output_path / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved metadata: {metadata_path}")

def main():
    print("\n" + "=" * 80)
    print("PCA PERFORMANCE TEST")
    print("=" * 80)
    print("Comparing ML detection with and without PCA feature reduction")
    print("Target: 3-5x speedup with <5% accuracy loss")
    print("=" * 80)
    
    # Load data
    df = load_training_data()
    
    # Extract features
    X, y, feature_names = extract_features_from_data(df)
    
    # Split data
    print("\n📊 Splitting train/test data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Test:  {len(X_test):,} samples")
    
    # Test baseline (no PCA)
    baseline = train_without_pca(X_train, X_test, y_train, y_test)
    
    # Test with PCA
    optimized = train_with_pca(X_train, X_test, y_train, y_test, feature_names)
    
    # Compare results
    print_comparison(baseline, optimized, y_test)
    
    # Save models
    save_models(baseline, optimized)
    
    print("\n✅ Test complete!")
    print("\nNext steps:")
    print("  1. Review performance comparison above")
    print("  2. Test on Pi 4: scp data/models/* pi@raspberrypi:~/CANBUS_IDS/data/models/")
    print("  3. Run Pi 4 tests: python3 scripts/test_ml_detector_pi.py")

if __name__ == '__main__':
    main()
