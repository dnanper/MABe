"""
Example usage of MABe package
"""

# ==================== BASIC USAGE ====================

# 1. Import package
from MABe import (
    load_train_test_data,
    generate_mouse_data,
    transform_single,
    transform_pair,
    StratifiedSubsetClassifierWEval,
    _make_xgb,
    optimize_thresholds_per_action,
    predict_multiclass_with_confidence,
    robustify,
    config
)

import pandas as pd
import numpy as np

# 2. Load data
print("Loading data...")
train, test = load_train_test_data(config.TRAIN_CSV, config.TEST_CSV)

# 3. Get unique body part configurations
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))
print(f"Found {len(body_parts_tracked_list)} configurations")

# 4. Train on one configuration (example)
import json
body_parts_str = body_parts_tracked_list[0]
body_parts = json.loads(body_parts_str)

train_subset = train[train.body_parts_tracked == body_parts_str]

# FPS lookup
fps_lookup = (
    train_subset[['video_id', 'frames_per_second']]
    .drop_duplicates('video_id')
    .set_index('video_id')['frames_per_second']
    .to_dict()
)

# Collect features
X_list = []
y_list = []
meta_list = []

print("Generating features...")
for switch, data, meta, labels in generate_mouse_data(
    train_subset.head(5), 'train', generate_pair=False  # Only 5 videos for demo
):
    fps_val = meta['frames_per_second'].iloc[0]
    arena_dims = (
        meta['arena_width_cm'].iloc[0],
        meta['arena_height_cm'].iloc[0]
    )
    
    if switch == 'single':
        X = transform_single(data, body_parts, fps_val, arena_dims)
        X_list.append(X)
        y_list.append(labels)
        meta_list.append(meta)

if X_list:
    X_train = pd.concat(X_list, axis=0, ignore_index=True)
    y_train = pd.concat(y_list, axis=0, ignore_index=True)
    print(f"Features shape: {X_train.shape}")
    print(f"Actions: {list(y_train.columns)}")
    
    # Train model for one action
    action = y_train.columns[0]
    print(f"\nTraining model for action: {action}")
    
    # Create model
    model = StratifiedSubsetClassifierWEval(
        estimator=_make_xgb(n_estimators=100, learning_rate=0.1, max_depth=5),
        n_samples=10000,
        valid_size=0.2
    )
    
    # Prepare data (only frames with this action labeled)
    action_mask = ~y_train[action].isna()
    X_action = X_train.loc[action_mask]
    y_action = y_train.loc[action_mask, action].astype(int)
    
    print(f"Training samples: {len(y_action)} (positive: {y_action.sum()})")
    
    # Train
    model.fit(X_action, y_action)
    print("Training complete!")
    
    # Predict on same data (just for demo)
    pred_proba = model.predict_proba(X_action)[:, 1]
    print(f"Predictions range: [{pred_proba.min():.3f}, {pred_proba.max():.3f}]")

print("\n" + "="*50)
print("Example complete!")
print("="*50)

# ==================== ADVANCED USAGE ====================

# Example: Ensemble with multiple models
from MABe import _make_lgbm, _make_cb, optimize_ensemble_weights
from sklearn.pipeline import make_pipeline

# Create multiple models
models = [
    make_pipeline(StratifiedSubsetClassifierWEval(
        _make_xgb(n_estimators=100, learning_rate=0.1),
        n_samples=10000
    )),
    make_pipeline(StratifiedSubsetClassifierWEval(
        _make_lgbm(n_estimators=100, learning_rate=0.1),
        n_samples=10000
    )),
]

# Train all (continuing from above example)
if X_list:
    val_preds = []
    for i, m in enumerate(models):
        print(f"Training model {i+1}/{len(models)}...")
        m_clone = m
        # In practice, use train/val split
        # Here we just use same data for demo
        m_clone.fit(X_action.iloc[:100], y_action.iloc[:100])
        pred = m_clone.predict_proba(X_action.iloc[100:200])[:, 1]
        val_preds.append(pred)
    
    # Optimize weights
    if len(val_preds) > 1:
        print("\nOptimizing ensemble weights...")
        weights = optimize_ensemble_weights(val_preds, y_action.iloc[100:200])
        print(f"Optimal weights: {weights}")

# Example: Threshold optimization
from collections import defaultdict

val_preds_map = defaultdict(list)
val_labels_map = defaultdict(list)

# Collect validation predictions across folds (demo)
if X_list:
    for action in y_train.columns[:2]:  # Just 2 actions for demo
        action_mask = ~y_train[action].isna()
        if action_mask.sum() > 0:
            val_preds_map[action].append(pred_proba[:100])
            val_labels_map[action].append(y_action.iloc[:100].values)

if val_preds_map:
    print("\nOptimizing thresholds...")
    optimal_thresholds = optimize_thresholds_per_action(
        val_preds_map, val_labels_map,
        threshold_range=(0.2, 0.5), step=0.05
    )

print("\nAdvanced example complete!")
