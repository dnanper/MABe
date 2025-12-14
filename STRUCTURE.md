# MABe Project Structure - Summary

## 📁 File Organization

```
MABe/
│
├── __init__.py                 # Package initialization, exports
├── setup.py                    # Installation script
├── requirements.txt            # Dependencies
├── README.md                   # Documentation chính
│
├── config.py                   # Configuration & constants
│   ├── SEED, USE_GPU
│   ├── Paths (TRAIN_CSV, TEST_CSV, MODEL_SAVE_DIR)
│   ├── Model hyperparameters (LGBM_CONFIG_1, XGB_CONFIG_1, etc.)
│   └── Training parameters (MAX_SAMPLES, NEG_POS_RATIO)
│
├── utils.py                    # Utility functions
│   ├── _make_lgbm(), _make_xgb(), _make_cb()  # Model factories
│   ├── _scale(), _scale_signed()               # FPS scaling
│   ├── _fps_from_meta()                        # FPS extraction
│   ├── rotate_xy_dataframe()                   # Data augmentation
│   └── _to_num()                               # String parsing
│
├── data_loader.py              # Data loading & generation
│   ├── load_train_test_data()                  # Load CSVs
│   └── generate_mouse_data()                   # Generator for tracking data
│       ├── Single mouse data
│       └── Pair interaction data
│
├── models.py                   # Custom classifiers
│   ├── StratifiedSubsetClassifier              # Basic stratified sampling
│   └── StratifiedSubsetClassifierWEval         # With validation & early stopping
│       ├── Auto metric selection (AUCPR/Logloss)
│       ├── Auto patience tuning
│       └── Support XGB/LGBM/CatBoost
│
├── scoring.py                  # Evaluation & optimization
│   ├── single_lab_f1()                         # Per-lab F1 score
│   ├── mouse_fbeta()                           # Competition metric
│   ├── optimize_ensemble_weights()             # Ensemble weight optimization
│   └── optimize_thresholds_per_action()        # Threshold grid search
│
├── features_helpers.py         # Feature engineering helpers
│   ├── add_curvature_features()                # Trajectory curvature
│   ├── add_multiscale_features()               # Multi-scale temporal
│   ├── add_state_features()                    # Behavioral states
│   ├── add_longrange_features()                # Long-range patterns
│   ├── add_groom_microfeatures()               # Grooming-specific
│   ├── add_spectral_features()                 # Frequency domain (NEW)
│   ├── add_velocity_acceleration_features()    # Kinematics (NEW)
│   ├── add_arena_position_features()           # Arena context (NEW)
│   ├── add_body_part_ratios()                  # Body proportions
│   └── add_shape_features()                    # Geometric shapes
│
├── feature_engineering.py      # Main transforms
│   ├── transform_single()                      # Single mouse features
│   │   ├── Distance features (pairwise body parts)
│   │   ├── Speed features (lagged)
│   │   ├── Body angles & ratios
│   │   ├── Rolling statistics (mean, std, range)
│   │   ├── Nose-tail dynamics
│   │   ├── Ear features
│   │   └── All helpers from features_helpers.py
│   │
│   ├── transform_pair()                        # Pair interaction features
│   │   ├── Inter-mouse distances
│   │   ├── Egocentric coordinates
│   │   ├── Relative orientation
│   │   ├── Approach rate
│   │   ├── Distance bins (very close/close/medium/far)
│   │   ├── Nose-nose dynamics
│   │   ├── Velocity alignment
│   │   └── Social interaction features
│   │
│   ├── add_egocentric_features()               # A's perspective
│   └── add_interaction_features()              # Chase, coordination
│
└── postprocessing.py           # Prediction post-processing
    ├── adaptive_temporal_smoothing()           # Action-aware smoothing
    ├── predict_multiclass_with_confidence()    # Prob → segments
    │   ├── Argmax with per-action thresholds
    │   ├── Run-length encoding
    │   ├── Min duration filtering
    │   └── Confidence scoring
    │
    ├── remove_overlaps_by_confidence()         # Conflict resolution
    ├── fill_missing_video_realistic()          # Dummy generation
    ├── robustify()                             # Fill missing videos
    └── compute_action_properties_from_df()     # Action statistics
```

## 🔑 Key Design Patterns

### 1. **Separation of Concerns**

- **Config**: All magic numbers in one place
- **Utils**: Reusable helpers
- **Data**: Loading logic separate from processing
- **Models**: Custom wrappers isolated
- **Features**: Split into helpers + main transforms
- **Scoring**: Evaluation separate from training
- **Postprocessing**: Output cleanup separate from inference

### 2. **FPS-Aware Processing**

Tất cả temporal features scale theo FPS:

```python
window_scaled = _scale(window_at_30fps, actual_fps)
```

### 3. **Modular Feature Engineering**

```python
# Single mouse
X = transform_single(data, body_parts, fps, arena_dims)
    └── Calls helpers: add_curvature_features()
                      add_spectral_features()
                      add_arena_position_features()
                      etc.

# Pair
X = transform_pair(data, body_parts, fps)
    └── Calls helpers: add_egocentric_features()
                      add_interaction_features()
                      etc.
```

### 4. **Smart Class Imbalance Handling**

```python
StratifiedSubsetClassifierWEval:
    - Stratified sampling
    - Auto class weights
    - Metric selection (AUCPR if imbalanced)
    - Adaptive early stopping patience
```

### 5. **Ensemble Strategy**

```python
# Train multiple models
models = [lgbm_1, lgbm_2, xgb_1, cb_1, ...]

# Optimize weights on validation
weights = optimize_ensemble_weights(val_preds, val_labels)

# Predict with weighted average
final_pred = np.average(all_preds, weights=weights)
```

### 6. **Post-Processing Pipeline**

```python
predictions
    ↓
adaptive_temporal_smoothing()       # Smooth based on action duration
    ↓
argmax + per-action thresholds      # Convert to discrete actions
    ↓
run-length encoding                 # Find segments
    ↓
min duration filtering              # Remove too-short segments
    ↓
remove_overlaps_by_confidence()     # Resolve conflicts
    ↓
robustify()                        # Fill missing videos
    ↓
submission
```

## 📊 Data Flow

```
CSV files (train.csv, test.csv)
    ↓
load_train_test_data()
    ↓
generate_mouse_data()  ← yields tracking + labels
    ↓
transform_single() / transform_pair()  ← feature engineering
    ↓
StratifiedSubsetClassifierWEval.fit()  ← training
    ↓
predict_proba()
    ↓
predict_multiclass_with_confidence()  ← postprocessing
    ↓
remove_overlaps_by_confidence()
    ↓
robustify()
    ↓
submission.csv
```

## 🚀 Usage Examples

### Quick Start

```python
from MABe import *

# Load
train, test = load_train_test_data(config.TRAIN_CSV, config.TEST_CSV)

# Generate features
for switch, data, meta, labels in generate_mouse_data(train, 'train'):
    X = transform_single(data, body_parts, fps, arena_dims)

# Train
model = StratifiedSubsetClassifierWEval(_make_xgb())
model.fit(X, y)

# Predict
pred = model.predict_proba(X_test)
```

### Advanced Ensemble

```python
models = [_make_lgbm(), _make_xgb(), _make_cb()]
weights = optimize_ensemble_weights(val_preds, val_labels)
final_pred = np.average(all_preds, weights=weights)
```

## 🎯 Advantages of This Structure

1. **Modularity**: Dễ test, debug, maintain từng phần
2. **Reusability**: Functions có thể dùng lại cho nhiều tasks
3. **Scalability**: Dễ thêm models, features mới
4. **Readability**: Code ngắn, rõ ràng, có docstrings
5. **Reproducibility**: Config tập trung, SEED cố định
6. **Performance**: Memory-efficient, GPU-aware
7. **Collaboration**: Nhiều người có thể work on different modules

## 📝 Next Steps

1. **train.py**: Full training pipeline script
2. **inference.py**: Full inference pipeline script
3. **CLI**: Add argparse for command-line usage
4. **Logging**: Add proper logging instead of prints
5. **Tests**: Unit tests for each module
6. **Documentation**: More detailed docstrings
7. **Optimization**: Profile and optimize bottlenecks

## 💡 Tips

- Mỗi file có một trách nhiệm rõ ràng
- Import chỉ những gì cần (avoid circular imports)
- Functions nhỏ, focused (single responsibility)
- Sử dụng type hints khi có thể
- Docstrings cho tất cả public functions
- Config values thay vì hardcode magic numbers
