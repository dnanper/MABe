"""
Training loop with stratified sampling and augmentation
"""
import json
import gc
import numpy as np
import pandas as pd

from config import SEED, DROP_BODY_PARTS
from utils import _fps_from_meta, rotate_xy_dataframe
from feature_engineering import transform_single, transform_pair
from data_loader import generate_mouse_data
from scoring import optimize_thresholds_per_action
from train_ensemble import submit_ensemble, VAL_RESULTS_COLLECTOR


# ==================== STRATIFIED DOWNSAMPLING ====================

def stratified_downsample_with_aug(X, y, m, X_aug, max_rows):
    """
    Downsample intelligently: Prioritize keeping Positive samples (especially rare actions).
    Only remove excess Negative (background) or overly common class samples.
    
    Args:
        X: Feature DataFrame
        y: Label DataFrame
        m: Metadata DataFrame
        X_aug: Augmented feature DataFrame
        max_rows: Maximum number of rows to keep
        
    Returns:
        Downsampled (X, y, m, X_aug)
    """
    n_total = len(y)
    if n_total <= max_rows:
        return X, y, m, X_aug

    print(f"    -> Stratified Downsampling from {n_total} to {max_rows}...")
    
    # 1. Identify rows with Positive labels (at least 1 action = 1)
    if isinstance(y, pd.DataFrame):
        row_sums = y.sum(axis=1)
        pos_indices = np.where(row_sums > 0)[0]
        neg_indices = np.where(row_sums == 0)[0]
    else:
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

    # 2. Sampling strategy
    n_pos = len(pos_indices)
    
    if n_pos >= max_rows:
        # Rare case: Too many Positives (e.g., all Sleep)
        # Randomly sample from Positives
        selected_indices = np.random.choice(pos_indices, max_rows, replace=False)
    else:
        # Keep ALL Positives
        # Add Negatives to reach max_rows
        n_neg_needed = max_rows - n_pos
        n_neg_needed = min(n_neg_needed, len(neg_indices))
        
        neg_sampled = np.random.choice(neg_indices, n_neg_needed, replace=False)
        selected_indices = np.concatenate([pos_indices, neg_sampled])

    # 3. Shuffle to mix Pos/Neg
    np.random.shuffle(selected_indices)
    
    # 4. Filter data
    X_new = X.iloc[selected_indices].reset_index(drop=True)
    y_new = y.iloc[selected_indices].reset_index(drop=True)
    m_new = m.iloc[selected_indices].reset_index(drop=True)
    X_aug_new = X_aug.iloc[selected_indices].reset_index(drop=True)
    
    print(f"       Kept {len(pos_indices)} Positives and "
          f"{len(selected_indices)-len(pos_indices)} Negatives.")
    
    return X_new, y_new, m_new, X_aug_new


# ==================== MAIN TRAINING LOOP ====================

def run_training_loop(train, body_parts_tracked_list, drop_body_parts=DROP_BODY_PARTS,
                     max_samples=2_000_000, start_section=0, end_section=None):
    """
    Main training loop for all body part configurations.
    
    Args:
        train: Training DataFrame
        body_parts_tracked_list: List of body part configurations
        drop_body_parts: List of body parts to drop from dense configs
        max_samples: Maximum samples per configuration for memory management
        start_section: Start from this configuration index
        end_section: End at this configuration index (None = all)
        
    Returns:
        VAL_RESULTS_COLLECTOR with predictions and labels
    """
    if end_section is None:
        end_section = len(body_parts_tracked_list)
    
    for section in range(start_section, end_section):
        body_parts_tracked_str = body_parts_tracked_list[section]
        current_config_name = f"config_{section}"
        
        try:
            body_parts_tracked = json.loads(body_parts_tracked_str)
            print(f"\nSection {section}: {len(body_parts_tracked)} body parts")
            
            if len(body_parts_tracked) > 5:
                body_parts_tracked = [b for b in body_parts_tracked 
                                    if b not in drop_body_parts]

            train_subset = train[train.body_parts_tracked == body_parts_tracked_str]

            _fps_lookup = (
                train_subset[['video_id', 'frames_per_second']]
                .drop_duplicates('video_id')
                .set_index('video_id')['frames_per_second']
                .to_dict()
            )

            # Lists to collect data
            single_feats_parts, single_label_parts, single_meta_parts = [], [], []
            pair_feats_parts, pair_label_parts, pair_meta_parts = [], [], []
            single_feats_aug = []
            pair_feats_aug = []
            FIXED_AUG_ANGLE = 180

            for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
                fps_i = _fps_from_meta(meta, _fps_lookup, default_fps=30.0)
                arena_w = meta['arena_width_cm'].iloc[0]
                arena_h = meta['arena_height_cm'].iloc[0]
                arena_dims = (float(arena_w), float(arena_h))
                
                # 1. ORIGINAL DATA
                if switch == 'single':
                    Xi = transform_single(data, body_parts_tracked, fps_i, 
                                        arena_dims).astype(np.float32)
                    single_feats_parts.append(Xi)
                    single_label_parts.append(label)
                    single_meta_parts.append(meta)
                else:
                    Xi = transform_pair(data, body_parts_tracked, fps_i).astype(np.float32)
                    pair_feats_parts.append(Xi)
                    pair_label_parts.append(label)
                    pair_meta_parts.append(meta)

                # 2. AUGMENTED DATA (180° rotation)
                data_rot = rotate_xy_dataframe(data, FIXED_AUG_ANGLE)
        
                if switch == 'single':
                    Xi_rot = transform_single(data_rot, body_parts_tracked, fps_i, 
                                            arena_dims).astype(np.float32)
                    single_feats_aug.append(Xi_rot)
                else:
                    Xi_rot = transform_pair(data_rot, body_parts_tracked, fps_i).astype(np.float32)
                    pair_feats_aug.append(Xi_rot)

            # --- TRAIN SINGLE ---
            if len(single_feats_parts) > 0:
                X_tr = pd.concat(single_feats_parts, axis=0, ignore_index=True)
                y_tr = pd.concat(single_label_parts, axis=0, ignore_index=True)
                m_tr = pd.concat(single_meta_parts, axis=0, ignore_index=True)
                X_tr_aug = pd.concat(single_feats_aug, axis=0, ignore_index=True)
                
                del single_feats_parts, single_label_parts, single_meta_parts, single_feats_aug
                gc.collect()

                # Downsample if needed
                X_tr, y_tr, m_tr, X_tr_aug = stratified_downsample_with_aug(
                    X_tr, y_tr, m_tr, X_tr_aug, max_samples
                )

                print(f"  Single (Final): {X_tr.shape}")
                submit_ensemble(
                    body_parts_tracked_str, 'single', X_tr, X_tr_aug, y_tr, m_tr, 
                    config_name=current_config_name, 
                    results_collector=VAL_RESULTS_COLLECTOR
                )
                
                del X_tr, y_tr, X_tr_aug, m_tr
                gc.collect()

            # --- TRAIN PAIR ---
            if len(pair_feats_parts) > 0:
                X_tr = pd.concat(pair_feats_parts, axis=0, ignore_index=True)
                y_tr = pd.concat(pair_label_parts, axis=0, ignore_index=True)
                m_tr = pd.concat(pair_meta_parts, axis=0, ignore_index=True)
                X_tr_aug = pd.concat(pair_feats_aug, axis=0, ignore_index=True)

                del pair_feats_parts, pair_label_parts, pair_meta_parts, pair_feats_aug
                gc.collect()

                # Downsample if needed
                X_tr, y_tr, m_tr, X_tr_aug = stratified_downsample_with_aug(
                    X_tr, y_tr, m_tr, X_tr_aug, max_samples
                )

                print(f"  Pair (Final): {X_tr.shape}")
                submit_ensemble(
                    body_parts_tracked_str, 'pair', X_tr, X_tr_aug, y_tr, m_tr, 
                    config_name=current_config_name, 
                    results_collector=VAL_RESULTS_COLLECTOR
                )
                
                del X_tr, y_tr, X_tr_aug, m_tr
                gc.collect()

        except Exception as e:
            print(f'***Exception*** {str(e)}')
            import traceback
            traceback.print_exc()

        gc.collect()
    
    return VAL_RESULTS_COLLECTOR


# ==================== THRESHOLD OPTIMIZATION ====================

def save_optimal_thresholds(results_collector, output_path="/kaggle/working/optimal_thresholds.json"):
    """
    Optimize and save action-specific thresholds based on validation results.
    
    Args:
        results_collector: Dictionary with 'preds' and 'labels' keys
        output_path: Output JSON file path
    """
    final_thresholds = optimize_thresholds_per_action(
        results_collector['preds'], 
        results_collector['labels']
    )

    with open(output_path, "w") as f:
        json.dump(final_thresholds, f)
        
    print(f"Optimal thresholds saved to {output_path}")
    return final_thresholds


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example: Train all configurations
    # from data_loader import load_train_test_data
    # train, test = load_train_test_data()
    # body_parts_tracked_list = train.body_parts_tracked.unique().tolist()
    #
    # # Run training
    # results = run_training_loop(train, body_parts_tracked_list, 
    #                            max_samples=2_000_000, 
    #                            start_section=0)
    #
    # # Save optimal thresholds
    # save_optimal_thresholds(results)
    
    print("Training module loaded. Use run_training_loop() to start training.")
