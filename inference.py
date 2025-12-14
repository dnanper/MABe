"""
Inference loop for generating predictions on test set
"""
import json
import os
import gc
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from config import DROP_BODY_PARTS
from utils import _fps_from_meta
from feature_engineering import transform_single, transform_pair
from data_loader import generate_mouse_data
from postprocessing import (
    predict_multiclass_with_confidence, 
    remove_overlaps_by_confidence, 
    robustify,
    compute_action_properties_from_df
)


model_save_dir = "/kaggle/input/mabe-ver-5/pytorch/default/1/results_5/saved_models"


# ==================== MODEL PRUNING ====================

def prune_models_from_weights_file(weights_path, model_files, threshold=0.01):
    """
    Load weights file and return list of model names that have weight >= threshold.
    
    Args:
        weights_path: Path to ensemble_weights.json
        model_files: List of .pkl filenames
        threshold: Minimum weight to keep model (default 0.01)
    
    Returns:
        List of model names (without .pkl extension) to keep
    """
    if not os.path.exists(weights_path):
        # No weights file - keep all models
        return [f.replace('.pkl', '') for f in model_files]
    
    try:
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        
        # Filter models with weight >= threshold
        keep_models = [m for m, w in weights.items() if w >= threshold]
        return keep_models
    except:
        # Error reading weights - keep all models
        return [f.replace('.pkl', '') for f in model_files]


# ==================== INFERENCE FUNCTION ====================

def load_models_and_predict(section_idx, body_parts_tracked_str, switch_mode, test_subset, 
                            action_thresholds=None, action_properties=None):
    """
    Load trained models and generate predictions for test subset.
    
    Args:
        section_idx: Configuration section index
        body_parts_tracked_str: JSON string of body parts
        switch_mode: 'single' or 'pair'
        test_subset: Test data subset
        action_thresholds: Optional dict of action-specific thresholds
        action_properties: Optional dict of action properties
        
    Returns:
        List of submission DataFrames
    """
    bp_config_name = f"config_{section_idx}"
    model_base_path = os.path.join(model_save_dir, bp_config_name, switch_mode)
    
    if not os.path.exists(model_base_path): 
        return []

    # --- 1. Load Models & Weights ---
    trained_models_map = {} 
    weights_map = {} 

    try:
        available_actions = [d for d in os.listdir(model_base_path) 
                           if os.path.isdir(os.path.join(model_base_path, d))]
    except: 
        return []
    
    # Filter by valid actions if defined globally
    if 'VALID_ACTIONS_GLOBAL' in globals():
        available_actions = [a for a in available_actions if a in VALID_ACTIONS_GLOBAL]

    for action in available_actions:
        action_dir = os.path.join(model_base_path, action)
        model_files = [f for f in os.listdir(action_dir) if f.endswith('.pkl')]
        if not model_files: 
            continue
        
        w_path = os.path.join(action_dir, "ensemble_weights.json")
        current_weights = {}
        if os.path.exists(w_path):
            try:
                with open(w_path, 'r') as f: 
                    current_weights = json.load(f)
            except: 
                pass
        
        # PRUNING LOGIC: Only keep models with weight >= threshold
        models_to_keep_names = prune_models_from_weights_file(w_path, model_files, threshold=0.00)
        keep_set = set(models_to_keep_names)
        
        loaded_models = []
        loaded_weights = []
        
        for f in model_files:
            m_name = f.replace('.pkl', '')
            # Skip if pruned
            if m_name not in keep_set: 
                continue
            
            try:
                m = joblib.load(os.path.join(action_dir, f))
                loaded_models.append(m)
                loaded_weights.append(current_weights.get(m_name, 1.0))
            except: 
                pass
        
        if loaded_models:
            # Normalize weights after pruning
            total_w = sum(loaded_weights)
            if total_w > 0: 
                loaded_weights = [w/total_w for w in loaded_weights]
            else: 
                loaded_weights = [1.0/len(loaded_models)] * len(loaded_models)
            
            trained_models_map[action] = loaded_models
            weights_map[action] = loaded_weights

    # --- 2. Generate & Predict ---
    submission_parts = []
    
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked 
                                if b not in DROP_BODY_PARTS]
    except: 
        return []

    generator = generate_mouse_data(
        test_subset, 'test',
        generate_single=(switch_mode == 'single'),
        generate_pair=(switch_mode == 'pair')
    )
    
    fps_lookup = (test_subset[['video_id','frames_per_second']]
                    .drop_duplicates('video_id')
                    .set_index('video_id')['frames_per_second'].to_dict())

    for switch_te, data_te, meta_te, actions_te in tqdm(generator, 
                                                         desc=f"Infer {bp_config_name} {switch_mode}", 
                                                         leave=False):
        try:
            # Check if any models available for actions in this clip
            relevant_actions = [a for a in actions_te if a in trained_models_map]
            if not relevant_actions:
                del data_te
                gc.collect()
                continue

            arena_w = meta_te['arena_width_cm'].iloc[0]
            arena_h = meta_te['arena_height_cm'].iloc[0]
            arena_dims = (float(arena_w), float(arena_h))

            fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)
            
            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked, fps_i, 
                                       arena_dims).astype(np.float32)
            else:
                X_te = transform_pair(data_te, body_parts_tracked, fps_i)
            
            del data_te
            pred = pd.DataFrame(index=meta_te.video_frame)
            
            for action in relevant_actions:
                models = trained_models_map[action]
                weights = weights_map[action]
                
                probs = []
                for i, mdl in enumerate(models):
                    try:
                        p = mdl.predict_proba(X_te)[:, 1]
                        if np.isnan(p).any(): 
                            p = np.nan_to_num(p, nan=0.0)
                        probs.append(p * weights[i])
                    except: 
                        pass
                
                if probs:
                    # Sum weighted probabilities
                    pred[action] = np.sum(probs, axis=0)
            
            del X_te
            gc.collect()
            
            if pred.shape[1] > 0:
                sub_part = predict_multiclass_with_confidence(
                    pred, 
                    meta_te, 
                    action_thresholds=action_thresholds,
                    action_properties=action_properties,
                    min_duration=3
                )
                if len(sub_part) > 0:
                    submission_parts.append(sub_part)
                    
        except Exception as e:
            gc.collect()

    return submission_parts


# ==================== MAIN INFERENCE LOOP ====================

def run_inference_loop(test, body_parts_tracked_list, train=None, 
                      train_annot_dir=None, output_file='submission.csv'):
    """
    Main inference loop to generate predictions for test set.
    
    Args:
        test: Test DataFrame
        body_parts_tracked_list: List of body part configurations
        train: Training DataFrame (optional, for computing action properties)
        train_annot_dir: Training annotation directory (optional)
        output_file: Output CSV filename
        
    Returns:
        Final submission DataFrame
    """
    # 1. Load Thresholds & Properties
    ACTION_PROPERTIES = None
    if train is not None and train_annot_dir and os.path.exists(train_annot_dir):
        ACTION_PROPERTIES = compute_action_properties_from_df(train, train_annot_dir)
    else:
        # Try to load from saved file
        if os.path.exists("/kaggle/working/action_properties.json"):
            with open("/kaggle/working/action_properties.json", 'r') as f:
                ACTION_PROPERTIES = json.load(f)

    OPTIMAL_THRESHOLDS = defaultdict(lambda: 0.27)
    if os.path.exists("/kaggle/working/optimal_thresholds.json"):
        with open("/kaggle/working/optimal_thresholds.json", 'r') as f:
            OPTIMAL_THRESHOLDS = json.load(f)

    submission_list = []
    unique_test_bp_set = set(test.body_parts_tracked.unique())

    print(f"Test set configurations: {len(unique_test_bp_set)}")

    for section, body_parts_tracked_str in enumerate(body_parts_tracked_list):
        if body_parts_tracked_str not in unique_test_bp_set:
            continue
            
        print(f"\n[{section}] Processing Config: config_{section}")
        test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
        
        # --- Single ---
        print(f"   -> Single Inference...")
        single_preds = load_models_and_predict(
            section, body_parts_tracked_str, 'single', test_subset,
            action_thresholds=OPTIMAL_THRESHOLDS,
            action_properties=ACTION_PROPERTIES
        )
        submission_list.extend(single_preds)
        gc.collect()
        
        # --- Pair ---
        print(f"   -> Pair Inference...")
        pair_preds = load_models_and_predict(
            section, body_parts_tracked_str, 'pair', test_subset,
            action_thresholds=OPTIMAL_THRESHOLDS,
            action_properties=ACTION_PROPERTIES
        )
        submission_list.extend(pair_preds)
        gc.collect()

    # ==================== FINAL SUBMISSION GENERATION ====================
    if len(submission_list) > 0:
        submission = pd.concat(submission_list, ignore_index=True)
        
        # 1. Resolve overlaps intelligently (based on confidence)
        print(f"Resolving overlaps for {len(submission)} predictions...")
        submission = remove_overlaps_by_confidence(submission)
        print(f"Predictions after overlap removal: {len(submission)}")
        
    else:
        print("Warning: No predictions made. Creating dummy submission.")
        submission = pd.DataFrame({
            'video_id': [test.video_id.iloc[0]],
            'agent_id': ['mouse1'],
            'target_id': ['self'],
            'action': ['other'],
            'start_frame': [0],
            'stop_frame': [1]
        })

    # 2. Robustify (fill missing videos) & Save
    submission_robust = robustify(submission, test, 'test')
    submission_robust.index.name = 'row_id'
    submission_robust.to_csv(output_file)
    print(f"\nSubmission created: {len(submission_robust)} predictions saved to {output_file}")
    
    return submission_robust


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example: Run inference on test set
    # from data_loader import load_train_test_data
    # train, test = load_train_test_data()
    # body_parts_tracked_list = test.body_parts_tracked.unique().tolist()
    #
    # # Run inference
    # submission = run_inference_loop(
    #     test, 
    #     body_parts_tracked_list,
    #     train=train,
    #     train_annot_dir="/kaggle/input/MABe-mouse-behavior-detection/train_annotation",
    #     output_file='submission.csv'
    # )
    
    print("Inference module loaded. Use run_inference_loop() to generate predictions.")
