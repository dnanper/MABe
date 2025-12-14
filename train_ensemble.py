"""
Ensemble training pipeline with multiple models
"""
import json
import os
import gc
import joblib
import numpy as np
import pandas as pd
from time import perf_counter
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from config import SEED, USE_GPU
from utils import _make_lgbm, _make_xgb, _make_cb, _fps_from_meta
from models import StratifiedSubsetClassifier, StratifiedSubsetClassifierWEval
from scoring import optimize_ensemble_weights
from feature_engineering import transform_single, transform_pair


BASE_SAVE_DIR = "/kaggle/working/saved_models"

# Global validation results collector
VAL_RESULTS_COLLECTOR = {'preds': {}, 'labels': {}}


def create_model_ensemble(n_samples, use_gpu=USE_GPU):
    """
    Create ensemble of diverse models for training.
    
    Args:
        n_samples: Number of samples for stratified subset
        use_gpu: Whether to use GPU-accelerated models
        
    Returns:
        tuple: (models_list, model_names_list)
    """
    models = []
    model_names = []
    
    # ==================== BASE MODELS (CPU/GPU) ====================
    
    # 1. LightGBM #1 - Balanced
    models.append(make_pipeline(
        StratifiedSubsetClassifier(_make_lgbm(
            n_estimators=225, learning_rate=0.07, min_child_samples=40,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8, 
            verbose=-1, gpu_use_dp=use_gpu
        ), n_samples)
    ))
    model_names.append('lgbm_225')
    
    # 2. LightGBM #2 - Deep
    models.append(make_pipeline(
        StratifiedSubsetClassifier(_make_lgbm(
            n_estimators=150, learning_rate=0.1, min_child_samples=20,
            num_leaves=63, max_depth=8, subsample=0.7, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=0.1, verbose=-1, gpu_use_dp=use_gpu
        ), n_samples and int(n_samples/1.25))
    ))
    model_names.append('lgbm_150')
    
    # 3. LightGBM #3 - Very Deep
    models.append(make_pipeline(
        StratifiedSubsetClassifier(_make_lgbm(
            n_estimators=100, learning_rate=0.05, min_child_samples=30,
            num_leaves=127, max_depth=10, subsample=0.75, 
            verbose=-1, gpu_use_dp=use_gpu
        ), n_samples and int(n_samples/1.66))
    ))
    model_names.append('lgbm_100')
    
    # 4. LightGBM #4 - Extra Trees
    models.append(make_pipeline(
        StratifiedSubsetClassifier(_make_lgbm(
            n_estimators=200, learning_rate=0.05, num_leaves=63,
            extra_trees=True,
            colsample_bytree=0.7, verbose=-1, gpu_use_dp=use_gpu
        ), n_samples)
    ))
    model_names.append('lgbm_extra')
    
    # 5. XGBoost #1 - Balanced
    xgb0 = _make_xgb(
        n_estimators=180, learning_rate=0.08, max_depth=6,
        min_child_weight=8 if use_gpu else 5, 
        gamma=1.0 if use_gpu else 0.,
        subsample=0.8, colsample_bytree=0.8, 
        single_precision_histogram=use_gpu,
        verbosity=1
    )
    models.append(make_pipeline(
        StratifiedSubsetClassifier(xgb0, n_samples and int(n_samples/1.2))
    ))
    model_names.append('xgb_180')
    
    # 6. XGBoost #2 - Shallow (Strong regularization)
    xgb_shallow = _make_xgb(
        n_estimators=300, learning_rate=0.1, max_depth=4,
        reg_lambda=2.0,
        random_state=SEED + 2, verbosity=1
    )
    models.append(make_pipeline(
        StratifiedSubsetClassifier(xgb_shallow, n_samples)
    ))
    model_names.append('xgb_300')
    
    # 7. XGBoost #3 - DART Booster
    xgb_dart = _make_xgb(
        n_estimators=250, learning_rate=0.05, max_depth=6,
        booster='dart', rate_drop=0.1, skip_drop=0.5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED + 3, verbosity=1
    )
    models.append(make_pipeline(
        StratifiedSubsetClassifier(xgb_dart, n_samples)
    ))
    model_names.append('xgb_250')
    
    # 8. CatBoost #1 - Basic
    cb_est = _make_cb(
        iterations=120, learning_rate=0.1, depth=6,
        verbose=50, allow_writing_files=False
    )
    models.append(make_pipeline(
        StratifiedSubsetClassifier(cb_est, n_samples)
    ))
    model_names.append('cat_120')
    
    # ==================== GPU-ONLY MODELS ====================
    if use_gpu:
        # 9. XGBoost Large #1 - Loss-guide growth
        xgb1 = XGBClassifier(
            random_state=SEED, booster="gbtree", tree_method="gpu_hist",
            n_estimators=2100, learning_rate=0.05, grow_policy="lossguide",
            max_leaves=255, max_depth=0, min_child_weight=10, gamma=0.0,
            subsample=0.90, colsample_bytree=1.00, colsample_bylevel=0.85,
            reg_alpha=0.0, reg_lambda=1.0, max_bin=256,
            single_precision_histogram=True, verbosity=1
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifierWEval(
                xgb1, n_samples and int(n_samples/2.),
                random_state=SEED, valid_size=0.10, val_cap_ratio=0.25,
                es_rounds="auto", es_metric="auto"
            )
        ))
        model_names.append('xgb1')
        
        # 10. XGBoost Large #2 - Standard growth
        xgb2 = XGBClassifier(
            random_state=SEED, booster="gbtree", tree_method="gpu_hist",
            n_estimators=1500, learning_rate=0.06, max_depth=7,
            min_child_weight=12, subsample=0.70, colsample_bytree=0.80,
            reg_alpha=0.0, reg_lambda=1.5, max_bin=256,
            single_precision_histogram=True, verbosity=1
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifierWEval(
                xgb2, n_samples and int(n_samples/1.5),
                random_state=SEED, valid_size=0.10, val_cap_ratio=0.25,
                es_rounds="auto", es_metric="auto"
            )
        ))
        model_names.append('xgb2')
        
        # 11. XGBoost Deep - High capacity
        xgb_deep = _make_xgb(
            n_estimators=1000, learning_rate=0.02, max_depth=10,
            min_child_weight=5, subsample=0.75, colsample_bytree=0.6,
            reg_lambda=5.0, reg_alpha=1.0,
            random_state=SEED + 10, verbosity=1
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifier(xgb_deep, n_samples and int(n_samples/1.5))
        ))
        model_names.append('xgb_deep')
        
        # 12. XGBoost Wide - Loss-guide with many leaves
        xgb_wide = _make_xgb(
            n_estimators=2000, learning_rate=0.03, max_depth=0,
            grow_policy='lossguide', max_leaves=128,
            min_child_weight=20, subsample=0.85, colsample_bytree=0.85,
            random_state=SEED + 11, verbosity=1
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifier(xgb_wide, n_samples and int(n_samples/1.5))
        ))
        model_names.append('xgb_wide')
        
        # 13. CatBoost Large #1 - Bayesian bootstrap
        cb1 = CatBoostClassifier(
            random_seed=SEED, task_type="GPU", devices="0",
            iterations=4100, learning_rate=0.03, depth=8, l2_leaf_reg=6.0,
            bootstrap_type="Bayesian", bagging_temperature=0.5,
            random_strength=0.5, loss_function="Logloss",
            eval_metric="PRAUC:type=Classic", auto_class_weights="Balanced",
            border_count=64, verbose=250, allow_writing_files=False
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifierWEval(
                cb1, n_samples and int(n_samples/2.0),
                random_state=SEED, valid_size=0.10, val_cap_ratio=0.25,
                es_rounds="auto", es_metric="auto"
            )
        ))
        model_names.append('cat_bay')
        
        # 14. CatBoost Deep - Very deep trees
        cb_deep = CatBoostClassifier(
            random_seed=SEED + 1, task_type="GPU", devices="0",
            iterations=3000, learning_rate=0.03, depth=10, l2_leaf_reg=8.0,
            loss_function="Logloss", eval_metric="PRAUC:type=Classic",
            auto_class_weights="Balanced", border_count=64, 
            verbose=250, allow_writing_files=False
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifierWEval(
                cb_deep, n_samples and int(n_samples/2.5),
                random_state=SEED, valid_size=0.10, val_cap_ratio=0.25,
                es_rounds="auto", es_metric="auto"
            )
        ))
        model_names.append('cat_deep')
        
        # 15. Extra Trees - Sklearn
        et_model = ExtraTreesClassifier(
            n_estimators=200, max_depth=18, min_samples_split=20, 
            max_features='sqrt',
            random_state=SEED, n_jobs=-1
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifier(et_model, n_samples and int(n_samples/1.5))
        ))
        model_names.append('et')
        
        # 16. Random Forest - Balanced
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            class_weight='balanced',
            random_state=SEED, n_jobs=-1
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifier(rf_model, n_samples and int(n_samples/1.5))
        ))
        model_names.append('rf')
        
        # 17. XGBoost Focal-style - High regularization
        xgb_focal_sim = _make_xgb(
            n_estimators=400, learning_rate=0.04, max_depth=5,
            gamma=2.0, min_child_weight=10,
            reg_alpha=2.0, reg_lambda=2.0,
            scale_pos_weight=1.0,
            verbosity=1
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifier(xgb_focal_sim, n_samples)
        ))
        model_names.append('xgb_focal_sim')
    
    return models, model_names


def submit_ensemble(body_parts_tracked_str, switch_tr, X_tr, X_tr_aug, label, meta, 
                   config_name, results_collector, test_data, n_samples=1_500_000, 
                   base_save_dir=BASE_SAVE_DIR):
    """
    Train ensemble of models and generate predictions.
    
    Args:
        body_parts_tracked_str: JSON string of body parts being tracked
        switch_tr: 'single' or 'pair'
        X_tr: Training features
        X_tr_aug: Augmented training features
        label: Training labels DataFrame
        meta: Metadata DataFrame
        config_name: Configuration name for saving
        results_collector: Dict to collect validation results
        test_data: Test dataset DataFrame
        n_samples: Number of samples for training
        base_save_dir: Base directory for saving models
        
    Returns:
        list: Submission predictions
    """
    try:
        bp_config_name = config_name
        bp_list = json.loads(body_parts_tracked_str)
    except Exception as e:
        print(f"Error parsing body parts: {e}")
        bp_config_name = "unknown_config"
    
    # Create model ensemble
    models, model_names = create_model_ensemble(n_samples, use_gpu=USE_GPU)
    
    print(f"\n{'='*60}")
    print(f"Training {len(models)} models for {bp_config_name} | {switch_tr.upper()}")
    print(f"{'='*60}")
    
    model_list = []  # Will contain: (action, trained_models, weights)
    submission_list = []
    
    # ==================== TRAIN PER ACTION ====================
    for action in label.columns:
        action_mask = ~label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)
        
        if len(y_action) == 0:
            continue
        
        print(f"\n>>> Action: {action} (Total samples: {len(y_action)}, "
              f"Positive: {y_action.sum()})")
        
        meta_masked = meta.iloc[action_mask]
        
        # Split train/val for weight optimization
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, val_idx = next(sss.split(meta_masked, y_action))
        
        X_masked = X_tr.iloc[action_mask]
        X_masked_aug = X_tr_aug.iloc[action_mask]
        
        X_train_fold = X_masked.iloc[train_idx]
        y_train_fold = y_action[train_idx]
        X_val_fold = X_masked.iloc[val_idx]
        y_val_fold = y_action[val_idx]
        
        # ==================== SMART BALANCING ====================
        NEG_POS_RATIO = 3.0
        
        neg_indices = np.where(y_train_fold == 0)[0]
        pos_indices = np.where(y_train_fold == 1)[0]
        
        n_neg = len(neg_indices)
        n_pos = len(pos_indices)
        
        target_n_pos = int(n_neg / NEG_POS_RATIO)
        
        if n_pos > 0 and n_pos < target_n_pos:
            print(f"   -> Balancing: Orig Pos={n_pos}, Neg={n_neg}. "
                  f"Target Pos={target_n_pos} (Ratio 1:{NEG_POS_RATIO})")
            
            # Create pool (Original + Augmented)
            X_pos_orig = X_train_fold.iloc[pos_indices]
            X_pos_rot = X_masked_aug.iloc[train_idx].iloc[pos_indices]
            
            X_pos_pool = pd.concat([X_pos_orig, X_pos_rot], axis=0)
            pool_size = len(X_pos_pool)
            
            replace_flag = (pool_size < target_n_pos)
            random_indices = np.random.choice(pool_size, size=target_n_pos, 
                                            replace=replace_flag)
            
            X_pos_balanced = X_pos_pool.iloc[random_indices]
            y_pos_balanced = np.ones(target_n_pos, dtype=int)
            
            X_neg = X_train_fold.iloc[neg_indices]
            y_neg = y_train_fold[neg_indices]
            
            X_train_final = pd.concat([X_neg, X_pos_balanced], axis=0)
            y_train_final = np.concatenate([y_neg, y_pos_balanced])
            
            # Shuffle
            shuffle_idx = np.random.permutation(len(y_train_final))
            X_train_final = X_train_final.iloc[shuffle_idx]
            y_train_final = y_train_final[shuffle_idx]
        else:
            X_train_final = X_train_fold
            y_train_final = y_train_fold
        
        # ==================== TRAIN ALL MODELS ====================
        trained = []
        val_predictions = []
        
        for model_idx, m in enumerate(models):
            model_name_str = model_names[model_idx]
            print(f"\n{'='*10} [{bp_config_name} | {switch_tr.upper()} | "
                  f"{action} | {model_name_str}] {'='*10}")
            
            m_clone = clone(m)
            try:
                t0 = perf_counter()
                m_clone.fit(X_train_final, y_train_final)
                
                val_pred = m_clone.predict_proba(X_val_fold)[:, 1]
                val_predictions.append(val_pred)
                
                dt = perf_counter() - t0
                
                # Save model
                save_dir = os.path.join(base_save_dir, bp_config_name, switch_tr, action)
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.join(save_dir, f"{model_name_str}.pkl")
                joblib.dump(m_clone, filename)
                
                print(f"--> DONE: {model_name_str} ({dt:.1f}s)")
                trained.append(m_clone)
                
            except Exception as e:
                print(f"ERROR training {model_name_str}: {e}")
                continue
        
        # ==================== OPTIMIZE WEIGHTS ====================
        best_weights = None
        if len(trained) > 0:
            print(f"\n--> Optimizing Ensemble Weights for {action}...")
            best_weights = optimize_ensemble_weights(val_predictions, y_val_fold)
            
            # Save weights
            weight_map = {
                model_names[i]: w
                for i, w in enumerate(best_weights)
                if i < len(model_names)
            }
            
            with open(os.path.join(save_dir, "ensemble_weights.json"), "w") as f:
                json.dump(weight_map, f)
            print(f"--> Best Weights: {weight_map}")
            
            # Collect results for threshold optimization
            if best_weights is not None:
                preds_stack = np.column_stack(val_predictions)
                final_val_prob = np.average(preds_stack, axis=1, weights=best_weights)
                
                if action not in results_collector['preds']:
                    results_collector['preds'][action] = []
                    results_collector['labels'][action] = []
                
                results_collector['preds'][action].append(final_val_prob)
                results_collector['labels'][action].append(y_val_fold)
        
        if trained:
            model_list.append((action, trained, best_weights))
    
    del X_tr
    gc.collect()
    
    # ==================== TEST INFERENCE ====================
    print(f"\n{'='*60}")
    print(f"Running inference on test set...")
    print(f"{'='*60}")
    
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked 
                             if b not in DROP_BODY_PARTS]
    
    test_subset = test_data[test_data.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(
        test_subset, 'test',
        generate_single=(switch_tr == 'single'),
        generate_pair=(switch_tr == 'pair')
    )
    
    fps_lookup = (test_subset[['video_id', 'frames_per_second']]
                  .drop_duplicates('video_id')
                  .set_index('video_id')['frames_per_second'].to_dict())
    
    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        arena_w = meta_te['arena_width_cm'].iloc[0]
        arena_h = meta_te['arena_height_cm'].iloc[0]
        arena_dims = (float(arena_w), float(arena_h))
        
        try:
            fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)
            
            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked, fps_i, 
                                       arena_dims).astype(np.float32)
            else:
                X_te = transform_pair(data_te, body_parts_tracked, fps_i)
            
            del data_te
            pred = pd.DataFrame(index=meta_te.video_frame)
            
            # Predict with weights
            for action, trained, weights in model_list:
                if action in actions_te:
                    probs = []
                    for mdl in trained:
                        probs.append(mdl.predict_proba(X_te)[:, 1])
                    
                    if weights is not None and len(weights) == len(probs):
                        pred[action] = np.average(probs, axis=0, weights=weights)
                    else:
                        pred[action] = np.mean(probs, axis=0)
            
            del X_te
            gc.collect()
            
            if pred.shape[1] != 0:
                submission_list.append(
                    predict_multiclass_with_confidence(pred, meta_te)
                )
        except Exception as e:
            print(f"Error in inference: {e}")
            try:
                del data_te
            except:
                pass
            gc.collect()
    
    return submission_list
