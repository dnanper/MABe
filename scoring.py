"""
Scoring and evaluation functions for MABe competition
"""
import pandas as pd
import polars as pl
import numpy as np
import json
from collections import defaultdict
from sklearn.metrics import f1_score, log_loss
from scipy.optimize import minimize


class HostVisibleError(Exception):
    pass


def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    """Calculate F1 score for a single lab"""
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    for video in lab_solution['video_id'].unique():
        active_labels: str = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()
        active_labels: set[str] = set(json.loads(active_labels))
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():
            if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
                continue
           
            new_frames = set(range(row['start_frame'], row['stop_frame']))
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                raise HostVisibleError('Multiple predictions for the same frame from one agent/target pair')
            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)
    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        matched_label_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    action_f1s = []
    for action in distinct_actions:
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            action_f1s.append(
                (1 + beta**2) * tps[action] / 
                ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action])
            )
    return sum(action_f1s) / len(action_f1s)


def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
    """Calculate F-beta score for mouse behavior prediction"""
    if len(solution) == 0 or len(submission) == 0:
        raise ValueError('Missing solution or submission data')

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']

    for col in expected_cols:
        if col not in solution.columns:
            raise ValueError(f'Solution is missing column {col}')
        if col not in submission.columns:
            raise ValueError(f'Submission is missing column {col}')

    solution: pl.DataFrame = pl.DataFrame(solution)
    submission: pl.DataFrame = pl.DataFrame(submission)
    assert (solution['start_frame'] <= solution['stop_frame']).all()
    assert (submission['start_frame'] <= submission['stop_frame']).all()
    solution_videos = set(solution['video_id'].unique())
    submission = submission.filter(pl.col('video_id').is_in(solution_videos))

    solution = solution.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('label_key'),
    )
    submission = submission.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('prediction_key'),
    )

    lab_scores = []
    for lab in solution['lab_id'].unique():
        lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    """Main scoring function"""
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)

# increase
def optimize_ensemble_weights(preds_list, y_true):
    """
    Find optimal ensemble weights to minimize LogLoss.
    preds_list: list of prediction arrays (N_samples,), one per model
    y_true: true labels (N_samples,)
    """
    if len(preds_list) == 1:
        return [1.0]
    
    predictions = np.column_stack(preds_list)
    n_models = predictions.shape[1]
    
    def loss_func(weights):
        weights = np.array(weights)
        if np.sum(weights) == 0:
            return 1e9
        weights /= np.sum(weights)
        
        final_pred = np.dot(predictions, weights)
        final_pred = np.clip(final_pred, 1e-15, 1 - 1e-15)
        
        return log_loss(y_true, final_pred, labels=[0, 1])
    
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * n_models
    initial_weights = [1.0 / n_models] * n_models
    
    res = minimize(loss_func, initial_weights, method='SLSQP', 
                   bounds=bounds, constraints=cons, tol=1e-4)
    
    opt_weights = res.x / np.sum(res.x)
    return list(opt_weights)

# decrease
def optimize_thresholds_per_action(val_preds_map, val_labels_map, 
                                   threshold_range=(0.10, 0.60), step=0.01):
    """
    Optimize threshold for each action separately using grid search.
    
    Args:
        val_preds_map: dict {action: list of prediction arrays}
        val_labels_map: dict {action: list of label arrays}
        threshold_range: tuple (min, max) for threshold search
        step: grid search step size
        
    Returns:
        dict {action: optimal_threshold}
    """
    optimal_thresholds = {}
    
    print("\n" + "="*40)
    print("OPTIMIZING THRESHOLDS")
    print("="*40)
    
    for action in val_preds_map.keys():
        y_pred_proba = np.concatenate(val_preds_map[action])
        y_true = np.concatenate(val_labels_map[action])
        
        if y_true.sum() == 0:
            optimal_thresholds[action] = 0.27
            continue
            
        best_f1 = -1
        best_thresh = 0.27
        
        for thresh in np.arange(threshold_range[0], threshold_range[1], step):
            y_pred_bin = (y_pred_proba >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred_bin, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        optimal_thresholds[action] = float(best_thresh)
        print(f"Action: {action:<15} | Best Thresh: {best_thresh:.2f} | Best F1: {best_f1:.4f}")
        
    return optimal_thresholds
