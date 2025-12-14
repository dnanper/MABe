"""
Post-processing functions for predictions
"""
import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
from tqdm import tqdm
from config import DROP_BODY_PARTS


def adaptive_temporal_smoothing(pred_df, action_properties=None):
    """Adaptive temporal smoothing based on action duration"""
    if action_properties is None:
        return pred_df.rolling(window=5, min_periods=1, center=True).mean()
    
    smoothed = pred_df.copy()
    for action in pred_df.columns:
        typical_dur = action_properties.get(action, {}).get('median_duration', 30)
        
        if typical_dur < 15:
            window = 3
        elif typical_dur < 60:
            window = 5
        else:
            window = 9
            
        smoothed[action] = pred_df[action].rolling(window, min_periods=1, center=True).mean()
    return smoothed


def predict_multiclass_with_confidence(pred, meta, action_thresholds=None, 
                                      action_properties=None, min_duration=3):
    """
    Multiclass prediction with confidence scoring and post-processing.
    
    Args:
        pred: DataFrame of probability predictions
        meta: DataFrame with metadata (video_id, agent_id, etc.)
        action_thresholds: dict of action-specific thresholds
        action_properties: dict of action statistics
        min_duration: minimum duration in frames
        
    Returns:
        DataFrame with predictions (video_id, agent_id, target_id, action, start_frame, stop_frame, confidence)
    """
    if action_thresholds is None:
        action_thresholds = defaultdict(lambda: 0.27)
    
    # 1. Adaptive Smoothing
    pred_smoothed = adaptive_temporal_smoothing(pred, action_properties)
    
    # 2. Thresholding & Argmax
    argmax_action = np.argmax(pred_smoothed.values, axis=1)
    max_probs = pred_smoothed.values.max(axis=1)
    
    threshold_mask = np.zeros(len(pred_smoothed), dtype=bool)
    for i, action in enumerate(pred_smoothed.columns):
        thresh = action_thresholds.get(action, 0.27)
        threshold_mask |= ((argmax_action == i) & (max_probs >= thresh))
    
    final_actions = np.where(threshold_mask, argmax_action, -1)
    action_series = pd.Series(final_actions, index=meta.video_frame)
    
    # 3. Detect changes (Run-length encoding)
    changes_mask = (action_series != action_series.shift(1)).values
    idx_changes = action_series.index[changes_mask]
    vals_changes = action_series.values[changes_mask]
    conf_changes = max_probs[changes_mask]
    
    # Filter valid segments (val >= 0)
    valid_mask = vals_changes >= 0
    
    # Create segments
    starts = idx_changes[:-1][valid_mask[:-1]] if len(idx_changes) > 1 else []
    stops = idx_changes[1:][valid_mask[:-1]] if len(idx_changes) > 1 else []
    actions_idx = vals_changes[:-1][valid_mask[:-1]] if len(idx_changes) > 1 else []
    confs = conf_changes[:-1][valid_mask[:-1]] if len(idx_changes) > 1 else []
    
    # Handle last segment
    if len(vals_changes) > 0 and valid_mask[-1]:
        last_start = idx_changes[-1]
        last_stop = meta.video_frame.iloc[-1] + 1
        starts = np.append(starts, last_start)
        stops = np.append(stops, last_stop)
        actions_idx = np.append(actions_idx, vals_changes[-1])
        confs = np.append(confs, conf_changes[-1])

    if len(starts) == 0:
        return pd.DataFrame()

    # Create result DataFrame
    df_temp = pd.DataFrame({
        'start_frame': starts,
        'stop_frame': stops,
        'action_idx': actions_idx,
        'confidence': confs
    })
    
    # Add metadata
    df_temp['video_id'] = meta['video_id'].iloc[0]
    df_temp['agent_id'] = meta['agent_id'].iloc[0]
    df_temp['target_id'] = meta['target_id'].iloc[0]
    
    # Map action names
    cols = pred.columns
    df_temp['action'] = df_temp['action_idx'].apply(lambda x: cols[int(x)])
    
    # 4. Filter Min Duration (Adaptive)
    if action_properties:
        def check_dur(row):
            min_d = action_properties.get(row['action'], {}).get('min_duration', min_duration)
            return (row['stop_frame'] - row['start_frame']) >= min_d
        
        df_temp = df_temp[df_temp.apply(check_dur, axis=1)]
    else:
        df_temp = df_temp[(df_temp.stop_frame - df_temp.start_frame) >= min_duration]
        
    return df_temp.drop(columns=['action_idx'])


def remove_overlaps_by_confidence(submission):
    """Resolve overlapping predictions using confidence scores"""
    if submission.empty:
        return submission
    if 'confidence' not in submission.columns:
        return submission
    
    df = submission.sort_values(['video_id', 'agent_id', 'target_id', 'start_frame']).reset_index(drop=True)
    
    keep_indices = []
    
    for _, group in df.groupby(['video_id', 'agent_id', 'target_id']):
        indices = group.index.tolist()
        if not indices:
            continue
        
        accepted_segments = []
        
        for idx in indices:
            curr_start = df.at[idx, 'start_frame']
            curr_stop = df.at[idx, 'stop_frame']
            curr_conf = df.at[idx, 'confidence']
            
            conflict = False
            remove_list = []
            
            for i, (acc_start, acc_stop, acc_conf, acc_idx) in enumerate(accepted_segments):
                if max(curr_start, acc_start) < min(curr_stop, acc_stop):
                    if curr_conf > acc_conf:
                        remove_list.append(i)
                    else:
                        conflict = True
                        break
            
            if not conflict:
                for i in sorted(remove_list, reverse=True):
                    accepted_segments.pop(i)
                
                accepted_segments.append((curr_start, curr_stop, curr_conf, idx))
        
        keep_indices.extend([x[3] for x in accepted_segments])
        
    return df.loc[sorted(keep_indices)].reset_index(drop=True)


def fill_missing_video_realistic(video_id, behaviors, frame_range, action_properties):
    """Fill missing video with realistic dummy predictions"""
    start_frame, stop_frame = frame_range
    total_frames = stop_frame - start_frame
    if total_frames <= 0:
        return []
    
    segments = []
    for (agent, target), action_group in behaviors.groupby(['agent', 'target']):
        actions = action_group['action'].tolist()
        if not actions:
            continue
        
        durations = []
        for action in actions:
            median_dur = action_properties.get(action, {}).get('median_duration', 30) \
                         if action_properties else 30
            durations.append(median_dur)
        
        total_duration = sum(durations)
        
        if total_duration == 0:
            allocated = [total_frames // len(actions)] * len(actions)
        else:
            weights = np.array(durations) / total_duration
            allocated = (weights * total_frames).astype(int)
        
        diff = total_frames - allocated.sum()
        if diff != 0:
            allocated[-1] += diff
        
        current_frame = start_frame
        for action, n_frames in zip(actions, allocated):
            if n_frames > 0:
                segments.append({
                    'video_id': video_id,
                    'agent_id': agent,
                    'target_id': target,
                    'action': action,
                    'start_frame': current_frame,
                    'stop_frame': current_frame + n_frames,
                    'confidence': 0.05
                })
                current_frame += n_frames
                
    return segments


def robustify(submission, dataset, traintest, action_properties=None, traintest_directory=None):
    """
    Clean and fill submission with missing videos.
    
    Args:
        submission: DataFrame with predictions
        dataset: original dataset (train or test)
        traintest: 'train' or 'test'
        action_properties: dict of action statistics
        traintest_directory: path to tracking data
        
    Returns:
        Cleaned and complete submission DataFrame
    """
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    
    # 1. Clean invalid frames
    if not submission.empty:
        submission = submission[submission.start_frame < submission.stop_frame]
        
        if 'confidence' not in submission.columns:
            submission['confidence'] = 1.0
    
    # 2. Fill Missing Videos
    s_list = []
    
    existing_videos = set(submission.video_id.unique()) if not submission.empty else set()
    
    missing_mask = ~dataset['video_id'].isin(existing_videos)
    missing_rows = dataset[missing_mask]
    
    if len(missing_rows) > 0:
        print(f"Filling {len(missing_rows)} missing videos with realistic dummies...")
        
        for _, row in missing_rows.iterrows():
            lab_id = row['lab_id']
            video_id = row['video_id']
            
            try:
                path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
                vid = pd.read_parquet(path, columns=['video_frame'])
                start_frame = vid.video_frame.iloc[0]
                stop_frame = vid.video_frame.iloc[-1] + 1
                del vid
                
                if isinstance(row['behaviors_labeled'], str):
                    b_list = json.loads(row['behaviors_labeled'])
                    b_data = []
                    for b in b_list:
                        parts = b.split(',')
                        if len(parts) == 3:
                            b_data.append(parts)
                    
                    vid_behaviors = pd.DataFrame(b_data, columns=['agent', 'target', 'action'])
                    
                    if action_properties:
                        segs = fill_missing_video_realistic(
                            video_id, vid_behaviors, (start_frame, stop_frame), action_properties
                        )
                        s_list.extend(segs)
                        
            except Exception:
                pass

    # 3. Merge & Final Sort
    if s_list:
        dummy_df = pd.DataFrame(s_list)
        submission = pd.concat([submission, dummy_df], ignore_index=True)
        
    submission = submission.sort_values(['video_id', 'start_frame']).reset_index(drop=True)
    return submission


def compute_action_properties_from_df(train_df, annotation_dir):
    """Compute action statistics from training annotations"""
    print("Computing action properties from annotations...")
    action_durations = defaultdict(list)
    
    sample_videos = train_df.sample(n=min(100, len(train_df)), random_state=42)
    
    for _, row in tqdm(sample_videos.iterrows(), total=len(sample_videos), desc="Scanning Props"):
        try:
            lab_id = row.lab_id
            video_id = row.video_id
            path = f"{annotation_dir}/{lab_id}/{video_id}.parquet"
            if not os.path.exists(path):
                continue
            
            annot = pd.read_parquet(path)
            
            for action in annot['action'].unique():
                subset = annot[annot['action'] == action]
                durs = (subset['stop_frame'] - subset['start_frame']).values
                action_durations[action].extend(durs)
        except:
            continue
            
    properties = {}
    for action, durs in action_durations.items():
        if len(durs) == 0:
            continue
        durs = np.array(durs)
        properties[action] = {
            'median_duration': int(np.median(durs)),
            'min_duration': max(2, int(np.percentile(durs, 5))),
            'frequency': len(durs)
        }
    
    print(f"Computed properties for {len(properties)} actions.")
    return properties
