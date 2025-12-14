"""
Utility functions for MABe project
"""
import numpy as np
import lightgbm
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from config import SEED, USE_GPU


# ==================== MODEL FACTORIES ====================
def _make_lgbm(**kw):
    """Create LightGBM classifier with default settings"""
    kw.setdefault("random_state", SEED)
    kw.setdefault("feature_fraction_seed", SEED)
    kw.setdefault("data_random_seed", SEED)
    kw.setdefault("device", 'gpu' if USE_GPU else 'cpu')
    if USE_GPU:
        kw.setdefault("gpu_use_dp", True)
    return lightgbm.LGBMClassifier(**kw)


def _make_xgb(**kw):
    """Create XGBoost classifier with default settings"""
    kw.setdefault("random_state", SEED)
    kw.setdefault("tree_method", "gpu_hist" if USE_GPU else "hist")
    if USE_GPU:
        kw.setdefault("single_precision_histogram", True)
    return XGBClassifier(**kw)


def _make_cb(**kw):
    """Create CatBoost classifier with default settings"""
    kw.setdefault("random_seed", SEED)
    if USE_GPU:
        kw.setdefault("task_type", "GPU")
        kw.setdefault("devices", "0")
    else:
        kw.setdefault("task_type", "CPU")
    return CatBoostClassifier(**kw)


# ==================== FPS UTILITIES ====================
def _scale(n_frames_at_30fps, fps, ref=30.0):
    """Scale a frame count defined at 30 fps to the current video's fps."""
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))


def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    """Signed version of _scale for forward/backward shifts."""
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag


def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    """Extract FPS from metadata with fallback options"""
    if 'frames_per_second' in meta_df.columns and meta_df['frames_per_second'].notna().any():
        return float(meta_df['frames_per_second'].iloc[0])
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))


def _speed(cx, cy, fps):
    """Calculate speed in cm/s from coordinates"""
    return np.hypot(cx.diff(), cy.diff()).fillna(0.0) * float(fps)


# ==================== ROLLING FUNCTIONS ====================
def _roll_future_mean(s, w, min_p=1):
    """Mean over future window [t, t+w-1]"""
    return s.iloc[::-1].rolling(w, min_periods=min_p).mean().iloc[::-1]


def _roll_future_var(s, w, min_p=2):
    """Variance over future window [t, t+w-1]"""
    return s.iloc[::-1].rolling(w, min_periods=min_p).var().iloc[::-1]


def safe_rolling(series, window, func, min_periods=None):
    """Safe rolling operation with NaN handling"""
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)


# ==================== DATA AUGMENTATION ====================
def rotate_xy_dataframe(df_in, angle_degrees):
    """
    Rotate all coordinates in DataFrame (MultiIndex columns) around origin (0,0).
    df_in structure: columns=[(mouse_id, bodypart, 'x'), (..., 'y')]
    """
    df_rot = df_in.copy()
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    
    cols = df_rot.columns
    coord_level = -1
    
    if 'x' not in cols.get_level_values(-1) or 'y' not in cols.get_level_values(-1):
        return df_in
    
    unique_paths = set(col[:coord_level] for col in cols)
    
    for path in unique_paths:
        try:
            col_x = path + ('x',)
            col_y = path + ('y',)
            
            if col_x in df_rot.columns and col_y in df_rot.columns:
                ox = df_rot[col_x]
                oy = df_rot[col_y]
                
                df_rot[col_x] = ox * c - oy * s
                df_rot[col_y] = ox * s + oy * c
        except:
            continue
            
    return df_rot


# ==================== MODEL HELPERS ====================
def _find_lgbm_step(pipe):
    """Find LightGBM step in pipeline"""
    try:
        if "stratifiedsubsetclassifier__estimator" in pipe.get_params():
            est = pipe.get_params()["stratifiedsubsetclassifier__estimator"]
            if isinstance(est, lightgbm.LGBMClassifier):
                return "stratifiedsubsetclassifier"
        if "stratifiedsubsetclassifierweval__estimator" in pipe.get_params():
            est = pipe.get_params()["stratifiedsubsetclassifierweval__estimator"]
            if isinstance(est, lightgbm.LGBMClassifier):
                return "stratifiedsubsetclassifierweval"
    except Exception as e:
        print(e)
    return None


def _to_num(x):
    """Extract numeric suffix from string (e.g., 'mouse1' -> 1)"""
    if isinstance(x, (int, np.integer)):
        return int(x)
    import re
    m = re.search(r'(\d+)$', str(x))
    return int(m.group(1)) if m else None
