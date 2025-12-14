"""
Feature engineering utilities - Part 1: Helper functions
"""
import numpy as np
import pandas as pd
from utils import _scale, _scale_signed, _speed, _roll_future_mean, _roll_future_var


def safe_rolling(series, window, func, min_periods=None):
    """Safe rolling operation with NaN handling"""
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)


# ==================== CURVATURE FEATURES ====================
def add_curvature_features(X, center_x, center_y, fps):
    """Trajectory curvature (window lengths scaled by fps)"""
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)

    for w in [30, 60]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 6)).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    w = 30
    ws = _scale(w, fps)
    X[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 6)).sum()

    return X


# ==================== MULTISCALE FEATURES ====================
def add_multiscale_features(X, center_x, center_y, fps):
    """Multi-scale temporal features (speed in cm/s)"""
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    scales = [10, 40, 160]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()

    if len(scales) >= 2 and f'sp_m{scales[0]}' in X.columns and f'sp_m{scales[-1]}' in X.columns:
        X['sp_ratio'] = X[f'sp_m{scales[0]}'] / (X[f'sp_m{scales[-1]}'] + 1e-6)

    return X


# ==================== STATE FEATURES ====================
def add_state_features(X, center_x, center_y, fps):
    """Behavioral state transitions"""
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    try:
        bins = [-np.inf, 0.5 * fps, 2.0 * fps, 5.0 * fps, np.inf]
        speed_states = pd.cut(speed_ma, bins=bins, labels=[0, 1, 2, 3]).astype(float)

        for window in [60, 120]:
            ws = _scale(window, fps)
            if len(speed_states) >= ws:
                for state in [0, 1, 2, 3]:
                    X[f's{state}_{window}'] = (
                        (speed_states == state).astype(float)
                        .rolling(ws, min_periods=max(1, ws // 6)).mean()
                    )
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                X[f'trans_{window}'] = state_changes.rolling(ws, min_periods=max(1, ws // 6)).sum()
    except Exception:
        pass

    return X


# ==================== LONG RANGE FEATURES ====================
def add_longrange_features(X, center_x, center_y, fps):
    """Long-range temporal features"""
    for window in [120, 240]:
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            X[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            X[f'y_ml{window}'] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    for span in [60, 120]:
        s = _scale(span, fps)
        X[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()
        X[f'y_e{span}'] = center_y.ewm(span=s, min_periods=1).mean()

    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
    for window in [60, 120]:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            X[f'sp_pct{window}'] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)

    return X


# ==================== CUMULATIVE DISTANCE ====================
def add_cumulative_distance_single(X, cx, cy, fps, horizon_frames_base: int = 180, 
                                  colname: str = "path_cum180"):
    """Cumulative path distance over time window"""
    L = max(1, _scale(horizon_frames_base, fps))
    step = np.hypot(cx.diff(), cy.diff())
    path = step.rolling(2*L + 1, min_periods=max(5, L//6), center=True).sum()
    X[colname] = path.fillna(0.0).astype(np.float32)
    return X


# ==================== GROOMING MICROFEATURES ====================
def add_groom_microfeatures(X, df, fps):
    """Grooming-specific microfeatures"""
    parts = df.columns.get_level_values(0)
    if 'body_center' not in parts or 'nose' not in parts:
        return X

    cx = df['body_center']['x']
    cy = df['body_center']['y']
    nx = df['nose']['x']
    ny = df['nose']['y']

    cs = (np.sqrt(cx.diff()**2 + cy.diff()**2) * float(fps)).fillna(0)
    ns = (np.sqrt(nx.diff()**2 + ny.diff()**2) * float(fps)).fillna(0)

    w30 = _scale(30, fps)
    X['head_body_decouple'] = (ns / (cs + 1e-3)).clip(0, 10).rolling(
        w30, min_periods=max(1, w30//3)).median()

    r = np.sqrt((nx - cx)**2 + (ny - cy)**2)
    X['nose_rad_std'] = r.rolling(w30, min_periods=max(1, w30//3)).std().fillna(0)

    if 'tail_base' in parts:
        ang = np.arctan2(df['nose']['y']-df['tail_base']['y'], 
                        df['nose']['x']-df['tail_base']['x'])
        dang = np.abs(ang.diff()).fillna(0)
        X['head_orient_jitter'] = dang.rolling(w30, min_periods=max(1, w30//3)).mean()

    return X


# ==================== SPEED ASYMMETRY ====================
def add_speed_asymmetry_future_past_single(X, cx, cy, fps, horizon_base: int = 30, 
                                          agg: str = "mean"):
    """Past vs Future speed asymmetry (acausal, continuous)"""
    w = max(3, _scale(horizon_base, fps))
    v = _speed(cx, cy, fps)
    if agg == "median":
        v_past = v.rolling(w, min_periods=max(3, w//4), center=False).median()
        v_fut = v.iloc[::-1].rolling(w, min_periods=max(3, w//4)).median().iloc[::-1]
    else:
        v_past = v.rolling(w, min_periods=max(3, w//4), center=False).mean()
        v_fut = _roll_future_mean(v, w, min_p=max(3, w//4))
    X["spd_asym_1s"] = (v_fut - v_past).fillna(0.0)
    return X


# ==================== GAUSSIAN SHIFT ====================
def add_gauss_shift_speed_future_past_single(X, cx, cy, fps, window_base: int = 30, 
                                            eps: float = 1e-6):
    """Distribution shift (future vs past) via symmetric KL"""
    w = max(5, _scale(window_base, fps))
    v = _speed(cx, cy, fps)

    mu_p = v.rolling(w, min_periods=max(3, w//4)).mean()
    va_p = v.rolling(w, min_periods=max(3, w//4)).var().clip(lower=eps)

    mu_f = _roll_future_mean(v, w, min_p=max(3, w//4))
    va_f = _roll_future_var(v, w, min_p=max(3, w//4)).clip(lower=eps)

    kl_pf = 0.5 * ((va_p/va_f) + ((mu_f - mu_p)**2)/va_f - 1.0 + np.log(va_f/va_p))
    kl_fp = 0.5 * ((va_f/va_p) + ((mu_p - mu_f)**2)/va_p - 1.0 + np.log(va_p/va_f))
    X["spd_symkl_1s"] = (kl_pf + kl_fp).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


# ==================== SPECTRAL FEATURES ====================
def add_spectral_features(X, series, fps, col_prefix, window_sec=1.0):
    """Frequency domain features for detecting tremor/scratching"""
    w = int(round(window_sec * fps))
    if w < 4:
        w = 4
    
    # Zero Crossing Rate
    centered = series - series.rolling(w, min_periods=1, center=True).mean()
    zcr = ((centered * centered.shift(1)) < 0).astype(float)
    X[f'{col_prefix}_zcr'] = zcr.rolling(w, min_periods=1).mean()

    # High Frequency Energy
    low_freq = series.rolling(window=3, center=True).mean()
    high_freq_content = series - low_freq
    X[f'{col_prefix}_hfreq'] = (high_freq_content ** 2).rolling(w, min_periods=1).mean()

    return X


# ==================== VELOCITY & ACCELERATION ====================
def add_velocity_acceleration_features(X, cx, cy, fps):
    """Kinematic features: velocity, acceleration, jerk"""
    vx = cx.diff().fillna(0) * fps
    vy = cy.diff().fillna(0) * fps
    ax = vx.diff().fillna(0) * fps
    ay = vy.diff().fillna(0) * fps
    accel = np.sqrt(ax**2 + ay**2)
    
    jerk = accel.diff().fillna(0) * fps
    
    for w in [5, 15, 30]:
        ws = _scale(w, fps)
        X[f'accel_m{w}'] = accel.rolling(ws, min_periods=1).mean()
        X[f'jerk_m{w}'] = jerk.rolling(ws, min_periods=1).mean()
        X[f'jerk_max{w}'] = jerk.abs().rolling(ws, min_periods=1).max()
    return X


# ==================== ARENA POSITION ====================
def add_arena_position_features(X, cx, cy, arena_dims):
    """Arena position and distance to walls/center"""
    w, h = arena_dims
    if pd.isna(w) or pd.isna(h) or w == 0 or h == 0:
        return X
        
    X['arena_x_norm'] = cx / w
    X['arena_y_norm'] = cy / h
    
    center_x, center_y = w / 2, h / 2
    X['dist_to_center'] = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
    
    d_left = cx
    d_right = w - cx
    d_top = cy
    d_bottom = h - cy
    X['dist_to_wall'] = np.minimum.reduce([d_left, d_right, d_top, d_bottom])
    
    return X


# ==================== BODY PART RATIOS ====================
def add_body_part_ratios(X, single_mouse):
    """Body proportions (scale-invariant)"""
    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['body_ratio'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)
    
    if 'nose+body_center' in X.columns and 'tail_base+body_center' in X.columns:
        X['head_tail_ratio'] = X['nose+body_center'] / (X['tail_base+body_center'] + 1e-6)
    return X


# ==================== SHAPE FEATURES ====================
def add_shape_features(X, single_mouse):
    """Head triangle area using Shoelace formula"""
    x1, y1 = single_mouse['nose']['x'], single_mouse['nose']['y']
    x2, y2 = single_mouse['ear_left']['x'], single_mouse['ear_left']['y']
    x3, y3 = single_mouse['ear_right']['x'], single_mouse['ear_right']['y']
    
    X['head_area'] = 0.5 * np.abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    return X
