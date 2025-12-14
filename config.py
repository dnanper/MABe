"""
Configuration file for MABe Mouse Behavior Detection
"""
import os
import random
import numpy as np

# ==================== SEED CONFIGURATION ====================
SEED = 38

# Set environment variables
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ==================== GPU CONFIGURATION ====================
USE_GPU = (
    ("KAGGLE_KERNEL_RUN_TYPE" in os.environ) and 
    (__import__("shutil").which("nvidia-smi") is not None)
)

# ==================== DATA PATHS ====================
TRAIN_CSV = '/kaggle/input/MABe-mouse-behavior-detection/train.csv'
TEST_CSV = '/kaggle/input/MABe-mouse-behavior-detection/test.csv'
TRAIN_TRACKING_DIR = "/kaggle/input/MABe-mouse-behavior-detection/train_tracking"
TEST_TRACKING_DIR = "/kaggle/input/MABe-mouse-behavior-detection/test_tracking"
TRAIN_ANNOT_DIR = "/kaggle/input/MABe-mouse-behavior-detection/train_annotation"

# ==================== MODEL PATHS ====================
MODEL_SAVE_DIR = "/kaggle/working/saved_models"
MODEL_LOAD_DIR = "/kaggle/input/mabe-ver-5/pytorch/default/1/results_5/saved_models"

# ==================== BODY PARTS TO DROP ====================
DROP_BODY_PARTS = [
    'headpiece_bottombackleft', 'headpiece_bottombackright', 
    'headpiece_bottomfrontleft', 'headpiece_bottomfrontright',                  
    'headpiece_topbackleft', 'headpiece_topbackright', 
    'headpiece_topfrontleft', 'headpiece_topfrontright',                  
    'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint'
]

# ==================== TRAINING PARAMETERS ====================
MAX_SAMPLES = 2_000_000
NEG_POS_RATIO = 3.0  # Negative to Positive ratio for balancing
N_SAMPLES_DEFAULT = 1_500_000

# ==================== THRESHOLD OPTIMIZATION ====================
THRESHOLD_RANGE = (0.10, 0.60)
THRESHOLD_STEP = 0.01
DEFAULT_THRESHOLD = 0.27

# ==================== POST-PROCESSING ====================
MIN_DURATION = 3  # Minimum duration for action segments
CONFIDENCE_THRESHOLD = 0.05  # Minimum confidence for predictions

# ==================== VALIDATION ====================
VALID_SIZE = 0.2  # Validation split size
VAL_CAP_RATIO = 0.25

# ==================== MODEL CONFIGURATIONS ====================
LGBM_CONFIG_1 = {
    'n_estimators': 225,
    'learning_rate': 0.07,
    'min_child_samples': 40,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'verbose': -1
}

LGBM_CONFIG_2 = {
    'n_estimators': 150,
    'learning_rate': 0.1,
    'min_child_samples': 20,
    'num_leaves': 63,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1
}

XGB_CONFIG_1 = {
    'n_estimators': 180,
    'learning_rate': 0.08,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'verbosity': 1
}

CB_CONFIG_1 = {
    'iterations': 120,
    'learning_rate': 0.1,
    'depth': 6,
    'verbose': 50,
    'allow_writing_files': False
}

# ==================== DATA AUGMENTATION ====================
FIXED_AUG_ANGLE = 180  # Rotation angle for data augmentation

# ==================== VERBOSE ====================
VERBOSE = True

print(f'Configuration loaded. Using GPU: {USE_GPU}')
