"""
MABe Mouse Behavior Detection Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main modules for easy access
from . import config
from . import utils
from . import data_loader
from . import models
from . import scoring
from . import feature_engineering
from . import postprocessing

# Import commonly used functions
from .data_loader import load_train_test_data, generate_mouse_data
from .feature_engineering import transform_single, transform_pair
from .models import StratifiedSubsetClassifier, StratifiedSubsetClassifierWEval
from .utils import _make_lgbm, _make_xgb, _make_cb
from .scoring import optimize_ensemble_weights, optimize_thresholds_per_action
from .postprocessing import (
    predict_multiclass_with_confidence,
    remove_overlaps_by_confidence,
    robustify,
    compute_action_properties_from_df
)

__all__ = [
    # Modules
    'config',
    'utils',
    'data_loader',
    'models',
    'scoring',
    'feature_engineering',
    'postprocessing',
    
    # Functions - Data Loading
    'load_train_test_data',
    'generate_mouse_data',
    
    # Functions - Feature Engineering
    'transform_single',
    'transform_pair',
    
    # Classes - Models
    'StratifiedSubsetClassifier',
    'StratifiedSubsetClassifierWEval',
    
    # Functions - Model Factories
    '_make_lgbm',
    '_make_xgb',
    '_make_cb',
    
    # Functions - Scoring
    'optimize_ensemble_weights',
    'optimize_thresholds_per_action',
    
    # Functions - Post-processing
    'predict_multiclass_with_confidence',
    'remove_overlaps_by_confidence',
    'robustify',
    'compute_action_properties_from_df',
]
