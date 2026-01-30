# src/__init__.py
# Water Potability Classification - Utility Package

from .utils import (
    load_artifacts,
    preprocess_input,
    predict_single,
    predict_batch,
    get_feature_names,
    get_feature_ranges,
    get_model_info,
    get_feature_importance,
    get_feature_statistics,
    load_dataset,
    get_dataset_stats
)

__version__ = "2.0.0"
__author__ = "Atep Solihin - 301230038 - IF 5A"
