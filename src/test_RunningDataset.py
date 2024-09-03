import pandas as pd
import numpy as np
import random
from exploration.Image_Encoding.RunningDataset import RunningDataset

def test_formatting_long_and_wide():
    data = RunningDataset()
    long = data.long_form(data.data)
    wide = data.wide_form(long, 7)
    original = data.data.sort_values(by=data.identifiers).reset_index(drop=True)
    pd.testing.assert_frame_equal(wide, original, check_dtype=True)

def test_z_score_normalization():
    data = RunningDataset()
    long = data.long_form(data.data)
    long.sort_values(by=data.identifiers, inplace=True)
    long.reset_index(drop=True, inplace=True)
    long_normalized = data.z_score_normalization(long)  
    long_normalized.sort_values(by=data.identifiers, inplace=True)
    long_normalized.reset_index(drop=True, inplace=True)
    random_rows = random.sample(range(0, len(long)), 100)  
    for row in random_rows:
        athlete_id = long.loc[row, data.identifiers[0]]
        athlete_data = long[long[data.identifiers[0]] == athlete_id] 
        for metric in data.base_metrics:
            mean = athlete_data[metric].mean()
            std = athlete_data[metric].std()
            expected_z_score = (long.loc[row, metric] - mean) / std if std != 0 else 0 
            actual_z_score = long_normalized.loc[row, metric]
            
            assert abs(expected_z_score - actual_z_score) < 0.1, \
                f"Z-score normalization failed for row {row}, metric {metric}, athlete id {athlete_id}. " \
                f"Expected {expected_z_score}, got {actual_z_score}"

def test_min_max_normalization():
    data = RunningDataset()
    long = data.long_form(data.data)
    long.sort_values(by=data.identifiers, inplace=True)
    long.reset_index(drop=True, inplace=True)
    long_normalized = data.min_max_normalization(long)  
    long_normalized.sort_values(by=data.identifiers, inplace=True)
    long_normalized.reset_index(drop=True, inplace=True)
    random_rows = random.sample(range(0, len(long)), 10000)
    for row in random_rows:
        athlete_id = long.loc[row, data.identifiers[0]]
        athlete_data = long[long[data.identifiers[0]] == athlete_id]
        for metric in data.base_metrics:
            min_val = athlete_data[metric].min()
            max_val = athlete_data[metric].max()
            actual_min_max = (long.loc[row, metric] - min_val) / (max_val - min_val) if max_val != 0 else 0
            expected_min_max = long_normalized.loc[row, metric]
            long.reset_index(drop=True, inplace=True)
            assert abs(expected_min_max -  actual_min_max) < 0.001, \
                f"Min-max normalization failed for row {row}, metric {metric}, athlete id {athlete_id}. " \
                f"Expected {expected_min_max}, got {actual_min_max}"