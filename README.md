# Coaching Injury Prediction Model

## Overview

Source code for Injury Prediction Model 

## Directory Structure

```bash
src/
    ├── surrogate/
    │   ├── model/                     # Implementation of the surrogate model
    │   ├── report/                    # Reports and visualizations generated, incl. report.ipynb
    │   ├── RunningDatasetSurrogate.py # Dataset handling for the surrogate model
    │   ├── config.py                  # Configuration file for model hyperparameters and settings
    │   └── run_pipeline.py            # Main script to run the surrogate model pipeline
    ├── RunningDataset.py              # Main dataset handling script
    ├── model.py                       # Original XGBOOST injury prediction model
