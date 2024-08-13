import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import roc_auc_score
import sys
import h5py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
src_path = os.path.abspath(os.path.join(current_dir, '../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from RunningDataset import RunningDataset

def preprocess(): 
    data = RunningDataset()
    train_full, X_test, y_test = data.preprocess()
     # Drop specified columns from train_full
    y_train = train_full[['injury']]
    X_train = train_full.drop(columns=['Athlete ID', 'injury', 'Date'])    
    
    X_combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

    expected_length = len(train_full) + len(X_test)
    if len(X_combined) != expected_length or len(y_combined) != expected_length:
        raise ValueError("Lengths of X_combined and y_combined do not match the expected combined length.")

    directory = '../models'
    models = []

    # Load all models from the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as file:
                model = pickle.load(file)
                models.append(model)
    
    all_predictions = []

    # Predict using each model and store the results
    for model in models:
        predictions = model.predict_proba(X_combined)[:, 1].astype('float32')
        all_predictions.append(predictions)

    # Stack predictions and calculate the mean
    all_predictions_array = np.stack(all_predictions, axis=0)
    mean_predictions = np.mean(all_predictions_array, axis=0)

    X_reshaped = stack(X_combined, 7)

    with h5py.File('./surrogate/data/dataset.h5', 'w') as h5file:
        h5file.create_dataset('X', data=X_reshaped)
        h5file.create_dataset('y', data=mean_predictions)

    auc = roc_auc_score(y_combined, mean_predictions)
    print(f'AUC: {auc:.4f}') 


def stack(df, days):
        """
        Converts data from 2D shape (no_samples, no_variables * no_time_steps), i.e., (N, 140) to 
        3D shape (no_samples, no_timesteps, no_variables), i.e., (N, 14, 10) as required by the LSTM model.
        """
        df.reset_index(drop=True, inplace=True)
        num_variables = 10 
        time_steps_per_variable = days
        num_samples = len(df)
    
        reshaped_data = np.zeros((num_samples, time_steps_per_variable, num_variables))
        for index, row in df.iterrows():
            for time_step in range(time_steps_per_variable):
                segment_start = time_step * num_variables
                segment_end = segment_start + num_variables
                reshaped_data[index, time_step, :] = row.iloc[segment_start:segment_end].values
                
        return reshaped_data

if __name__ == '__main__':
    preprocess()
