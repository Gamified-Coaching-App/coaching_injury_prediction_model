import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from imblearn.combine import SMOTETomek
from pyts.image import GramianAngularField
from scipy.spatial.distance import pdist, squareform
from numpy.random import default_rng
import os
import pickle
from sklearn.metrics import roc_auc_score

"""
RunningDatasetSurrogate class manages operations related to loading and processing of data
"""
class RunningDatasetSurrogate:    
    """
    __init__ method initialises the dataset intance with predetermined parameters
    """
    def __init__(self):
        self.filename = '../../data/day_approach_maskedID_timeseries.csv'
        self.WINDOW_DAYS = 7
        self.base_metrics = ['nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'km sprinting', 
                             'strength training', 'hours alternative', 'perceived exertion', 
                             'perceived trainingSuccess', 'perceived recovery']
        self.identifiers = ['Athlete ID', 'Date']
        self.class_name = 'injury'
        self.fixed_columns = ['Athlete ID', 'injury', 'Date']
        self.data_types_metrics = [float] * len(self.base_metrics)
        self.data_types_fixed_columns = [int] * len(self.identifiers)
        self.data = pd.read_csv(self.filename)
        self.data.columns = [f"{col}.0" if i < 10 else col for i, col in enumerate(self.data.columns)]
        self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    """
    function computes mean and standard deviation for normalization purposes, considering only non-injured cases.
    """
    def getMeanStd(self, data):
        mean = data[data['injury'] == 0].groupby(self.identifiers[0]).mean()
        std = data[data['injury'] == 0].groupby(self.identifiers[0]).std()
        std.replace(to_replace=0.0, value=0.01, inplace=True)  
        return mean, std
    
    """
    function applies z-score normalisation for a given row using precomputed mean and standard deviation
    """
    def normalize_athlete(self, row, metric, mean_df, std_df):
        
        athlete_id = row[self.identifiers[0]]
        if athlete_id in mean_df.index and athlete_id in std_df.index:
            mu = mean_df.loc[athlete_id, metric]
            su = std_df.loc[athlete_id, metric]
            return (row[metric] - mu) / su
        raise IndexError(f"Athlete ID {athlete_id} not found in mean and standard deviation dataframes.")

    """
    function applies z-score normalisation grouped by athlete to each metric in the dataframe
    """
    def z_score_normalization(self, df):
        mean_df, std_df = self.getMeanStd(df)
        for metric in self.base_metrics:
            df[metric] = df.apply(lambda row: self.normalize_athlete(row, metric, mean_df, std_df), axis=1)
        return df
    
    """
    function applies min-max normalisation to each metric in the dataframe - column by column
    """
    def min_max_normalization(self, df):
        for metric in self.base_metrics:
            df[metric] = self.min_max_scaler.fit_transform(df[metric].values.reshape(-1, 1)).flatten()
        return df.reset_index(drop=True)
    
    """
    function normalises the dataset using both z-score normalisation athlete by athlete and then and Min-Max normalisation column by column
    """
    def normalise(self, dataset):
        long = self.long_form(dataset)
        long = self.z_score_normalization(long)
        long = self.min_max_normalization(long)
        return self.wide_form(long, 7)
    
    """
    function converts the dataset to long format required for normalising.
    """
    def long_form(self, df):
        df_long = pd.wide_to_long(df, stubnames=self.base_metrics, i=self.fixed_columns, j='Offset', sep='.')
        df_long.reset_index(inplace=True)
        df_long[self.identifiers[1]] = df_long[self.identifiers[1]] - (self.WINDOW_DAYS - df_long['Offset'])
        df_long.drop(columns='Offset', inplace=True)
        df_long.drop_duplicates(subset=self.identifiers, keep='first', inplace=True)
        return df_long
    
    """
    function converts the dataset from long format to wide format after normalisation.
    """
    def wide_form(self, df_long, days):
        df_long['Date'] = df_long['Date'].astype(int)
        df_long['Athlete ID'] = df_long['Athlete ID'].astype(int)
        df_long['injury'] = df_long['injury'].astype(int)
        df_long = df_long.groupby(self.identifiers[0], as_index=False).apply(self.fill_missing_dates).reset_index(drop=True)
        df_long.sort_values(by=self.identifiers, inplace=True)
        athlete_info = df_long[self.fixed_columns]
        df_rolled = pd.DataFrame(index=athlete_info.index).join(athlete_info)
        for day in range(days):
            shifted = df_long.groupby(self.identifiers[0])[self.base_metrics].shift(day).add_suffix(f'.{days - 1 - day}')
            df_rolled = df_rolled.join(shifted)
        metric_columns = [f'{metric}.{day}' for day in range(days) for metric in self.base_metrics]
        df_rolled = df_rolled[metric_columns + self.fixed_columns]
        df_rolled.dropna(inplace=True)
        df_rolled.reset_index(drop=True, inplace=True)
        df_rolled.sort_values(by=self.identifiers, inplace=True)
        df_rolled[self.identifiers[1]] = df_rolled[self.identifiers[1]] + 1
        df_rolled = df_rolled.sort_values(by=self.identifiers).reset_index(drop=True)
        df_rolled = df_rolled.astype(dict(zip(df_rolled.columns, self.data_types_metrics * days + self.data_types_fixed_columns)))
        return df_rolled
    
    """
    function fills in missing dates for each athlete to ensure continuity in the data set - needed if the athlete 
    has no recorded data for a specific date.
    """
    def fill_missing_dates(self, group):
        min_date = group[self.identifiers[1]].min()
        max_date = group[self.identifiers[1]].max()
        int_range = range(min_date, max_date + 1)
        group = group.set_index(self.identifiers[1]).reindex(int_range).rename_axis(self.identifiers[1]).reset_index()
        group[self.identifiers[0]] = group[self.identifiers[0]].ffill()
        return group
    
    """
    function to get predictions from the XGBoost models trained on the dataset as the ground truth for surrogate model training
    """
    def get_xgboost_predictions(self, data):
        directory = '../../models'
        models = []

        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'rb') as file:
                    model = pickle.load(file)
                    models.append(model)

        all_predictions = []
        
        y = data[self.class_name]
        X = data.drop(columns=self.fixed_columns)
        
        for model in models:
            predictions = model.predict_proba(X)[:, 1].astype('float32')
            all_predictions.append(predictions)

        all_predictions_array = np.stack(all_predictions, axis=0)
        mean_predictions = np.mean(all_predictions_array, axis=0)
        auc = roc_auc_score(y, mean_predictions)
        return mean_predictions
    
    """
    function converts data from 2D shape (no_samples, no_variables * no_time_steps), i.e., (N, 140) to
    3D shape (no_samples, no_timesteps, no_variables), i.e., (N, 14, 10)
    """
    def stack(self, df, days):
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

    """
    function prepares the dataset for training by normalising and stacking it to 3D shape
    """
    def preprocess(self):
        data = self.normalise(self.data)
        y = self.get_xgboost_predictions(data)
        X = data.drop(columns=self.fixed_columns)
        X = self.stack(X, days=7)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        return X, y