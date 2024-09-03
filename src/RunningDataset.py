import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
import tensorflow as tf
import torch
from imblearn.combine import SMOTETomek
from pyts.image import GramianAngularField
from scipy.spatial.distance import pdist, squareform
from numpy.random import default_rng

"""
RunningDataset class manages operations related to loading and processing and splitting of data for XGBoost model training
"""
class RunningDataset:
    """
    __init__ method initialises the dataset intance with predetermined parameters
    """
    def __init__(self):
        """Initialize the dataset handling with predetermined parameters."""
        self.filename = '../data/day_approach_maskedID_timeseries.csv'
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
    
    def split_data(self):
        """
        Splits the data into training and testing datasets based on the last 10 entries as test data.
        """
        athletes = pd.Series(self.data[self.identifiers[0]].unique())
        sorted_athletes = athletes.sort_values()
        test_ids = sorted_athletes[-10:].values
        train_ids = sorted_athletes[:-10].values
        train = self.data[self.data[self.identifiers[0]].isin(train_ids)]
        test = self.data[self.data[self.identifiers[0]].isin(test_ids)]
        print("Training Data - Injury counts:", train['injury'].value_counts())
        print("Testing Data - Injury counts:", test['injury'].value_counts())
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        return train, test
    
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
    function prepares the dataset for training by normalising and splitting in train and test set
    """
    def preprocess(self):
        self.data = self.normalise(self.data)
        self.train, self.test = self.split_data()
        self.X_test = self.test.drop(columns=self.fixed_columns)
        self.y_test = self.test[self.class_name]
        return self.train, self.X_test, self.y_test