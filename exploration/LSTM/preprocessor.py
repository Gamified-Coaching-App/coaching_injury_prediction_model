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


class Preprocessor:
    def __init__(self):
        self.filename = 'data/day_approach_maskedID_timeseries.csv'
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
        self.standard_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        print("Shape data before cleaning: ", self.data.shape)
        #self.data = self.data_clear(self.data)
        print("Shape data after cleaning: ", self.data.shape)
        self.data = self.normalise(self.data, method='athlete-history', min=0)
        self.split_data()
    
    def judge_sum(self, a, b, c, d, e, f, g):
        return a + b + c + d + e + f + g

    def data_clear(self, data):
        columns_original = data.columns
        data_original = data.values
        judge = self.judge_sum(data_original[:, 0], data_original[:, 10], data_original[:, 20], data_original[:, 30], data_original[:, 40], data_original[:, 50], data_original[:, 60])

        index = np.where(judge != 0)
        data_new = data_original[index]
        return pd.DataFrame(data_new, columns=columns_original)

    def reorder_columns(self, data):
        n = 70
        new_order = []
        for i in range(10):
            new_order.extend([(i + 10 * j) % 70 for j in range(7)])
        data = data.iloc[:, new_order]
        return data
    
    def split_data(self):
        athletes = pd.Series(self.data[self.identifiers[0]].unique())
        sorted_athletes = athletes.sort_values()
        test_ids = sorted_athletes[-10:].values
        train_ids = sorted_athletes[:-10].values
        self.train = self.data[self.data[self.identifiers[0]].isin(train_ids)]
        self.test = self.data[self.data[self.identifiers[0]].isin(test_ids)]
        print("Training Data - Injury counts:", self.train['injury'].value_counts())
        print("Testing Data - Injury counts:", self.test['injury'].value_counts())
        self.train.reset_index(drop=True, inplace=True)
        self.test.reset_index(drop=True, inplace=True)

    def long_form(self, df):
        df_long = pd.wide_to_long(df, stubnames=self.base_metrics, i=self.fixed_columns, j='Offset', sep='.')
        df_long.reset_index(inplace=True)
        df_long[self.identifiers[1]] = df_long[self.identifiers[1]] - (self.WINDOW_DAYS - df_long['Offset'])
        df_long.drop(columns='Offset', inplace=True)
        df_long.drop_duplicates(subset=self.identifiers, keep='first', inplace=True)
        return df_long
    
    def z_score_normalization(self, df):
        for metric in self.base_metrics:
            df[metric] = df.groupby([self.identifiers[0]])[metric].transform(
                lambda x: self.standard_scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
            )
        return df.reset_index(drop=True)
    
    def min_max_normalization(self, df):
        for metric in self.base_metrics:
            df[metric] = self.min_max_scaler.fit_transform(df[metric].values.reshape(-1, 1)).flatten()
        return df.reset_index(drop=True)
    
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
    
    def fill_missing_dates(self, group):
        min_date = group[self.identifiers[1]].min()
        max_date = group[self.identifiers[1]].max()
        int_range = range(min_date, max_date + 1)
        group = group.set_index(self.identifiers[1]).reindex(int_range).rename_axis(self.identifiers[1]).reset_index()
        group[self.identifiers[0]] = group[self.identifiers[0]].ffill()
        return group
    
    def normalise(self, dataset, method = 'sliding-window', min=0):
        if method == 'sliding-window':
            normalized_data = pd.DataFrame(index=dataset.index, columns=dataset.columns, data=0.0)
            for index, row in dataset.iterrows():
                for start in range(0, 70, 7):
                    scaler = MinMaxScaler(feature_range=(min, 1))
                    end = start + 7
                    block = row[start:end]
                    scaled_block = scaler.fit_transform(block.values.reshape(-1, 1)).flatten()
                    normalized_data.iloc[index, start:end] = scaled_block
            normalized_data.iloc[:, -3:] = dataset.iloc[:, -3:]
            return normalized_data
        
        elif method == 'athlete-history':
            long = self.long_form(dataset)
            long = self.z_score_normalization(long)
            long = self.min_max_normalization(long)
            return self.wide_form(long, 7)
        else:
            raise ValueError("Invalid normalization method")
    
    def multi_resample(self, dataset):
        # Step 1: Balanced Sampling
        dataset = self.balanced_sampling(dataset)
        # Step 2: Unbalanced Sampling
        dataset = self.unbalanced_sampling(dataset, 650, 0.136)
        # Step 3: Synthetic Sampling
        X_resampled, y_resampled = self.synthetic_sampling(dataset, 1)
        
        return X_resampled, y_resampled

    def balanced_sampling(self, data):
        groups = data.groupby('Athlete ID')
        balanced_data = []

        all_injured = data[data[self.class_name] == 1]

        for _, group in groups:
            injured = group[group[self.class_name] == 1]
            uninjured = group[group[self.class_name] == 0]

            n_samples = len(uninjured)
            if injured.empty:
                injured_samples = all_injured.sample(n_samples, replace=True, random_state = 42)
                #continue
            else:
                injured_samples = injured.sample(n_samples, replace=True, random_state = 42)
            uninjured_samples = uninjured.sample(n_samples, replace=False, random_state = 42)  
            balanced_data.append(injured_samples)
            balanced_data.append(uninjured_samples)

        balanced_df = pd.concat(balanced_data)

        print("Number of samples after balanced sampling: ", len(balanced_df))
        print("Number of injuries (1) after balanced sampling:", balanced_df[self.class_name].sum())
        print("Number of non-injuries (0) after balanced sampling:", len(balanced_df) - balanced_df[self.class_name].sum())

        return balanced_df

    def unbalanced_sampling(self, balanced_data, injury_count, sampling_ratio):
        injured = balanced_data[balanced_data['injury'] == 1]
        uninjured = balanced_data[balanced_data['injury'] == 0]
        num_uninjured = int(injury_count / sampling_ratio)

        injured_samples = injured.sample(injury_count, random_state = 42, replace=False) 
        uninjured_samples = uninjured.sample(num_uninjured, random_state = 42,  replace=False) 
        sampled_data = pd.concat([injured_samples, uninjured_samples])

        # Output the number of samples after unbalanced sampling
        print("Number of samples after unbalanced sampling: ", len(sampled_data))
        print("Number of injuries (1) after unbalanced sampling:", len(sampled_data[sampled_data['injury'] == 1]))
        print("Number of non-injuries (0) after unbalanced sampling:", len(sampled_data[sampled_data['injury'] == 0]))

        return sampled_data

    def synthetic_sampling(self, data, sampling_rate=1):
        X = data.drop(columns=self.fixed_columns)
        y = data[self.class_name]

        # First, apply Tomek Links to remove majority class samples in Tomek pairs
        tl = TomekLinks(sampling_strategy='majority')
        X_cleaned, y_cleaned = tl.fit_resample(X, y)

        print("Number of samples after Tomek Links: ", X_cleaned.shape[0])
        print("Number of injuries (1) after Tomek Links:", (y_cleaned == 1).sum())
        print("Number of non-injuries (0) after Tomek Links:", (y_cleaned == 0).sum())

        # Then, use SMOTE to oversample the minority class based on the new class distribution
        smote = SMOTE(sampling_strategy=sampling_rate, random_state=42, k_neighbors=600) 
        X_res, y_res = smote.fit_resample(X_cleaned, y_cleaned)

        print("Number of samples after SMOTE: ", X_res.shape[0])
        print("Number of injuries (1) after SMOTE:", (y_res == 1).sum())
        print("Number of non-injuries (0) after SMOTE:", (y_res == 0).sum())

        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)

    def stack(self, df):
        df = self.reorder_columns(df)
        num_variables = 10  # Total number of different variables (features)
        time_steps_per_variable = 7  # Number of time steps per variable
        num_samples = len(df)
        # Initialize reshaped_data to accommodate the transposed shape (7, 10)
        reshaped_data = np.zeros((num_samples, time_steps_per_variable, num_variables))
        
        for index, row in df.iterrows():
            temp_row = np.zeros((num_variables, time_steps_per_variable))
            for var_index in range(num_variables):
                start_col = var_index * time_steps_per_variable
                end_col = start_col + time_steps_per_variable
                temp_row[var_index, :] = row.iloc[start_col:end_col].values
            
            # Transpose temp_row to switch the order of variables and time steps
            temp_row = temp_row.T
            #temp_row = temp_row[::-1] # Reverse the order of time steps
            reshaped_data[index, :, :] = temp_row  # Correctly assign the transposed temp_row
        return reshaped_data

    def preprocess(self):
        normalisation_method = 'athlete-history'
        norm_min=0
        #self.train = self.normalise(self.train, method=normalisation_method, min=norm_min)
        #self.test = self.normalise(self.test, method=normalisation_method, min=norm_min)
        self.X_train, self.y_train = self.multi_resample(self.train)
        self.X_restructure_train = self.stack(self.X_train)
        self.X_restructure_test = self.stack(self.test.drop(columns=self.fixed_columns))
        self.y_test = self.test[self.class_name]

        X_gasf_train = tf.convert_to_tensor(self.X_restructure_train, dtype=tf.float32)
        X_gasf_test = tf.convert_to_tensor(self.X_restructure_test, dtype=tf.float32)
        y_train = tf.convert_to_tensor(self.y_train.values, dtype=tf.float32)
        y_test = tf.convert_to_tensor(self.y_test.values, dtype=tf.float32)

        print("Shapes of the datasets: X_train:", X_gasf_train.shape, 
              "y_train:", y_train.shape, 
              "X_test:", X_gasf_test.shape, 
              "y_test:", y_test.shape)
        return X_gasf_train, y_train, X_gasf_test, y_test