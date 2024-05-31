import pandas as pd
import numpy as np
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


class RunningDataset:
    """
    Manages operations related to loading, processing, and splitting athlete performance data.
    """
    def __init__(self):
        """Initialize the dataset handling with predetermined parameters."""
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
    
    def split_data(self):
        """
        Splits the data into training and testing datasets based on the last 10 entries as test.
        Returns:
            train (DataFrame): Training dataset.
            test (DataFrame): Testing dataset.
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
    
    def getMeanStd(self, data):
        """
        Computes mean and standard deviation for normalization purposes, considering only non-injured cases.
        """
        mean = data[data['injury'] == 0].groupby(self.identifiers[0]).mean()
        std = data[data['injury'] == 0].groupby(self.identifiers[0]).std()
        std.replace(to_replace=0.0, value=0.01, inplace=True)  # Avoid division by zero
        return mean, std

    def normalize_athlete(self, row, metric, mean_df, std_df):
        """
        Applies z-score normalization for a given row using precomputed mean and standard deviation.
        """
        athlete_id = row[self.identifiers[0]]
        if athlete_id in mean_df.index and athlete_id in std_df.index:
            mu = mean_df.loc[athlete_id, metric]
            su = std_df.loc[athlete_id, metric]
            return (row[metric] - mu) / su
        raise IndexError(f"Athlete ID {athlete_id} not found in mean and standard deviation dataframes.")

    def z_score_normalization(self, df):
        """
        Applies z-score normalization grouped by athlete to each metric in the dataframe.
        """
        mean_df, std_df = self.getMeanStd(df)
        for metric in self.base_metrics:
            df[metric] = df.apply(lambda row: self.normalize_athlete(row, metric, mean_df, std_df), axis=1)
        return df
    
    def min_max_normalization(self, df):
        """
        Applies Min-Max normalization to each metric in the dataframe.
        """
        for metric in self.base_metrics:
            df[metric] = self.min_max_scaler.fit_transform(df[metric].values.reshape(-1, 1)).flatten()
        return df.reset_index(drop=True)
    
    def normalise(self, dataset, days=14):
        """
        Normalizes the dataset using both z-score normalisation athlete by athlete and then and Min-Max normalisation column by column.
        """
        long = self.long_form(dataset)
        long = self.z_score_normalization(long)
        long = self.min_max_normalization(long)
        return self.wide_form(long, days=days)
    
    def long_form(self, df):
        """
        Converts the dataset to long format required for normalising.
        """
        df_long = pd.wide_to_long(df, stubnames=self.base_metrics, i=self.fixed_columns, j='Offset', sep='.')
        df_long.reset_index(inplace=True)
        df_long[self.identifiers[1]] = df_long[self.identifiers[1]] - (self.WINDOW_DAYS - df_long['Offset'])
        df_long.drop(columns='Offset', inplace=True)
        df_long.drop_duplicates(subset=self.identifiers, keep='first', inplace=True)
        return df_long
    
    def wide_form(self, df_long, days):
        """
        Converts the dataset from long format to wide format after normalisation.
        """
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
        """
        Fills in missing dates for each athlete to ensure continuity in the data set.
        Needed if the athlete has not recorded data for a specific date.
        """
        min_date = group[self.identifiers[1]].min()
        max_date = group[self.identifiers[1]].max()
        int_range = range(min_date, max_date + 1)
        group = group.set_index(self.identifiers[1]).reindex(int_range).rename_axis(self.identifiers[1]).reset_index()
        group[self.identifiers[0]] = group[self.identifiers[0]].ffill()
        return group
    
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
        smote = SMOTE(sampling_strategy=sampling_rate, random_state=42, k_neighbors=300) 
        X_res, y_res = smote.fit_resample(X_cleaned, y_cleaned)

        print("Number of samples after SMOTE: ", X_res.shape[0])
        print("Number of injuries (1) after SMOTE:", (y_res == 1).sum())
        print("Number of non-injuries (0) after SMOTE:", (y_res == 0).sum())

        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)


    def stack(self, df, days):
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

    def preprocess(self, days=7):
        """
        Normalises, splits, resamples and stacks data for training and testing.
        """
        self.data = self.normalise(self.data, days=days)
        self.train, self.test = train_test_split(self.data)
        self.X_train, self.y_train = self.multi_resample(self.train)

        self.X_train = self.stack(self.X_train, days)
        self.X_test = self.stack(self.test.drop(columns=self.fixed_columns), days)
        self.y_test = self.test[self.class_name]

        X_train = tf.convert_to_tensor(self.X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(self.y_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(self.X_test, dtype=tf.float32)
        y_test = tf.convert_to_tensor(self.y_test, dtype=tf.float32)

        print("Shapes of the datasets: X_train:", X_train.shape, 
              "y_train:", y_train.shape, 
              "X_test:", X_test.shape, 
              "y_test:", y_test.shape)
        return X_train, y_train, X_test, y_test