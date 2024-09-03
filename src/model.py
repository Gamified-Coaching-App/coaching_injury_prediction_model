"""
code is based on Lovdal, Sofie; den Hartigh, Ruud; Azzopardi, George, 2021, "Replication Data for: Injury Prediction In Competitive Runners With Machine Learning",
Link for original code: https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/UWU9PV
"""

import os
import pandas as pd
import xgboost as xgb
import numpy as np
import random
import csv
import sklearn.metrics as metrics
from os import path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, confusion_matrix
from RunningDataset import RunningDataset
import pickle

def getPerformanceMeasurements(y_test, y_prob, in_thresh):
    if in_thresh == -1:
        cm = confusion_matrix(y_test, y_prob >= 0.5)
    else:
        cm = confusion_matrix(y_test, y_prob >= in_thresh)

    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
    TN = cm[0][0]

    PR, RE, mcc = (0, 0, 0)
    if (TP + FP) != 0 and (TP + FN) != 0 and (TN + FP) != 0 and (TN + FN) != 0:
        PR = TP / (TP + FP)
        mcc = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    RE = TP / (TP + FN) if (TP + FN) != 0 else 0
    SP = TN / (TN + FP) if (TN + FP) != 0 else 0
    acc = (TP + TN) / (TP + FP + FN + TN)
    F1 = getFScore(1, PR, RE)

    return PR, RE, SP, F1, acc, mcc, cm

def getFScore(beta, PR, RE):
    return (1 + (beta ** 2)) * (PR * RE) / ((beta ** 2 * PR) + RE) if PR + RE != 0 else 0

def getStats(y_test, y_pred, y_prob, in_thresh):
    auc = roc_auc_score(y_test, y_prob)
    PR, RE, SP, F1, acc, mcc, cm = getPerformanceMeasurements(y_test, y_prob, in_thresh)
    print(f'Performance Metrics: Precision={PR:.2f}, Recall={RE:.2f}, F1 Score={F1:.2f}, MCC={mcc:.2f}, Accuracy={acc:.2f}')
    return {"auc": auc, "PR": PR, "RE": RE, "SP": SP, "F1": F1, "acc": acc, "mcc": mcc, "cm": cm}

def getBalancedSubset(train, samplesPerClass):
    healthySet = pd.DataFrame()  
    unhealthySet = pd.DataFrame()  

    stats = pd.DataFrame(train[['Athlete ID', 'injury']].groupby(['Athlete ID', 'injury']).size().reset_index(name='counts'))
    stats = pd.DataFrame(stats[['Athlete ID']].groupby(['Athlete ID']).size().reset_index(name='counts'))
    stats.drop(stats[stats['counts'] < 2].index, inplace=True)
    athleteList = stats['Athlete ID'].unique()

    samplesPerAthlete = int(np.floor(samplesPerClass / len(athleteList)))

    for athlete in athleteList:
        athlete_data_unhealthy = train[(train['Athlete ID'] == athlete) & (train['injury'] == 0)].sample(samplesPerAthlete, replace=True)
        unhealthySet = pd.concat([unhealthySet, athlete_data_unhealthy], ignore_index=True)

        athlete_data_healthy = train[(train['Athlete ID'] == athlete) & (train['injury'] == 1)].sample(samplesPerAthlete, replace=True)
        healthySet = pd.concat([healthySet, athlete_data_healthy], ignore_index=True)

    balancedSet = pd.concat([healthySet, unhealthySet], ignore_index=True)
    X = balancedSet.drop(columns=['Athlete ID', 'injury', 'Date'])
    y = balancedSet['injury']
    return X, y

def trainModel(params, X_train, y_train, X_val, y_val, bag):
    model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.01,
                              max_depth=random.choice(params["XGBDepthList"]),
                              n_estimators=random.choice(params["XGBEstimatorsList"]),
                              importance_type='total_gain', eval_metric='auc', verbosity=1)
    model.fit(X_train, y_train)
    calib_model = CalibratedClassifierCV(model, method=params["calibrationType"], cv="prefit")
    calib_model.fit(X_val, y_val)

    model_dir = '../models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'calib_model_bag_{bag}.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(calib_model, file)
    return model, calib_model

def applyBagging(modelList, X_data, y_data, params):
    y_probList = pd.DataFrame()
    for idx, model in enumerate(modelList):
        y_prob = model.predict_proba(X_data)[:, 1]
        y_probList[f"Bag {idx}"] = y_prob

    y_prob_avg = y_probList.mean(axis=1)
    stats = getStats(y_data, y_data, y_prob_avg, 0.5)
    return stats

def runExperiment(train_full, X_test, y_test, params, exp):
    
    X_val, y_val = getBalancedSubset(train_full, params["samplesPerClass"])

    modelList = []
    featureRanking = pd.DataFrame()

    for bag in range(params["nbags"]):
        X_train, y_train = getBalancedSubset(train_full, params["samplesPerClass"])
        model, calib_model = trainModel(params, X_train, y_train, X_val, y_val, bag)
        modelList.append(calib_model)
        y_val_prob = calib_model.predict_proba(X_val)[:, 1]
        featureRanking[f"Bag {bag}"] = model.feature_importances_

    val_stats = applyBagging(modelList, X_val, y_val, params)
    test_stats = applyBagging(modelList, X_test, y_test, params)

    print(f"Validation AUC: {val_stats['auc']:.2f}, Test AUC: {test_stats['auc']:.2f}")
    return featureRanking

def main():
    params = {
        "nTestAthletes": 10,
        "nbags": 9,
        "calibrationType": "sigmoid",
        "nExp": 5,
        "samplesPerClass": 2048,
        "approachList": ["day"],
        "XGBEstimatorsList": [256, 512],
        "XGBDepthList": [2, 3]
    }

    data = RunningDataset()
    train_full, X_test, y_test = data.preprocess()

    for approach in params["approachList"]:
        for exp in range(params["nExp"]):
            print(f"Experiment {exp + 1}/{params['nExp']} for approach '{approach}'")
            featureRanking = runExperiment(train_full, X_test, y_test, params, exp)
            if exp == 0:
                aggFeatureRanking = featureRanking
            else:
                aggFeatureRanking += featureRanking

if __name__ == "__main__":
    main()