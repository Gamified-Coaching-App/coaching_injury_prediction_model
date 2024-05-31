from RunningDataset import RunningDataset
from autoencoder import run_and_encode
from classifier import run_and_evaluate
import torch
from torch.utils.data import DataLoader, TensorDataset
from preprocessor import Preprocessor
from classifier_lstm import run_and_evaluate
#from classifier import run_and_evaluate



def run():
    # dataset = RunningDataset()
    # X_train, y_train, X_validate, y_validate, X_test, y_test = dataset.preprocess()
    # X_train_encoded, X_validate_encoded, X_test_encoded, = run_and_encode(X_train=X_train, X_validate=X_validate, X_test=X_test)
    # run_and_evaluate(X_train_encoded, y_train, X_validate_encoded, y_validate, X_test_encoded, y_test)
    preprocessor = Preprocessor()
    X_train, y_train, X_test, y_test = preprocessor.preprocess()
    run_and_evaluate(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    run()