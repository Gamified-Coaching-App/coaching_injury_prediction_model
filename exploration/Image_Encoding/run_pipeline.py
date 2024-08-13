from classifier import run_and_evaluate
from autoencoder import run_and_encode
from RunningDataset import RunningDataset

def run():
    data = RunningDataset()
    X_train, y_train, X_validate, y_validate, X_test, y_test = data.preprocess()
    encoded_X_train, encoded_X_validate, encoded_X_test = run_and_encode(X_train, X_validate, X_test)
    run_and_evaluate(encoded_X_train, y_train, encoded_X_validate, y_validate, encoded_X_test, y_test)

if __name__ == "__main__":
    run()