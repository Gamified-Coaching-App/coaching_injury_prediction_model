from classifier_lstm import run_and_evaluate
from RunningDataset import RunningDataset

def run():
    data = RunningDataset()
    X_train, y_train, X_test, y_test = data.preprocess()
    run_and_evaluate(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    run()