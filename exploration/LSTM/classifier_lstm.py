import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, initializers, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, Bidirectional, BatchNormalization, Dropout
from sklearn.metrics import roc_auc_score
from exploration.Image_Encoding.RunningDataset import RunningDataset
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import GlorotUniform, Orthogonal

class Classifier(models.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = Sequential([
            Bidirectional(LSTM(50, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), input_shape=(7,10)), 
            BatchNormalization(),
            Dropout(0.2),
            Bidirectional(LSTM(50, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.2),
            Bidirectional(LSTM(50, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.2),
            Bidirectional(LSTM(50, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=False)),
            BatchNormalization(),
            Dropout(0.2),
            layers.Dense(units=10, activation='selu'), 
            BatchNormalization(),
            Dropout(0.2),
            layers.Dense(units=10, activation='selu'), 
            BatchNormalization(),
            Dropout(0.2),
            layers.Dense(units=1, activation='sigmoid')
        ])

    def call(self, x):
        return self.model(x)
    
def train_model(model, training_set, epochs=700, learning_rate=0.0001, batch_size=512):
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate, beta_1=0.3, beta_2=0.999999, epsilon=1e-07, amsgrad=True),
                 loss=losses.BinaryFocalCrossentropy(alpha=0.55, gamma=5.0), metrics=['accuracy'])
    #best: beta_1=0.5, beta_2=0.99999, epsilon=1e-07, amsgrad=True
    
    history = model.fit(training_set, epochs=epochs, verbose=2)
    return history

# def get_random_batch(X_train, y_train, batch_size):
#     X_train = np.array(X_train)
#     y_train = np.array(y_train)
#     indices_0 = np.where(y_train == 0)[0]
#     indices_1 = np.where(y_train == 1)[0]
#     indices_0 = np.random.choice(indices_0, batch_size // 2, replace=False)
#     indices_1 = np.random.choice(indices_1, batch_size // 2, replace=False)
#     indices = np.concatenate([indices_0, indices_1])
#     X_train = X_train[indices]
#     y_train = y_train[indices]
#     X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
#     y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
#     return X_train, y_train

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    auc = roc_auc_score(y_test, predictions)
    return test_loss, auc

def run_and_evaluate(X_train, y_train, X_test, y_test):
    classifier = Classifier()
    training_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(512)    
    history = train_model(classifier, training_set)
    test_loss, auc_score_test = evaluate_model(classifier, X_test, y_test)
    train_loss, auc_score_train= evaluate_model(classifier, X_train, y_train)
    print(f"Test Loss: {test_loss}, Test AUC: {auc_score_test}")
    print(f"Train Loss: {train_loss}, Train AUC: {auc_score_train}")
    classifier.model.export('../coaching_optimiser/final/injury')