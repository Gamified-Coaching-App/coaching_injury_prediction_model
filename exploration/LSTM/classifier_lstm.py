import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, initializers, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.layers import LSTM, Bidirectional, BatchNormalization, Dropout, AlphaDropout
from sklearn.metrics import roc_auc_score
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import GlorotUniform, Orthogonal

class Classifier(models.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = Sequential([
            layers.Input(shape=(7, 10)),
            layers.Bidirectional(LSTM(20, return_sequences=True)),
            Dropout(0.1),
            layers.Bidirectional(LSTM(20, return_sequences=False)),
            layers.Dense(units=50, activation='selu'), 
            BatchNormalization(),
            AlphaDropout(0.1),
            layers.Dense(units=50, activation='selu'), 
            BatchNormalization(),
            AlphaDropout(0.1),
            layers.Dense(units=50, activation='selu'), 
            BatchNormalization(),
            AlphaDropout(0.1),
            # layers.Dense(units=10, activation='selu'), 
            # BatchNormalization(),
            # Dropout(0.1),
            layers.Dense(units=1, activation='sigmoid')
        ])

    def call(self, x):
        return self.model(x)
    
def train_model(model, training_set, epochs=1000, learning_rate=0.1, batch_size=512):
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate, beta_1=0.8, beta_2=0.9999, epsilon=1e-07),
                 loss=losses.BinaryFocalCrossentropy(alpha=0.55, gamma=5.0), metrics=['accuracy'])
    #best: beta_1=0.5, beta_2=0.99999, epsilon=1e-07, amsgrad=True
    
    history = model.fit(training_set, epochs=epochs, verbose=2)
    return history

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
    classifier.model.export('model/injury_prediction')