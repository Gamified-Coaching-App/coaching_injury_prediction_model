import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, initializers
from sklearn.metrics import roc_auc_score
from exploration.Image_Encoding.RunningDataset import RunningDataset
import numpy as np

class Classifier(models.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc_layers = [
            layers.Dense(units=50, activation='selu'),
            layers.BatchNormalization(),
            layers.AlphaDropout(0.5),

            # layers.Dense(units=50, activation='selu'),
            # layers.BatchNormalization(),
            # layers.AlphaDropout(0.5),

            # layers.Dense(units=50, activation='selu'),
            # layers.BatchNormalization(),
            # layers.AlphaDropout(0.5),

            layers.Dense(units=50, activation='selu'),
            layers.BatchNormalization(),
            layers.AlphaDropout(0.5),
        ]
        self.output_layer = layers.Dense(units=1, activation='sigmoid')

    def call(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        return self.output_layer(x)
    
def train_model(model, train_dataset, epochs=100, learning_rate=0.01):
    model.compile(optimizer=optimizers.Adadelta(learning_rate=learning_rate),
                 loss=losses.BinaryFocalCrossentropy(alpha=0.55, gamma=5.0))
    history = model.fit(train_dataset, epochs=epochs, verbose=2)
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

def run_and_evaluate(X_train, y_train, X_validate, y_validate, X_test, y_test):
    classifier = Classifier()
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(512)    
    history = train_model(classifier, train_dataset)
    train_loss, auc_score_train= evaluate_model(classifier, X_train, y_train)
    validate_loss, auc_score_validate= evaluate_model(classifier, X_validate, y_validate)
    test_loss, auc_score_test = evaluate_model(classifier, X_test, y_test)
    
    print(f"Test Loss: {test_loss}, Test AUC: {auc_score_test}")
    print(f"Train Loss: {train_loss}, Train AUC: {auc_score_train}")