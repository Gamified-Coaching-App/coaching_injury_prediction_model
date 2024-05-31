import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses, initializers
import numpy as np
import os

#tf.keras.backend.set_floatx('float64')

class Autoencoder(models.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=(8, 8, 10)),
            layers.Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='selu'),
            layers.BatchNormalization(),
            layers.AlphaDropout(0.3),
            layers.Conv2D(filters=16, kernel_size=2, strides=1, padding='same', activation='selu'),
            layers.BatchNormalization(),
            layers.AlphaDropout(0.3),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(units=64, activation='linear'),
            layers.BatchNormalization(),
            layers.AlphaDropout(0.3),
            layers.Dense(units=10, activation='linear')
        ])
        self.decoder = models.Sequential([
            layers.Dense(units=64, activation='linear'),
            layers.BatchNormalization(),
            layers.AlphaDropout(0.3),
            layers.Dense(units=256, activation='linear'),
            layers.BatchNormalization(),
            layers.AlphaDropout(0.3),
            layers.Reshape((4, 4, 16)),
            layers.UpSampling2D(),
            layers.Conv2D(filters=16, kernel_size=2, strides=1, padding='same', activation='selu'),
            layers.BatchNormalization(),
            layers.AlphaDropout(0.3),
            layers.Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='selu'),
            layers.BatchNormalization(),
            layers.AlphaDropout(0.3),
            layers.Conv2D(filters=10, kernel_size=2, strides=1, padding='same', activation='tanh')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(autoencoder, train_dataset, num_epochs=100, learning_rate=0.01):
    autoencoder.compile(optimizer=optimizers.Adadelta(learning_rate=learning_rate), loss='mse')
    history = autoencoder.fit(train_dataset, epochs=num_epochs, verbose=2)
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
#     X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
#     return X_train

def evaluate(autoencoder, dataset):
    return autoencoder.evaluate(dataset, dataset, verbose=0)

def encode_data(dataset, autoencoder):
    if len(dataset.shape) == 3:  # data shape should be [height, width, channels]
        dataset = tf.expand_dims(dataset, axis=0)
        print("Expanded dimensions before encoding")
    encoded_data = autoencoder.encoder.predict(dataset) 
    # Remove batch dimension
    if encoded_data.shape[0] == 1:
        print("Squeeze before encoding")
        encoded_data = tf.squeeze(encoded_data, axis=0)
    return encoded_data

def normalise(encoded_X_train, encoded_X_validate, encoded_X_test):
    concatenated_data = np.concatenate([encoded_X_train, encoded_X_validate, encoded_X_test], axis=0)
    min_vals = np.min(concatenated_data, axis=0)
    max_vals = np.max(concatenated_data, axis=0)
    range_vals = max_vals - min_vals
    normalized_data = (concatenated_data - min_vals) / range_vals
    train_len = encoded_X_train.shape[0]
    validate_len = encoded_X_validate.shape[0]
    test_len = encoded_X_test.shape[0]

    normalized_X_train = normalized_data[:train_len]
    normalized_X_validate = normalized_data[train_len:train_len + validate_len]
    normalized_X_test = normalized_data[train_len + validate_len:train_len + validate_len + test_len]
   
    return normalized_X_train, normalized_X_validate, normalized_X_test

def run_and_encode(X_train, X_validate, X_test):
    autoencoder = Autoencoder()
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train)).batch(512)    
    history = train_autoencoder(autoencoder, train_dataset)
    
    print(f'Train loss: {evaluate(autoencoder, X_train)}')
    print(f'Validation loss: {evaluate(autoencoder, X_validate)}')
    print(f'Test loss: {evaluate(autoencoder, X_test)}')
    
    encoded_X_train = encode_data(X_train, autoencoder)
    encoded_X_validate = encode_data(X_validate, autoencoder)
    encoded_X_test = encode_data(X_test, autoencoder)
    #encoded_X_train, encoded_X_validate, encoded_X_test = normalise(encoded_X_train, encoded_X_validate, encoded_X_test)
    return encoded_X_train, encoded_X_validate, encoded_X_test
    

