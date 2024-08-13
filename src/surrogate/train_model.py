from model.model import Model
from config import model_configs
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

def configure_gpu(device_to_use):
    if device_to_use == 'CPU':
        tf.config.set_visible_devices([], 'GPU')
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPUs Available: {gpus}")
            except RuntimeError as e:
                print(e)
        else: 
            print("No GPUs available.")

def train_model():
    configure_gpu(device_to_use='CPU')
    
    # Load data
    with h5py.File('./data/dataset.h5', 'r') as hf:
        X = hf['X'][:]
        y = hf['y'][:]
    
    # Convert data to numpy arrays
    X = X.astype('float32')
    y = y.astype('float32')
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert data back to TensorFlow tensors
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    
    # Parameters
    batch_size = 50 #batch size 50 most successful so far
    epochs = 1500
    
    for model_config in model_configs:
        model = Model(model_config)

        # Train the model
        model.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

        # Evaluate the model on the test set
        test_loss = model.model.evaluate(X_test, y_test)
        print(f"Model: {model_config['layers']}, Test Loss: {test_loss:.4f}")

        # Save the model
        #model.model.export('../../surrogate_model/final')

# Execute the function
if __name__ == '__main__':
    train_model()