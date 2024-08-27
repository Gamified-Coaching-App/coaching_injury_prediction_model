import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, initializers, Sequential, Layer
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from keras_nlp.layers import TransformerEncoder, TransformerDecoder
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras_nlp.layers import TransformerEncoder, TransformerDecoder
import json

NUM_EPOCHS = 1000

class PositionalEncoding(Layer):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.get_positional_encoding(max_len, d_model)

    def get_positional_encoding(self, max_len, d_model):
        positions = tf.range(start=0, limit=max_len, delta=1, dtype=tf.float32)
        angle_rates = 1 / tf.pow(10000, (2 * (tf.cast(tf.range(d_model), tf.float32) // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = positions[:, tf.newaxis] * angle_rates[tf.newaxis, :]

        # Apply the sine to even indices in the array; 2i
        angle_rads = tf.concat([tf.sin(angle_rads[:, 0::2]), tf.cos(angle_rads[:, 1::2])], axis=-1)

        pos_encoding = angle_rads[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        # Add the positional encoding to the inputs
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]

class Model(tf.Module):
    """
    This class builds a neural network model based on a given JSON configuration.
    """
    def __init__(self, architecture, optimiser_params):
        """
        Initializes the model based on the given configuration.

        Parameters:
        - config: Configuration dictionary for the model, containing layer definitions and optimizer settings.
        """
        super(Model, self).__init__()
        self.model = self.create_model(architecture)
    
        if optimiser_params['scheduler'] == 'cosine_decay':
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=optimiser_params['scheduler_params']['min_lr'],
                decay_steps=optimiser_params['scheduler_params']['decay_steps'],
                warmup_target=optimiser_params['scheduler_params']['max_lr'],
                warmup_steps=optimiser_params['scheduler_params']['warmup_steps'],
                alpha=optimiser_params['scheduler_params']['min_lr'],
            )
        else:
            lr_schedule = optimiser_params['learning_rate']

        if optimiser_params['optimiser'] == 'adam':
            self.optimizer = optimizers.Adam(learning_rate=lr_schedule, weight_decay=optimiser_params['weight_decay'])
        elif optimiser_params['optimiser'] == 'adamw':
            self.optimizer = optimizers.AdamW(learning_rate=lr_schedule, weight_decay=optimiser_params['weight_decay'])
        else:
            self.optimizer = optimizers.Adadelta(learning_rate=lr_schedule, weight_decay=optimiser_params['weight_decay'])

        self.model.compile(optimizer=self.optimizer, loss='mse')


    def create_model(self, config):
        """
        Creates the neural network model based on the given configuration.

        Args:
        config (dict): Configuration dictionary for the model.

        Returns:
        model (tf.keras.Sequential): Compiled neural network model.
        """
        model = models.Sequential()
        model.add(layers.Input(shape=(7, 10), name='original_input'))

        for layer_config in config:
            layer_type = layer_config['type']

            if layer_type == 'lstm':
                params = {
                    'units': layer_config['units'],
                    'return_sequences': layer_config['return_sequences'],
                }
                if 'dropout' in layer_config:
                    params['dropout'] = layer_config['dropout']
                if 'recurrent_dropout' in layer_config:
                    params['recurrent_dropout'] = layer_config['recurrent_dropout']
                if 'activation' in layer_config:
                    params['activation'] = layer_config['activation']
                model.add(layers.LSTM(**params))

            elif layer_type == 'bidirectional_lstm':
                params = {
                    'units': layer_config['units'],
                    'return_sequences': layer_config['return_sequences'],
                }
                if 'dropout' in layer_config:
                    params['dropout'] = layer_config['dropout']
                if 'recurrent_dropout' in layer_config:
                    params['recurrent_dropout'] = layer_config['recurrent_dropout']
                if 'activation' in layer_config:
                    params['activation'] = layer_config['activation']
                model.add(layers.Bidirectional(layers.LSTM(**params)))

            elif layer_type == 'dense':
                params = {
                    'units': layer_config['units'],
                }
                if 'activation' in layer_config:
                    params['activation'] = layer_config['activation']
                if 'kernel_initializer' in layer_config:
                    params['kernel_initializer'] = layer_config['kernel_initializer']
                if 'bias_initializer' in layer_config:
                    params['bias_initializer'] = layer_config['bias_initializer']
                
                model.add(layers.Dense(**params))

            elif layer_type == 'conv2d':
                model.add(layers.Conv2D(
                    filters=layer_config['filters'],
                    kernel_size=layer_config['kernel_size'],
                    activation=layer_config['activation'],
                    padding=layer_config.get('padding', 'valid')
                ))

            elif layer_type == 'max_pooling2d':
                model.add(layers.MaxPooling2D(pool_size=layer_config['pool_size']))

            elif layer_type == 'batch_normalization':
                model.add(layers.BatchNormalization())
            
            elif layer_type == 'conv1d':
                model.add(layers.Conv1D(
                    filters=layer_config['filters'],
                    kernel_size=layer_config['kernel_size'],
                    activation=layer_config['activation']
                ))

            elif layer_type == 'max_pooling1d':
                model.add(layers.MaxPooling1D(pool_size=layer_config['pool_size']))

            elif layer_type == 'dropout':
                model.add(layers.Dropout(rate=layer_config['rate']))

            elif layer_type == 'reshape':
                model.add(layers.Reshape(target_shape=layer_config['target_shape']))
            
            elif layer_type == 'alphadropout':
                model.add(layers.AlphaDropout(rate=layer_config['rate']))

            elif layer_type == 'activation':
                model.add(layers.Activation(layer_config['activation']))

            elif layer_type == 'flatten':
                model.add(layers.Flatten())
            
            elif layer_type == 'positional_encoding':
                model.add(PositionalEncoding(max_len=7, d_model=10))
            
            elif layer_type == 'transformer_encoder':
                params = {
                    'num_heads': layer_config['num_heads'],
                    'intermediate_dim': layer_config['intermediate_dim'],
                    'dropout': layer_config.get('dropout'),
                    'activation': layer_config.get('activation'), 
                    'normalize_first': layer_config.get('normalize_first')
                }
                model.add(TransformerEncoder(**params)) 

            elif layer_type == 'transformer_decoder':
                params = {
                    'num_heads': layer_config['num_heads'],
                    'intermediate_dim': layer_config['intermediate_dim'],
                    'dropout': layer_config.get('dropout'),
                    'activation': layer_config.get('activation'), 
                    'normalize_first': layer_config.get('normalize_first')
                }
                model.add(TransformerDecoder(**params)) 
        
            elif layer_type == 'global_average_pooling1d':
                model.add(GlobalAveragePooling1D())
            
            else:   
                raise ValueError(f'Invalid layer type: {layer_type}')

        return model
    

def train(X_train, Y_train, X_val, Y_val, architecture, optimiser_params, mode):
    model = Model(architecture=architecture, optimiser_params=optimiser_params)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(optimiser_params['batch_size'])
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(optimiser_params['batch_size'])
    num_epochs = NUM_EPOCHS

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=300,  # Stop training after 100 epochs with no improvement
        restore_best_weights=False,
        verbose=0,
        min_delta=0.0000001
    )
    
    if mode == 'final_training':
        checkpoint_filepath = 'best_weights.weights.h5'
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=0
        )
        history = model.model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=val_dataset,
            verbose=0,
            callbacks=[checkpoint_callback, early_stopping]
        )
    else: 
        history = model.model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=val_dataset,
            verbose=0,
            callbacks=[early_stopping]
        )
    
    min_val_loss = min(history.history['val_loss'])
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    
    num_epochs = len(train_losses)

    return model, min_val_loss, train_losses, val_losses, num_epochs

def evaluate(model, X_test, Y_test):
    test_loss = model.model.evaluate(X_test, Y_test, verbose=0)
    return test_loss

def cross_validate(Y, X, architecture, optimiser_params, report, n_splits=5):

    kf = KFold(n_splits=n_splits, shuffle=True)
    val_losses = []
   
    for train_index, val_index in kf.split(Y):
        #X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
        Y_train= Y[train_index]
        Y_val = Y[val_index]
        X_train= X[train_index]
        X_val= X[val_index]

        
        #Train the model and get the minimum validation loss for this fold
        _, min_val_loss, _, _, num_epochs = train(
            X_train=tf.convert_to_tensor(X_train), 
            Y_train=tf.convert_to_tensor(Y_train), 
            X_val=tf.convert_to_tensor(X_val), 
            Y_val=tf.convert_to_tensor(Y_val), 
            architecture=architecture,
            optimiser_params=optimiser_params,
            mode='cross_validation'
        )
        print('finished training at epoch:', num_epochs, 'with min_val_loss:', min_val_loss)
        
        val_losses.append(min_val_loss)
        
    # Compute the average validation loss across all folds
    average_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)

    key = f'architecture={str(architecture)}, optimiser_params={str(optimiser_params)}'

    report[key] = {
        'val_losses': val_losses,
        'mean_val_loss': round(average_val_loss, 6),
        'stdv_val_loss': round(std_val_loss, 6),
        'num_epochs': num_epochs
    }       
    return report

def final_training(X_train, Y_train, X_val, Y_val, X_test, Y_test, architecture, optimiser_params, report):
    model, min_val_loss, train_losses, val_losses, num_epochs = train(
        X_train=tf.convert_to_tensor(X_train), 
        Y_train=tf.convert_to_tensor(Y_train), 
        X_val=tf.convert_to_tensor(X_val), 
        Y_val=tf.convert_to_tensor(Y_val), 
        architecture=architecture,
        optimiser_params=optimiser_params,
        mode='final_training'
    )
    
    # Load the best weights saved during training
    model.model.load_weights('best_weights.weights.h5')
    model.model.export('../surrogate_model/final')

    test_loss = evaluate(model, X_test, Y_test)
    print(f'Test loss: {test_loss}')
    predictions = model.model.predict(X_test)

    # Convert predictions and ground truth to lists for JSON serialization
    predictions_list = predictions.tolist()
    ground_truth_list = Y_test.tolist()

    # Save predictions and ground truth to a JSON file
    results = {
        'predictions': predictions_list,
        'ground_truth': ground_truth_list
    }
    with open('report/report_files/inference_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    report['train_losses'] = train_losses
    report['val_losses'] = val_losses
    report['min_val_loss'] = round(min_val_loss, 6)
    report['test_loss'] = round(test_loss, 6)
    report['num_epochs'] = num_epochs

    return report