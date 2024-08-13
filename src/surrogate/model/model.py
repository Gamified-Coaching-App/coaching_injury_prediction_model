import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class Model(tf.Module):
    """
    This class builds a neural network model based on a given JSON configuration.
    """
    def __init__(self, config):
        """
        Initializes the model based on the given configuration.

        Parameters:
        - config: Configuration dictionary for the model, containing layer definitions and optimizer settings.
        """
        super(Model, self).__init__()
        self.config = config
        self.model = self.create_model(config['layers'])
        if config['optimiser'] == 'adam':
            self.optimizer = optimizers.Adam(learning_rate=config['learning_rate'])
            print("Using Adam optimizer")
        else:
            self.optimizer = optimizers.Adadelta(learning_rate=config['learning_rate'])
            print("Using Adadelta optimizer")

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

        return model