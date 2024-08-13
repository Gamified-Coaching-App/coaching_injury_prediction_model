model_configs = [
    {
        'layers': [
                    {'type': 'lstm', 'units': 32, 'return_sequences': False},
                    {'type' : 'batch_normalisation'},
                    {'type': 'dense', 'units': 1, 'activation': 'sigmoid', 'kernel_initializer': 'glorot_normal'}
                ], 
                'learning_rate': 2.0,
                'optimiser': 'adadelta'
    }
]