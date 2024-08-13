model_configs = [
    {
        ## Possible models: LSTM or Dense: Currently Dense hat best performace: 0.0031 (64, 32, 16) vs. 0.0035 with lstm
        'layers': [
                    {'type': 'flatten'},
                    # {'type': 'dense', 'units': 256, 'activation': 'relu'},
                    # {'type': 'dropout', 'rate': 0.15},
                    #{'type' : 'batch_normalisation'},
                    {'type': 'dense', 'units': 64, 'activation': 'relu'},
                    {'type': 'dropout', 'rate': 0.02},
                    #{'type' : 'batch_normalisation'},
                    {'type': 'dense', 'units': 32, 'activation': 'relu'},
                    {'type': 'dropout', 'rate': 0.02},
                    #{'type' : 'batch_normalisation'},
                    {'type': 'dense', 'units': 16, 'activation': 'relu'},
                    #{'type': 'dense', 'units': 8, 'activation': 'relu'},
                    {'type': 'dropout', 'rate': 0.02},
                    #{'type' : 'batch_normalisation'},
                    # {'type': 'lstm', 'units': 32, 'return_sequences': False},
                    # {'type' : 'batch_normalisation'},
                    {'type': 'dense', 'units': 1, 'activation': 'sigmoid', 'kernel_initializer': 'glorot_normal'}
                ], 
                'learning_rate': 2.0,
                'optimiser': 'adadelta'
    }
]