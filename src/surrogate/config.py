"""
configurations for cross validation and final training
"""

cross_val_architectures = [
#    [
#         {'type': 'lstm', 'units': 64, 'return_sequences': True},
#         {'type': 'batch_normalization'},
#         {'type': 'lstm', 'units': 32, 'return_sequences': True},
#         {'type': 'batch_normalization'},
#         {'type': 'lstm', 'units': 16, 'return_sequences': False},
#         {'type': 'batch_normalization'},
#         {'type': 'dense', 'units': 16},
#         {'type': 'batch_normalization'},
#         {'type': 'dense', 'units': 16},
#         {'type': 'batch_normalization'},
#         {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
#     ],
    # [
    #     {'type': 'lstm', 'units': 32, 'return_sequences': True},
    #     {'type': 'batch_normalization'},
    #     {'type': 'lstm', 'units': 16, 'return_sequences': True},
    #     {'type': 'batch_normalization'},
    #     {'type': 'lstm', 'units': 8, 'return_sequences': False},
    #     {'type': 'batch_normalization'},
    #     {'type': 'dense', 'units': 8},
    #     {'type': 'batch_normalization'},
    #     {'type': 'dense', 'units': 8},
    #     {'type': 'batch_normalization'},
    #     {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
    # ],
    # [
    #     {'type': 'lstm', 'units': 16, 'return_sequences': True},
    #     {'type': 'batch_normalization'},
    #     {'type': 'lstm', 'units': 8, 'return_sequences': True},
    #     {'type': 'batch_normalization'},
    #     {'type': 'lstm', 'units': 4, 'return_sequences': False},
    #     {'type': 'batch_normalization'},
    #     {'type': 'dense', 'units': 4},
    #     {'type': 'batch_normalization'},
    #     {'type': 'dense', 'units': 4},
    #     {'type': 'batch_normalization'},
    #     {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
    # ],
    # [
    #     {'type': 'reshape', 'target_shape': (7, 10, 1)},  
    #     {'type': 'conv2d', 'filters': 32, 'kernel_size': (5, 5), 'activation': 'relu', 'padding': 'same'},
    #     {'type': 'max_pooling2d', 'pool_size': (2, 2)},
    #     {'type': 'batch_normalization'},
    #     {'type': 'conv2d', 'filters': 32, 'kernel_size': (5, 5), 'activation': 'relu', 'padding': 'same'},
    #     {'type': 'max_pooling2d', 'pool_size': (2, 2)},
    #     {'type': 'batch_normalization'},
    #     {'type': 'flatten'},
    #     {'type': 'dense', 'units': 50, 'activation': 'relu'},
    #     {'type': 'batch_normalization'},
    #     {'type': 'dense', 'units': 50, 'activation': 'relu'},
    #     {'type': 'batch_normalization'},
    #     {'type': 'dense', 'units': 50, 'activation': 'relu'},
    #     {'type': 'batch_normalization'},
    #     {'type': 'dense', 'units': 50, 'activation': 'relu'},
    #     {'type': 'batch_normalization'},
    #     {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
    # ],
    # [
    #     {'type': 'transformer_encoder', 'num_heads': 4, 'intermediate_dim': 16, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_encoder', 'num_heads': 4, 'intermediate_dim': 16, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_decoder', 'num_heads': 4, 'intermediate_dim': 16, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'transformer_decoder', 'num_heads': 4, 'intermediate_dim': 16, 'dropout': 0.1, 'activation': 'relu'},
    #     {'type': 'flatten'},
    #     {'type': 'dense', 'units': 40},
    #     {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
    # ],
    [
        {'type': 'transformer_encoder', 'num_heads': 8, 'intermediate_dim': 16, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'transformer_encoder', 'num_heads': 8, 'intermediate_dim': 16, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'transformer_decoder', 'num_heads': 8, 'intermediate_dim': 16, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'transformer_decoder', 'num_heads': 8, 'intermediate_dim': 16, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'flatten'},
        {'type': 'dense', 'units': 40},
        {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
    ]
]

cross_val_optimiser_params = [
    {
        'optimiser': 'adam',
        'learning_rate': 0.001,
        'weight_decay': 0.0,
        'scheduler': 'None',
        'scheduler_params': 'None', 
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,  
        'weight_decay': 0.0,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.001,
            'warmup_steps': 1500,
            'max_lr': 0.01,
            'decay_steps': 100000,
            'alpha': 0.0001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,  
        'weight_decay': 0.00001,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.001,
            'warmup_steps': 1500,
            'max_lr': 0.01,
            'decay_steps': 100000,
            'alpha': 0.0001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,  
        'weight_decay': 0.0001,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.001,
            'warmup_steps': 1500,
            'max_lr': 0.01,
            'decay_steps': 100000,
            'alpha': 0.0001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,  
        'weight_decay': 0.0,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.0001,
            'warmup_steps': 1500,
            'max_lr': 0.01,
            'decay_steps': 100000,
            'alpha': 0.0001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,  
        'weight_decay': 0.00001,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.0001,
            'warmup_steps': 1500,
            'max_lr': 0.01,
            'decay_steps': 100000,
            'alpha': 0.0001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1, 
        'weight_decay': 0.0,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.001,
            'warmup_steps': 1500,
            'max_lr': 0.1,
            'decay_steps': 100000,
            'alpha': 0.0001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,  
        'weight_decay': 0.00001,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.001,
            'warmup_steps': 1500,
            'max_lr': 0.1,
            'decay_steps': 100000,
            'alpha': 0.0001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1,  
        'weight_decay': 0.0,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.001,
            'warmup_steps': 1500,
            'max_lr': 0.01,
            'decay_steps': 10000,
            'alpha': 0.001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1, 
        'weight_decay': 0.0001,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.001,
            'warmup_steps': 1500,
            'max_lr': 0.01,
            'decay_steps': 10000,
            'alpha': 0.001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.1, 
        'weight_decay': 0.0,
        'scheduler': 'cosine_decay',
        'scheduler_params': {
            'min_lr': 0.001,
            'warmup_steps': 5000,
            'max_lr': 0.1,
            'decay_steps': 10000,
            'alpha': 0.0001
        },
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.0001,
        'weight_decay': 0.0,
        'scheduler': 'None',
        'scheduler_params': 'None', 
        'batch_size': 64
    },
    {
        'optimiser': 'adam',
        'learning_rate': 0.001,
        'weight_decay': 0.00001,
        'scheduler': 'None',
        'scheduler_params': 'None', 
        'batch_size': 64
    }
]
   
final_training_architecture= [
        {'type': 'transformer_encoder', 'num_heads': 8, 'intermediate_dim': 16, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'transformer_encoder', 'num_heads': 8, 'intermediate_dim': 16, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'transformer_decoder', 'num_heads': 8, 'intermediate_dim': 16, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'transformer_decoder', 'num_heads': 8, 'intermediate_dim': 16, 'dropout': 0.1, 'activation': 'relu'},
        {'type': 'flatten'},
        {'type': 'dense', 'units': 40},
        {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
    ]

final_training_optimiser_params={
        'optimiser': 'adam',
        'learning_rate': 0.001,
        'weight_decay': 0.00001,
        'scheduler': 'None',
        'scheduler_params': 'None', 
        'batch_size': 64
    }