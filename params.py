# TODO: Make a run with different parameters and plot results
parameters = {
    'learning_rate': 0.01,
    'training_iters': 50000,
    'display_step': 10,
    'decay': 0.99995,
    # 16 input layer size (against the size 12 input) seems to be too little to capture the necessary features.
    # None seems to work better.
    'input_layer': None,
    # 6, 3 leads to high variance (difference between training and testing), so at least those are too large.
    # For single-layer LSTMs seem more stable, [16] works ok, but doesn't learn enough (high bias).
    # [24] is too high => high variance
    # [18] also has too much variance.
    'lstm_layers': [16],
    # How many targets are there
    'n_targets': 23,
    'n_peers': 2,
    # x, y for 3 targets
    # TODO: Add enabled flag
    'n_input': 3*4,
    # The minibatch is 16 sequences of 5 steps.
    'batch_size': 16,
    'n_steps': 5, # timesteps
    # x, y for 1 target. TODO: Add enabled flag.
    'n_output': 2,
    'lstm_clip': 10.0
}
