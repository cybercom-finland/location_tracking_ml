# TODO: Make a run with different parameters and plot results
parameters = {
    'learning_rate': 0.02,
    'training_iters': 10000,
    'display_step': 10,
    'decay': 1.0, #0.99995,
    # 16 input layer size (against the size 12 input) seems to be too little to capture the necessary features.
    # None seems to work better.
    'input_layer': None,
    # 6, 3 leads to high variance (difference between training and testing), so at least those are too large.
    # For single-layer LSTMs seem more stable.
    # [12] also has too much variance.
    # [6] has too big bias, and doesn't learn well.
    'lstm_layers': [7],
    # How many targets are there
    'n_targets': 23,
    'n_peers': 2,
    # x, y for 3 targets
    # TODO: Add enabled flag
    'n_input': 3*4,
    # The minibatch is 32 sequences of 5 steps.
    'batch_size': 32,
    'n_steps': 5, # timesteps
    # x, y, dx, dy for 1 target. TODO: Add enabled flag.
    'n_output': 4,
    'lstm_clip': 10.0
}
