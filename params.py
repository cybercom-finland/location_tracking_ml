# TODO: Make a run with different parameters and plot results
parameters = {
    'learning_rate': 0.01,
    'display_step': 40,
    'input_layer': 20, # This should be at least 2*3 = 6
    'lstm_layers': [16, 14], # The last layer should be at least 12
    # How many targets are there
    'n_targets': 22,
    'n_peers': 1,
    # x, y for 2 targets
    # TODO: Add enabled flag
    'n_input': 2*2,
    'batch_size': 16,
    'n_steps': 6, # timesteps
    # x, y mixtures for 1 target. TODO: Add enabled flag.
    # Note: This must currently always be two because of bivariate gaussian mixture network
    'n_output': 2,
    'lstm_clip': 200.0,
    # For the mixture density network, how many mixtures we use per variable.
    'n_mixtures': 2,
    'keep_prob': 1.0,
    'clip_gradients': 100.0
}
