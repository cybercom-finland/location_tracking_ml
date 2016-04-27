# TODO: Make a run with different parameters and plot results
parameters = {
    'learning_rate': 0.01, #0.01,
    'training_iters': 400000000,
    'display_step': int(400000000 / 1024 / 10000) + 1,
    # Should not put too many display steps here, because accumulation of data takes memory. Doing ~10000 print outs.
    'decay': 1.0, #0.999, # 0.99, #0.99995,
    'input_layer': 12, # This should be at least 2*3 = 6
    'lstm_layers': [10, 10, 10], # The last layer should be at least 6
    # How many targets are there
    'n_targets': 22,
    'n_peers': 1,
    # x, y, dx, dy for 3 targets
    # TODO: Add enabled flag
    'n_input': 2*2,
    # The minibatch is 2048 * 8 sequences of 5 steps.
    'batch_size': 1024, #4096,
    'n_steps': 2, # timesteps
    # x, y mixtures for 1 target. TODO: Add enabled flag.
    # Note: This must currently always be two because of bivariate gaussian mixture network
    'n_output': 2,
    'lstm_clip': 50.0,
    # For the mixture density network, how many mixtures we use per variable.
    'n_mixtures': 2,
    'keep_prob': 1.0,
    'clip_gradients': 10.0
}
