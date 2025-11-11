import math

def compute_display_params(config, settings):
    """
    Compute batch_size and display_step based on run mode.
    Returns (batch_size, display_step).
    """
    run_mode = settings['run_mode']

    if run_mode == 'tune':
        batch_size = config['batch_size']
        if settings.get('tune_even_reporting', False):
            tune_examples_per_report = settings.get('tune_examples_per_report', 5120)
            display_step = max(1, round(tune_examples_per_report / batch_size))
        else:
            display_step = max(1, int(settings.get('tune_iter_per_report', 1)))
    elif run_mode == 'train':
        batch_size = config['batch_size']
        display_step = settings['train_display_step']
    elif run_mode == 'test':
        batch_size = settings.get('test_batch_size', config['batch_size'])
        display_step = settings['test_display_step']
    elif run_mode == 'visualize':
        batch_size = settings.get('visualize_batch_size', 9)
        display_step = 1
    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")

    return batch_size, display_step


def get_tune_session():
    """
    Safely import and return Ray Tune session if available.
    Returns None if Ray Tune is not installed.
    """
    try:
        from ray.air import session
        return session
    except ImportError:
        return None