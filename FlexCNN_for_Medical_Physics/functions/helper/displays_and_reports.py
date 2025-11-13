import math

def compute_display_step(config, settings):
    """
    Compute batch_size and display_step based on run mode.
    Returns (batch_size, display_step).
    """
    run_mode = settings['run_mode']

    if run_mode == 'tune':
        if settings.get('tune_even_reporting', False):
            tune_examples_per_report = settings.get('tune_examples_per_report', 5120)
            display_step = max(1, round(tune_examples_per_report / batch_size))
        else:
            display_step = max(1, int(settings.get('tune_batches_per_report', 1)))
    elif run_mode == 'train':
        display_step = settings['train_display_step']
    elif run_mode == 'test':
        display_step = settings['test_display_step']
    elif run_mode == 'visualize':
        display_step = 1
    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")

    return display_step


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