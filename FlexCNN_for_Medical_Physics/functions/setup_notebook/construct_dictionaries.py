import os
import torch
import sys


def construct_config(
    train_type,
    train_SI,
    image_size,
    sino_size,
    image_channels,
    sino_channels,
    config_SUP_SI=None,
    config_SUP_IS=None,
    config_GAN_SI=None,
    config_GAN_IS=None,
    config_CYCLEGAN=None,
    config_CYCLESUP=None,
    config_RAY_SI=None,
    config_RAY_IS=None,
    config_RAY_SUP=None,
    config_RAY_GAN=None,
    config_SUP_RAY_cycle=None,
    config_GAN_RAY_cycle=None):

    """
    Combines configuration dictionaries based on train_type and train_SI.
    Returns the appropriate config dictionary.
    """

    # Normalize strings (avoid accidental None or case mismatch)
    train_type = str(train_type).upper()

    # Train, test, or visualize modes
    if train_type == 'SUP':
        config = config_SUP_SI if train_SI else config_SUP_IS
    elif train_type == 'GAN':
        config = config_GAN_SI if train_SI else config_GAN_IS
    elif train_type == 'CYCLEGAN':
        config = config_CYCLEGAN
    elif train_type == 'CYCLESUP':
        config = config_CYCLESUP
    else:
        raise ValueError(f"Unknown train_type '{train_type}'.")

    # Add data dimensions to config
    config['image_size'] = image_size
    config['sino_size'] = sino_size
    config['image_channels'] = image_channels
    config['sino_channels'] = sino_channels
    config['train_SI'] = train_SI

    return config


def setup_run_paths(
    project_dirPath,
    plot_dirName,
    checkpoint_dirName,
    tune_storage_dirName,
    tune_dataframe_dirName,
    test_dataframe_dirName,
    data_dirName,
    tune_sino_file,
    tune_image_file,
    train_sino_file,
    train_image_file,
    test_sino_file,
    test_image_file,
    run_mode,
    device,
    train_display_step,
    test_display_step,
    visualize_batch_size,
    test_batch_size,
    # Dataset slicing controls
    offset=0,
    num_examples=-1,
    sample_division=1,
    tune_iter_per_report=None,
    tune_examples_per_report=None,
    tune_even_reporting=False,
    tune_augment=None,
    train_augment=None,
    training_epochs=None,
    train_load_state=False,
    train_save_state=False,
    train_checkpoint_file='',
    train_show_times=False,
    train_sample_division=1,
    test_set_type='test',
    test_checkpoint_file='',
    test_show_times=False,
    visualize_type='train',
    visualize_shuffle=False,
    visualize_checkpoint_file='',
    visualize_offset=0,
    tune_max_t=None,
    tune_csv_file='tune',
    test_csv_file='test'
    ):
    """
    Sets up paths, filenames, and run-specific settings for a project.
    Returns a dictionary with all relevant paths and configuration values.
    """
    
    # Base directories
    paths = {}
    paths['plot_dirPath'] = os.path.join(project_dirPath, plot_dirName)    
    paths['checkpoint_dirPath'] = os.path.join(project_dirPath, checkpoint_dirName)
    paths['tune_storage_dirPath'] = os.path.join(project_dirPath, tune_storage_dirName)
    paths['tune_dataframe_dirPath'] = os.path.join(project_dirPath, tune_dataframe_dirName)
    paths['test_dataframe_dirPath'] = os.path.join(project_dirPath, test_dataframe_dirName)
    paths['data_dirPath'] = os.path.join(project_dirPath, data_dirName)

    # Data files
    paths['tune_sino_path'] = os.path.join(paths['data_dirPath'], tune_sino_file)
    paths['tune_image_path'] = os.path.join(paths['data_dirPath'], tune_image_file)
    paths['train_sino_path'] = os.path.join(paths['data_dirPath'], train_sino_file)
    paths['train_image_path'] = os.path.join(paths['data_dirPath'], train_image_file)
    paths['test_sino_path'] = os.path.join(paths['data_dirPath'], test_sino_file)
    paths['test_image_path'] = os.path.join(paths['data_dirPath'], test_image_file)

    # Run-mode specific settings
    settings = {}
    settings['run_mode'] = run_mode
    settings['device'] = device
    settings['train_display_step'] = train_display_step
    settings['test_display_step'] = test_display_step
    settings['visualize_batch_size'] = visualize_batch_size
    settings['test_batch_size'] = test_batch_size
    # Dataset slicing controls live in settings for all run modes
    settings['offset'] = offset
    settings['num_examples'] = num_examples
    settings['sample_division'] = sample_division

    if run_mode == 'tune':
        settings.update({
            'sino_path': paths['tune_sino_path'],
            'image_path': paths['tune_image_path'],
            'augment': tune_augment,
            'shuffle': True,
            'num_epochs': 1000,
            'load_state': False,
            'save_state': False,
            'checkpoint_file': '',
            'offset': 0,
            'show_times': False,
            'sample_division': 1
        })
        os.makedirs(paths['tune_dataframe_dirPath'], exist_ok=True)
        settings['tune_dataframe_path'] = os.path.join(paths['tune_dataframe_dirPath'], f"{tune_csv_file}.csv")
        # Tuning reporting controls: either fixed iterations or even reporting by example count
        settings['tune_iter_per_report'] = tune_iter_per_report
        settings['tune_even_reporting'] = tune_even_reporting
        # Default baseline: 512 examples * 10 iterations = 5120 examples per report
        if tune_examples_per_report is None and tune_iter_per_report is not None:
            settings['tune_examples_per_report'] = 512 * tune_iter_per_report
        else:
            settings['tune_examples_per_report'] = tune_examples_per_report if tune_examples_per_report is not None else 5120

    elif run_mode == 'train':
        settings.update({
            'sino_path': paths['train_sino_path'],
            'image_path': paths['train_image_path'],
            'augment': train_augment,
            'shuffle': True,
            'num_epochs': training_epochs,
            'load_state': train_load_state,
            'save_state': train_save_state,
            'checkpoint_file': train_checkpoint_file,
            'offset': 0,
            'show_times': train_show_times,
            'sample_division': train_sample_division
        })

    elif run_mode == 'test':
        if test_set_type == 'test':
            settings.update({'sino_path': paths['test_sino_path'], 'image_path': paths['test_image_path']})
        else:
            settings.update({'sino_path': paths['train_sino_path'], 'image_path': paths['train_image_path']})
        settings.update({
            'augment': False,
            'shuffle': False,
            'num_epochs': 1,
            'load_state': True,
            'save_state': False,
            'checkpoint_file': test_checkpoint_file,
            'offset': 0,
            'show_times': test_show_times,
            'sample_division': 1
        })

    elif run_mode == 'visualize':
        if visualize_type == 'test':
            settings.update({'sino_path': paths['test_sino_path'], 'image_path': paths['test_image_path']})
        else:
            settings.update({'sino_path': paths['train_sino_path'], 'image_path': paths['train_image_path']})
        settings.update({
            'augment': False,
            'shuffle': visualize_shuffle,
            'num_epochs': 1,
            'load_state': True,
            'save_state': False,
            'checkpoint_file': visualize_checkpoint_file,
            'show_times': False,
            'offset': visualize_offset,
            'sample_division': 1
        })

    # Other paths
    paths['test_dataframe_path'] = os.path.join(paths['test_dataframe_dirPath'], f"{test_csv_file}.csv")
    paths['checkpoint_path'] = os.path.join(paths['checkpoint_dirPath'], settings['checkpoint_file'])

    return paths, settings

