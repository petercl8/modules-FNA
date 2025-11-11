import os


def setup_paths(base_dirs, data_files, mode_files, run_mode, test_set_type='test', visualize_type='train'):
    """
    Build all path-related configuration.
    
    Args:
        base_dirs: dict with keys: project_dirPath, plot_dirName, checkpoint_dirName, tune_storage_dirName,
                   tune_dataframe_dirName, test_dataframe_dirName, data_dirName
        data_files: dict with keys: tune_sino_file, tune_image_file, train_sino_file, train_image_file,
                    test_sino_file, test_image_file
        mode_files: dict with keys: train_checkpoint_file, test_checkpoint_file, visualize_checkpoint_file,
                    tune_csv_file, test_csv_file
        run_mode: 'tune', 'train', 'test', or 'visualize'
        test_set_type: 'test' or 'train'
        visualize_type: 'test' or 'train'
    
    Returns:
        paths dict with directory paths, mode-specific data paths, active sino/image paths, checkpoint_path,
        tune/test dataframe paths.
    """
    paths = {}
    
    # Base directories
    paths['plot_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['plot_dirName'])
    paths['checkpoint_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['checkpoint_dirName'])
    paths['tune_storage_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['tune_storage_dirName'])
    paths['tune_dataframe_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['tune_dataframe_dirName'])
    paths['test_dataframe_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['test_dataframe_dirName'])
    paths['data_dirPath'] = os.path.join(base_dirs['project_dirPath'], base_dirs['data_dirName'])
    
    # Mode-specific data file paths
    paths['tune_sino_path'] = os.path.join(paths['data_dirPath'], data_files['tune_sino_file'])
    paths['tune_image_path'] = os.path.join(paths['data_dirPath'], data_files['tune_image_file'])
    paths['train_sino_path'] = os.path.join(paths['data_dirPath'], data_files['train_sino_file'])
    paths['train_image_path'] = os.path.join(paths['data_dirPath'], data_files['train_image_file'])
    paths['test_sino_path'] = os.path.join(paths['data_dirPath'], data_files['test_sino_file'])
    paths['test_image_path'] = os.path.join(paths['data_dirPath'], data_files['test_image_file'])
    
    # Active paths and checkpoint filename selection
    if run_mode == 'tune':
        paths['sino_path'] = paths['tune_sino_path']
        paths['image_path'] = paths['tune_image_path']
        checkpoint_file = ''
    elif run_mode == 'train':
        paths['sino_path'] = paths['train_sino_path']
        paths['image_path'] = paths['train_image_path']
        checkpoint_file = mode_files['train_checkpoint_file']
    elif run_mode == 'test':
        if test_set_type == 'test':
            paths['sino_path'] = paths['test_sino_path']
            paths['image_path'] = paths['test_image_path']
        else:
            paths['sino_path'] = paths['train_sino_path']
            paths['image_path'] = paths['train_image_path']
        checkpoint_file = mode_files['test_checkpoint_file']
    elif run_mode == 'visualize':
        if visualize_type == 'test':
            paths['sino_path'] = paths['test_sino_path']
            paths['image_path'] = paths['test_image_path']
        else:
            paths['sino_path'] = paths['train_sino_path']
            paths['image_path'] = paths['train_image_path']
        checkpoint_file = mode_files['visualize_checkpoint_file']
    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")
    
    # Checkpoint path
    paths['checkpoint_path'] = os.path.join(paths['checkpoint_dirPath'], checkpoint_file)
    
    # Dataframe paths (always constructed for clarity)
    paths['tune_dataframe_path'] = os.path.join(paths['tune_dataframe_dirPath'], f"{mode_files['tune_csv_file']}.csv")
    paths['test_dataframe_path'] = os.path.join(paths['test_dataframe_dirPath'], f"{mode_files['test_csv_file']}.csv")
    
    return paths


def setup_settings(common_settings, tune_opts, train_opts, test_opts, viz_opts, run_mode):
    """
    Build all non-path runtime settings.
    
    Args:
        common_settings: dict with keys: device, offset, num_examples, sample_division,
                         train_display_step, test_display_step, visualize_batch_size, test_batch_size
        tune_opts: dict with keys: tune_augment, tune_iter_per_report, tune_examples_per_report,
                   tune_even_reporting, tune_max_t, tune_dataframe_fraction
        train_opts: dict with keys: train_augment, training_epochs, train_load_state, train_save_state,
                    train_show_times, train_sample_division
        test_opts: dict with keys: test_show_times
        viz_opts: dict with keys: visualize_shuffle, visualize_offset
        run_mode: 'tune', 'train', 'test', or 'visualize'
    
    Returns:
        settings dict containing runtime (non-path) configuration.
    """
    settings = {}
    
    # Common settings
    settings['run_mode'] = run_mode
    settings['device'] = common_settings['device']
    settings['train_display_step'] = common_settings['train_display_step']
    settings['test_display_step'] = common_settings['test_display_step']
    settings['visualize_batch_size'] = common_settings['visualize_batch_size']
    settings['test_batch_size'] = common_settings['test_batch_size']
    settings['offset'] = common_settings.get('offset', 0)
    settings['num_examples'] = common_settings.get('num_examples', -1)
    settings['sample_division'] = common_settings.get('sample_division', 1)
    
    # Mode-specific
    if run_mode == 'tune':
        settings['augment'] = tune_opts['tune_augment']
        settings['shuffle'] = True
        settings['num_epochs'] = 1000
        settings['load_state'] = False
        settings['save_state'] = False
        settings['offset'] = 0
        settings['show_times'] = False
        settings['sample_division'] = 1
        settings['tune_iter_per_report'] = tune_opts.get('tune_iter_per_report')
        settings['tune_even_reporting'] = tune_opts.get('tune_even_reporting', False)
        settings['tune_max_t'] = tune_opts.get('tune_max_t', 0)
        settings['tune_dataframe_fraction'] = tune_opts.get('tune_dataframe_fraction', 1.0)
        tune_examples_per_report = tune_opts.get('tune_examples_per_report')
        tune_iter_per_report = tune_opts.get('tune_iter_per_report')
        if tune_examples_per_report is None and tune_iter_per_report is not None:
            settings['tune_examples_per_report'] = 512 * tune_iter_per_report
        else:
            settings['tune_examples_per_report'] = tune_examples_per_report if tune_examples_per_report is not None else 5120
    
    elif run_mode == 'train':
        settings['augment'] = train_opts['train_augment']
        settings['shuffle'] = True
        settings['num_epochs'] = train_opts['training_epochs']
        settings['load_state'] = train_opts['train_load_state']
        settings['save_state'] = train_opts['train_save_state']
        settings['offset'] = 0
        settings['show_times'] = train_opts['train_show_times']
        settings['sample_division'] = train_opts['train_sample_division']
    
    elif run_mode == 'test':
        settings['augment'] = False
        settings['shuffle'] = False
        settings['num_epochs'] = 1
        settings['load_state'] = True
        settings['save_state'] = False
        settings['offset'] = 0
        settings['show_times'] = test_opts['test_show_times']
        settings['sample_division'] = 1
    
    elif run_mode == 'visualize':
        settings['augment'] = False
        settings['shuffle'] = viz_opts['visualize_shuffle']
        settings['num_epochs'] = 1
        settings['load_state'] = True
        settings['save_state'] = False
        settings['show_times'] = False
        settings['offset'] = viz_opts['visualize_offset']
        settings['sample_division'] = 1
    else:
        raise ValueError(f"Unknown run_mode: {run_mode}")
    
    return settings


def construct_config(
    run_mode,
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

    ####
    ## Combine Dictionaries ##
    if run_mode=='train' or 'test' or 'visualize':
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

    if run_mode=='tune':
        if train_type=='SUP':
            config = {**(config_RAY_SI if train_SI else config_RAY_IS), **config_RAY_SUP}
        elif train_type=='GAN':
            config = {**(config_RAY_SI if train_SI else config_RAY_IS), **config_RAY_GAN}
        elif train_type=='CYCLESUP':
            config = {**config_SUP_SI, **config_SUP_IS, **config_SUP_RAY_cycle}
        if train_type=='CYCLEGAN':
            config = {**config_GAN_SI, **config_GAN_IS, **config_GAN_RAY_cycle}
        else :
            raise ValueError(f"Unknown train_type '{train_type}'.")

    # Add data dimensions to config
    config['image_size'] = image_size
    config['sino_size'] = sino_size
    config['image_channels'] = image_channels
    config['sino_channels'] = sino_channels
    config['train_SI'] = train_SI

    return config