import os
import torch

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

    return {'device': device, 'paths': paths, 'settings': settings}
