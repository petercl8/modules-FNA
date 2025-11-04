import os
import torch
import sys

def setup_project_dirs(IN_COLAB, project_local_dirPath, project_colab_dirPath=None, mount_colab_drive=True):
    """
    Sets up project directories and adds the project path to sys.path.

    Parameters
    ----------
    project_local_dirPath : str
        Path to the project on the local machine.
    project_colab_dirPath : str, optional
        Path to the project on Google Drive (used only in Colab).
    mount_colab_drive : bool, default True
        Whether to mount Google Drive if running in Colab.

    Returns
    -------
    str
        The path being used for the project (Colab or local).
    """

    # --- Determine project directory ---
    if IN_COLAB and project_colab_dirPath is not None:
        if mount_colab_drive:
            from google.colab import drive
            drive.mount('/content/drive')
        project_dirPath = project_colab_dirPath
    else:
        project_dirPath = project_local_dirPath

    # --- Add project directory to sys.path if not already present ---
    if project_dirPath not in sys.path:
        sys.path.insert(0, project_dirPath)

    # --- Optional debugging: show current sys.path ---
    # for p in sys.path:
    #     print("   ", p)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return project_dirPath

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

    return paths, settings

def construct_config(
    run_mode,
    train_type,
    train_SI,
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
    Combines configuration dictionaries based on run_mode, train_type, and train_SI.
    Returns the appropriate config dictionary.
    """

    # Normalize strings (avoid accidental None or case mismatch)
    run_mode = str(run_mode).lower()
    train_type = str(train_type).upper()

    # Train, test, or visualize modes
    if run_mode in ['train', 'test', 'visualize']:
        if train_type == 'SUP':
            config = config_SUP_SI if train_SI else config_SUP_IS
        elif train_type == 'GAN':
            config = config_GAN_SI if train_SI else config_GAN_IS
        elif train_type == 'CYCLEGAN':
            config = config_CYCLEGAN
        elif train_type == 'CYCLESUP':
            config = config_CYCLESUP
        else:
            raise ValueError(f"Unknown train_type '{train_type}' for run_mode '{run_mode}'.")

    # Tune mode
    elif run_mode == 'tune':
        if train_type == 'SUP':
            config = {**(config_RAY_SI if train_SI else config_RAY_IS), **config_RAY_SUP}
        elif train_type == 'GAN':
            config = {**(config_RAY_SI if train_SI else config_RAY_IS), **config_RAY_GAN}
        elif train_type == 'CYCLESUP':
            config = {**config_SUP_SI, **config_SUP_IS, **config_SUP_RAY_cycle}
        elif train_type == 'CYCLEGAN':
            config = {**config_GAN_SI, **config_GAN_IS, **config_GAN_RAY_cycle}
        else:
            raise ValueError(f"Unknown train_type '{train_type}' for run_mode '{run_mode}'.")
    else:
        raise ValueError(f"Unknown run_mode '{run_mode}'.")

    return config

def list_compute_resources(check_ray_tune=False):
    """
    Quickly lists available CPUs and GPUs.
    Optionally checks Ray/Ray Tune resources if check_ray_tune=True.
    """
    import os
    import multiprocessing
    import torch

    # Basic system info
    cpu_count = multiprocessing.cpu_count()
    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if gpu_count else []

    print(f"CPUs available: {cpu_count}")
    print(f"GPUs available: {gpu_count}")
    if gpu_count:
        for i, name in enumerate(gpu_names):
            print(f"  GPU {i}: {name}")

    # Optional Ray/Ray Tune resource check
    if check_ray_tune:
        import ray
        ray.shutdown()
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        print("\nRay resources:")
        print(ray.available_resources())

