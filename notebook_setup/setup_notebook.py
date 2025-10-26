
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

    return project_dirPath