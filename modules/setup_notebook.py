
def install_required_packages(force_reinstall=False, include_optional=True):
    """
    Installs required Python packages efficiently.
    - Detects if running in Colab or locally.
    - Installs missing packages only (unless force_reinstall=True).
    - Ensures Ray Tune dependencies are installed even if ray is already present.
    """

    # Base list of packages
    packages = [
        "torch", "torchvision", "torchaudio",
        "ray[tune]", "tensorboardX", "hyperopt",
        "numpy", "pandas", "matplotlib",
        "scikit-image", "scipy"
    ]

    # Optional packages for visualization
    optional_packages = ["tensorboard"]
    # Widgets for tqdm are optional; plain progress bar is fine
    widgets_packages = ["ipywidgets"]

    # Detect environment
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    missing = []

    for pkg in packages:
        pkg_name = pkg.split("[")[0]

        # Special handling for Ray Tune
        if pkg_name == "ray":
            try:
                import ray
                import ray.tune
                ray_tune_installed = True
            except ImportError:
                ray_tune_installed = False
            if force_reinstall or not ray_tune_installed:
                missing.append(pkg)
            continue

        # General case
        if importlib.util.find_spec(pkg_name) is None or force_reinstall:
            missing.append(pkg)

    # Optionally add optional packages
    if include_optional:
        missing += optional_packages + widgets_packages

    if not missing:
        print("✅ All required packages already installed.")
        return

    print(f"📦 Installing missing packages: {', '.join(missing)}")

    # Build pip command
    if in_colab:
        cmd = ["pip", "install", "--upgrade"] + missing
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + missing

    try:
        subprocess.check_call(cmd)
        print("✅ Installation complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")

def setup_project_dirs(project_local_dirPath, project_colab_dirPath=None, mount_colab_drive=True):
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
    # --- Detect environment ---
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

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
