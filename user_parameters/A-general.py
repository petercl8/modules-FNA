#####################
### General Setup ###
#####################

# Basic Options #
run_mode='tune'  # Options: 'tune' / 'train' / 'test' / 'visualize' (visualize data set)
sino_size=180          # Resize input sinograms to this size (integer). Sinograms are square, which was found to give the best results.
sino_channels=3       # Number of channels (sinograms). Options: 1, 3. Unless using scattered coincidences, set to 1.
image_size=90         # Image size (Options: 90, 180). Images are square.
image_channels=1      # Number of channels (images)
train_type='SUP'      # 'SUP' / 'GAN' / 'CYCLESUP' / 'CYCLEGAN' = (Supervisory only/GAN/Cycle consistency+supervisory/CycleGAN)
train_SI=True         # If training GAN or SUP, set True to train Gen_SI (Sinogram-->Image), or False to train Gen_IS (Image-->Sinogram)

# Github Repository for Functions & Classes #
github_username='petercl8'
repo_name='modules-FNA'

# Directories #
project_colab_dirPath = '/content/drive/MyDrive/Colab/Working/'     # Directory, relative to which all other directories are specified (if working on Colab)
project_local_dirPath = r'C:\Users\Peter Lindstrom\Documents\Projects\MyProject'  # Directory, relative to which all other directories are specified (if working Locally)

data_dirName = 'dataset-sets'      # Dataset directory, placed in project directory (above)
plot_dirName=  'plots'             # Plots Directory, placed in project directory (above)
checkpoint_dirName='checkpoints'   # If not using Ray Tune (not tuning), PyTorch saves and loads checkpoint file from here
                                   # All checkpoint files (for training, testing, visualizing) save the states for a particular network.
                                   # Therefore, the hyperparameters for the loaded CNN must match the data in the checkpoint file.
                                   # The configuration dictionary, which contains these hyperparameter values, is set in the 'Supervisory" cell, below.