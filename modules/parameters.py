#####################
### General Setup ###
#####################

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

# Basic Options #
run_mode='tune'  # Options: 'tune' / 'train' / 'test' / 'visualize' (visualize data set)
sino_size=180          # Resize input sinograms to this size (integer). Sinograms are square, which was found to give the best results.
sino_channels=3       # Number of channels (sinograms). Options: 1, 3. Unless using scattered coincidences, set to 1.
image_size=90         # Image size (Options: 90, 180). Images are square.
image_channels=1      # Number of channels (images)
train_type='SUP'      # 'SUP' / 'GAN' / 'CYCLESUP' / 'CYCLEGAN' = (Supervisory only/GAN/Cycle consistency+supervisory/CycleGAN)
train_SI=True         # If training GAN or SUP, set True to train Gen_SI (Sinogram-->Image), or False to train Gen_IS (Image-->Sinogram)


############
## Tuning ##
############
# Note: When tuning, ALWAYS select "restart session and run all" from Runtime menu in Google Colab, or there may be bugs.
tune_storage_dirName=''     # Create tuning folders (one for each run, each of which contains multiple trials) in this directory. Leave blank to place search files project directory
tune_scheduler = 'ASHA'     # Use FIFO for simple first in/first out to train to the end, or ASHA for utilizing early stopping poorly performing trials.
tune_dataframe_dirName= 'Dataframes-TuneTemp'  # Directory for tuning dataframe (stores network information for each network trialed). Code will create it if it doesn't exist.
tune_csv_file='frame-tunedOnLowSSIM-tunedSSIM-ASHA' # .csv file to save tuning dataframe to
tune_exp_name='search-Temp'                         # Experiment directory: Ray tune (and Tensorboard) write to this directory, relative to the local dir this notebook is run from.
tune_dataframe_fraction=0.33# At what fraction of the max tuning steps (tune_max_t) do you save values to the tuning dataframe.
tune_restore=False          # Restore a run (from the file tune_exp_name in tune_storage_dirPath). Use this if a tuning run terminated early for some reason.
tune_max_t = 10             # Maximum number of reports per network. For even training example reporting (reports made at a constant number of training
                            # examples), 20 is a good number for ASHA. For FIFO, 10 is a good number.
                            # For constant batch size reporting (tune_even_reporting=False), 35 works well.
tune_minutes = 30           # How long to run RayTune. 180 minutes is good for 90x90 input. 210 minutes for 180x180.
tune_for = 'SSIM'           # Tune for which optimization metric?: 'MSE', 'SSIM', or 'CUSTOM' (user defined, defined later in code)
tune_even_reporting=True    # Set to True to ensure we report to Raytune at an even number of training examples,
                            # regardless of batch size.
tune_iter_per_report=10     # If tune_even_reporting = False, this is the number of batches per report (30 works pretty well).
                            # Default value = 10.
                            # If tune_even_reporting = True, this is the number of training iterations per Raytune report for a batch size = 512.
                            # For a batch size = 256, the iterations/report would be twice this number. For batch size # = 128, it would be four
                            # times, etc.
tune_augment=True           # Augment data (on the fly) for tuning?
num_CPUs=4                  # Number of CPUs to use
num_GPUs=1                  # Number of GPUs to use


## Select Data Files ##
## ----------------- ##
#tune_sino_file=  'tune_sino-10k.npy'
#tune_image_file= 'tune_image-10k.npy'
#tune_sino_file= 'train_sino-highMSE-17500.npy'
#tune_image_file='train_image-highMSE-17500.npy'
#tune_sino_file= 'train_sino-lowMSE-17500.npy'
#tune_image_file='train_image-lowMSE-17500.npy'
tune_sino_file= 'train_sino-lowSSIM-17500.npy'
tune_image_file='train_image-lowSSIM-17500.npy'



##############
## Training ##
##############
train_load_state=False      # Set to True to load pretrained weights. Use if training terminated early.
train_save_state=False      # Save network weights to train_checkpoint_file file as it trains
train_checkpoint_file='checkpoint-tunedLowSSIM-trainedHighSSIM-100epochs' # Checkpoint file to load or save to
training_epochs = 100      # Number of training epochs.
train_augment=True         # Augment data (on the fly) for training?
train_display_step=10      # Number of steps/visualization. Good values: for supervised learning or GAN, set to: 20, For cycle-consistent, set to 10
train_sample_division=1    # To evenly sample the training set by a given factor, set this to an integer greater than 1 (ex: to sample every other example, set to 2)
train_show_times=False     # Show calculation times during training?


## Select Data Files ##
## ----------------- ##
#train_sino_file= 'train_sino-70k.npy'
#train_image_file='train_image-70k.npy'
#train_sino_file= 'train_sino-highMSE-17500.npy'
#train_image_file='train_image-highMSE-17500.npy'
#train_sino_file= 'train_sino-lowMSE-17500.npy'
#train_image_file='train_image-lowMSE-17500.npy'
train_sino_file= 'train_sino-lowSSIM-17500.npy'
train_image_file= 'train_image-lowSSIM-17500.npy'



###########
# Testing #
###########
test_dataframe_dirName= 'TestOnFull'  # Directory for test metric dataframes
test_csv_file = 'combined-tunedLowSSIM-trainedLowSSIM-onTestSet-wMLEM' # csv dataframe file to save testing results to
test_checkpoint_file='checkpoint-tunedLowSSIM-trainedLowSSIM-100epochs' # Checkpoint to load model for testing
test_display_step=15        # Make this a larger number to save bit of time (displays images/metrics less often)
test_batch_size=25          # This doesn't affect the final metrics, just the displayed metrics as testing procedes
test_chunk_size=875              # How many examples do you want to test at once? NOTE: This should be a multiple of test_batch_size AND also go into the test set size evenly.
testset_size=35000          # Size of the set to test. This must be <= the number of examples in your test set file.
test_begin_at=0             # Begin testing at this example number.
compute_MLEM=False          # Compute a simple MLEM reconstruction from the sinograms when running testing.
                            # This takes a lot longer. If set to false, only FBP is calculated.
test_set_type='test'        # Set to 'test' to test on the test set. Set to 'train' to test on the training set.
test_merge_dataframes=True  # Merge the smaller/chunked dataframes at the end of the test run into one large dataframe?
test_show_times=False       # Show calculation times?


## Select Data Files ##
## ----------------- ##
test_sino_file=  'test_sino-35k.npy'
test_image_file= 'test_image-35k.npy'
#test_sino_file= 'test_sino-highMSE-8750.npy'
#test_image_file= 'test_image-highMSE-8750.npy'
#test_sino_file= 'test_sino-lowMSE-8750.npy'
#test_image_file= 'test_image-lowMSE-8750.npy'


####################
## Visualize Data ##
####################

#visualize_checkpoint_file='checkpoint-90x1-tunedMSE-fc6-6epochs' # Checkpoint file to load/save
visualize_checkpoint_file='checkpoint-tunedHigh-trainedHigh-100epochs'
visualize_batch_size = 10   # Set value to exactly 120 to see a large grid of images OR =<10 for reconstructions
                            #  and ground truth with matched color scales
visualize_offset=0          # Image to begin at. Set to 0 to start at beginning.
visualize_type='train'      # Set to 'test' or 'train' to visualize the test set or training set, respectively
visualize_shuffle=True      # Shuffle data set when visualizing?