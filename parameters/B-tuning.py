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