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