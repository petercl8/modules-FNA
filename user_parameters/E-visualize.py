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