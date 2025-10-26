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