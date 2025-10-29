import torch
import numpy as np
import pandas as pd
from skimage import metrics
from skimage.metrics import structural_similarity
from .functions.reconstruction import reconstruct
from .functions.image_processing import crop_image_tensor_by_factor, crop_image_tensor_with_corner


##################################################
## Functions for Calculating Metrics Dataframes ##
##############################  ####################

## Calculate Arbitrary Metric ##
def calculate_metric(batch_A, batch_B, img_metric_function, return_dataframe=False, label='default', crop_factor=1):
    '''
    Function which calculates metric values for two batches of images.
    Returns either the average metric value for the batch or a dataframe of individual image metric values.

    batch_A:                tensor of images to compare [num, chan, height, width]
    batch_B:                tensor of images to compare [num, chan, height, width]
    img_metric_function:    a function which calculates a metric (MSE, SSIM, etc.) from two INDIVIDUAL images
    return_dataframe:       If False, then the average is returned.
                            Otherwise both the average, and a dataframe containing the metric values of the images in the batches, are returned.
    label:                  what to call dataframe, if it is created
    crop_factor:            factor by which to crop both batches of images. 1 = whole image is retained.
    '''

    if crop_factor != 1:
        A = crop_image_tensor_by_factor(batch_A, crop_factor=crop_factor)
        B = crop_image_tensor_by_factor(batch_B, crop_factor=crop_factor)

    length = len(batch_A)
    metric_avg = 0
    metric_list = []

    for i in range(length):
        image_A = batch_A[i:i+1,:,:,:] # Using i:i+1 instead of just i preserves the dimensionality of the array
        image_B = batch_B[i:i+1,:,:,:]

        metric_value = img_metric_function(image_A, image_B)
        metric_avg += metric_value/length
        if return_dataframe==True:
            metric_list.append(metric_value)

    metric_frame = pd.DataFrame({label : metric_list})

    if return_dataframe==False:
        return metric_avg
    else:
        return metric_frame, metric_avg


def update_tune_dataframe(tune_dataframe, tune_dataframe_path, model, config, mean_CNN_MSE, mean_CNN_SSIM, mean_CNN_CUSTOM):
    '''
    Function to update the tune_dataframe for each trial run that makes it partway through the tuning process.

    tune_dataframe      a dataframe that stores model and IQA metric information for a particular trial
    model               model being trained (in tuning)
    config              configuration dictionary
    mean_CNN_MSE        mean MSE for the CNN
    mean_CNN_SSIM       mean SSIM for the CNN
    mean_CNN_CUSTOM     mean custom metric for the CNN

    '''
    # Extract values from config dictionary
    SI_dropout =        config['SI_dropout']
    SI_exp_kernel =     config['SI_exp_kernel']
    SI_gen_fill =       config['SI_gen_fill']
    SI_gen_hidden_dim = config['SI_gen_hidden_dim']
    SI_gen_neck =       config['SI_gen_neck']
    SI_layer_norm =     config['SI_layer_norm']
    SI_normalize =      config['SI_normalize']
    SI_pad_mode =       config['SI_pad_mode']
    batch_size =        config['batch_size']
    gen_lr =            config['gen_lr']

    # Calculate number of trainable weights in CNN
    num_params = sum(map(torch.numel, model.parameters()))

    # Concatenate Dataframe
    add_frame = pd.DataFrame({'SI_dropout': SI_dropout, 'SI_exp_kernel': SI_exp_kernel, 'SI_gen_fill': SI_gen_fill, 'SI_gen_hidden_dim': SI_gen_hidden_dim,
                            'SI_gen_neck': SI_gen_neck, 'SI_layer_norm': SI_layer_norm, 'SI_normalize': SI_normalize, 'SI_pad_mode': SI_pad_mode, 'batch_size': batch_size,
                            'gen_lr': gen_lr, 'num_params': num_params, 'mean_CNN_MSE': mean_CNN_MSE, 'mean_CNN_SSIM': mean_CNN_SSIM, 'mean_CNN_CUSTOM': mean_CNN_CUSTOM}, index=[0])

    tune_dataframe = pd.concat([tune_dataframe, add_frame], axis=0)

    # Save Dataframe to File
    tune_dataframe.to_csv(tune_dataframe_path, index=False)

    return tune_dataframe


def reconstruct_images_and_update_test_dataframe(sino_tensor, image_size, CNN_output, ground_image, test_dataframe, config, compute_MLEM=False):
    '''
    Function which: A) performs reconstructions (FBP and possibly ML-EM)
                    B) constructs a dataframe of metric values (MSE & SSIM) for these reconstructions, and also for the CNN output, with respect to the ground truth image.
                    C) concatenates this with the test dataframe passed to this function
                    D) returns the concatenated dataframe, mean metric values, and reconstructions

    sino_tensor:    sinogram tensor of shape [num, chan, height, width]
    image_size:     image_size
    CNN_output:     CNN reconstructions
    ground_image:   ground truth images
    test_dataframe: dataframe to append metric values to
    config:         general config dictionary
    '''

    # Construct Outputs #
    FBP_output = reconstruct(sino_tensor, config, image_size=image_size, recon_type='FBP')
    if compute_MLEM==True:
        MLEM_output = reconstruct(sino_tensor, config, image_size=image_size, recon_type='MLEM')
    else: # If not looking at ML-EM, don't waste time computing the MLEM images, which can take awhile.
        MLEM_output = FBP_output

    # Dataframes: build dataframes for every reconstruction technique/metric combination #
    batch_CNN_MSE,  mean_CNN_MSE   = calculate_metric(ground_image, CNN_output, MSE,  return_dataframe=True, label='MSE (Network)')
    batch_CNN_SSIM,  mean_CNN_SSIM = calculate_metric(ground_image, CNN_output, SSIM, return_dataframe=True, label='SSIM (Network)')
    batch_FBP_MSE,  mean_FBP_MSE   = calculate_metric(ground_image, FBP_output, MSE,  return_dataframe=True, label='MSE (FBP)')
    batch_FBP_SSIM,  mean_FBP_SSIM = calculate_metric(ground_image, FBP_output, SSIM, return_dataframe=True, label='SSIM (FBP)')
    batch_MLEM_MSE, mean_MLEM_MSE  = calculate_metric(ground_image, MLEM_output, MSE, return_dataframe=True, label='MSE (ML-EM)')
    batch_MLEM_SSIM, mean_MLEM_SSIM= calculate_metric(ground_image, MLEM_output, SSIM,return_dataframe=True, label='SSIM (ML-EM)')

    # Concatenate batch dataframes and larger running test dataframe
    add_frame = pd.concat([batch_CNN_MSE, batch_FBP_MSE, batch_MLEM_MSE, batch_CNN_SSIM, batch_FBP_SSIM, batch_MLEM_SSIM], axis=1)
    test_dataframe = pd.concat([test_dataframe, add_frame], axis=0)

    # Return a whole lot of stuff
    return test_dataframe, mean_CNN_MSE, mean_CNN_SSIM, mean_FBP_MSE, mean_FBP_SSIM, mean_MLEM_MSE, mean_MLEM_SSIM, FBP_output, MLEM_output

######################
## Metric Functions ##
######################

## Metrics which take only single images as inputs ##
## ----------------------------------------------- ##
def SSIM(image_A, image_B, win_size=-1):
    '''
    Function to return the SSIM for two 2D images.

    image_A:        pytorch tensor for a single image
    image_B:        pytorch tensor for a single image
    win_size:       window size to use when computing the SSIM. This must be an odd number. If =-1, the full size of the image is used (or full size-1 so it's odd).
    '''

    if win_size == -1:   # The default shape of the window size is the same size as the image.
        x = image_A.shape[2]
        win_size = (x if x % 2 == 1 else x-1) # Guarantees the window size is odd.

    image_A_npy = image_A.detach().squeeze().cpu().numpy()
    image_B_npy = image_B.detach().squeeze().cpu().numpy()

    max_value = max([np.amax(image_A_npy, axis=(0,1)), np.amax(image_B_npy, axis=(0,1))])   # Find maximum among the images
    min_value = min([np.amin(image_A_npy, axis=(0,1)), np.amin(image_B_npy, axis=(0,1))])   # Find minimum among the images
    data_range = max_value-min_value

    SSIM_image = structural_similarity(image_A_npy, image_B_npy, data_range=data_range, gaussian_weights=False, use_sample_covariance=False, win_size=win_size)

    return SSIM_image

## Metrics which take either batches or images as inputs ##
## ----------------------------------------------------- ##
def MSE(image_A, image_B):
    '''
    Function to return the mean square error for two 2D images (or two batches of images).

    image_A:        pytorch tensor for a single image
    image_B:        pytorch tensor for a single image
    '''
    image_A_npy = image_A.detach().squeeze().cpu().numpy()
    image_B_npy = image_B.detach().squeeze().cpu().numpy()

    return torch.mean((image_A-image_B)**2).item()

def NMSE(image_A, image_B):
    '''
    Function to return the normalized mean square error for two 2D images (or two batches of images).

    image_A:        pytorch tensor for a single image (reference image)
    image_B:        pytorch tensor for a single image
    '''
    image_A_npy = image_A.detach().squeeze().cpu().numpy()
    image_B_npy = image_B.detach().squeeze().cpu().numpy()

    return (torch.mean((image_A-image_B)**2)/torch.mean(image_A**2)).item()

def MAE(image_A, image_B):
    '''
    Function to return the mean absolute error for two 2D images (or two batches of images).

    image_A:        pytorch tensor for a single image
    image_B:        pytorch tensor for a single image
    '''
    image_A_npy = image_A.detach().squeeze().cpu().numpy()
    image_B_npy = image_B.detach().squeeze().cpu().numpy()

    return torch.mean(torch.abs(image_A-image_B)).item()

def calculate_moments(batch_A, batch_B, window_size = 10, stride=10, dataframe=False):
    '''
    Function to return the three statistical moment scores for two image tensors.
    '''
    ## Nested Functions ##

    def compare_moments(win_A, win_B, moment):
        def compute_moment(win, moment, axis=1):
            mean_value = np.mean(win, axis=axis)
            if moment == 1:
                return mean_value
            else:
                mean_array = np.array([mean_value] * win.shape[1]).T  # The square brackets in win.shape[1] mean the value is repeated spatially
                moment = np.mean((win - mean_array)**moment, axis=1)
                return moment

        batch_size = win_A.shape[0]


        reshape_A = (torch.reshape(win_A, (batch_size, -1))).detach().cpu().numpy()
        reshape_B = (torch.reshape(win_B, (batch_size, -1))).detach().cpu().numpy()

        moment_A = compute_moment(reshape_A, moment=moment)
        moment_B = compute_moment(reshape_B, moment=moment)
        moment_score = np.mean(np.absolute(moment_A-moment_B)/(np.absolute(moment_A)+0.1))

        '''
        print('===============================')
        print('MOMENT: ', moment)
        print('moment_A shape: ', moment_A.shape)
        print('moment_A mean: ', np.mean(moment_A))
        print('moment_B shape: ', moment_B.shape)
        print('moment_B mean: ', np.mean(moment_B))
        print('moment_score, |moment_A-moment_B|/(moment_A+0.1) : ', moment_score)
        '''
        return moment_score

    ## Code ##
    image_size = batch_A.shape[2]

    num_windows = int((image_size)/stride) # Maximum number of windows occurs when: stride = window_size.
    while (num_windows-1)*stride + window_size > image_size: # Solve for the number of windows (crops)
        num_windows += -1

    moment_1_running_score = 0
    moment_2_running_score = 0
    moment_3_running_score = 0

    for i in range(0, num_windows):
        for j in range(0, num_windows):
            corner = (i*stride, j*stride)

            win_A = crop_image_tensor_with_corner(batch_A, window_size, corner)
            win_B = crop_image_tensor_with_corner(batch_B, window_size, corner)

            moment_1_score = compare_moments(win_A, win_B, moment=1)
            moment_2_score = compare_moments(win_A, win_B, moment=2)
            moment_3_score = compare_moments(win_A, win_B, moment=3)

            moment_1_running_score += moment_1_score
            moment_2_running_score += moment_2_score
            moment_3_running_score += moment_3_score

    return moment_1_running_score, moment_2_running_score, moment_3_running_score

def LDM(batch_A, batch_B):
    '''
    Calculate the local distributions metric (LDM) for two batches of images
    '''

    score_1, score_2, score_3 = calculate_moments(batch_A, batch_B, window_size=5, stride=5)

    score_1 = score_1*1
    score_2 = score_2*1
    score_3 = score_3*1

    '''
    print('Scores')
    print('====================')
    print(score_1)
    print(score_2)
    print(score_3)
    '''

    return score_1+score_2+score_3

def custom_metric(batch_A, batch_B):
    return 0
    #return MSE(batch_A, batch_B)



###############################################
## Average or a Batch Metrics: Good for GANs ##
###############################################

# Range #
def range_metric(real, fake):
    '''
    Computes a simple metric which penalizes "fake" images in a batch for having a range different than the "real" images in a batch.
    Only a single metric number is returned.
    '''
    range_real = torch.max(real).item()-torch.min(real).item()
    range_fake = torch.max(fake).item()-torch.min(fake).item()

    return abs(range_real-range_fake)/(range_real+.1)

# Average #
def avg_metric(real, fake):
    '''
    Computes a simple metric which penalizes "fake" images in a batch for having an average value different than the "real" images in a batch.
    Only a single metric number is returned.
    '''
    avg_metric = abs((torch.mean(real).item()-torch.mean(fake).item())/(torch.mean(real)+.1).item())
    return avg_metric

# Pixel Variation #
def pixel_dist_metric(real, fake):
    '''
    Computes a metric which penalizes "fake" images for having a pixel distance different than the "real" images.

    real: real image tensor
    fake: fake image tensor
    '''
    def pixel_dist(image_tensor):
        '''
        Function for computing the pixel distance (standard deviation from mean) for a batch of images.
        For simplicity, it only looks at the 0th channel.
        '''
        array = image_tensor[:,0,:,:].detach().cpu().numpy().squeeze()
        sd = np.std(array, axis=0)
        avg=np.mean(sd)
        return(avg)

    pix_dist_fake = pixel_dist(fake)
    pix_dist_real = pixel_dist(real)

    return abs((pix_dist_real-pix_dist_fake)/(pix_dist_real+.1)) # The +0.1 in the denominators guarantees we don't divide by zero

###################
## Old Functions ##
###################


def LDM_OLD(real, fake, crop_size = 10, stride=10):
    '''
    Function to return the local distributions metric for two images.

    image_A:        pytorch tensor for a single image
    image_B:        pytorch tensor for a single image
    '''
    image_size = real.shape[2]

    i_max = int((image_size)/stride) # Maximum number of windows occurs when the stride equals the crop_size
    while (i_max-1)*stride + crop_size > image_size: # If stride < crop_size, we need fewer need to solve for the number of crops
        i_max += -1

    def crop_image_tensor_with_corner(A, corner=(0,0), crop_size=1):
        '''
        Function which returns a small, cropped version of an image.

        A           a batch of images with dimensions: (num_images, channel, height, width)
        corner      upper-left corner of window
        crop_size   size of croppiong window
        '''
        x_min = corner[1]
        x_max = corner[1]+crop_size
        y_min = corner[0]
        y_max = corner[0]+crop_size
        return A[:,:, y_min:y_max , x_min:x_max ]

    running_dist_score = 0
    running_avg_score = 0

    for i in range(0, i_max):
        for j in range(0, j_max):
            corner = (i*crop_size, j*crop_size)
            win_real = crop_image_tensor_with_corner(real, corner, crop_size)
            win_fake = crop_image_tensor_with_corner(fake, corner, crop_size)

            #range_score = range_metric(win_real, win_fake)
            avg_score = avg_metric(win_real, win_fake)
            pixel_dist_score = pixel_dist_metric(win_real, win_fake)

            running_dist_score += pixel_dist_score
            running_avg_score += avg_score

    combined_score = running_dist_score + running_avg_score

    return combined_score