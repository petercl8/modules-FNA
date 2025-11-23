import torch
import numpy as np
from skimage.metrics import structural_similarity
from .cropping import crop_image_tensor_with_corner

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