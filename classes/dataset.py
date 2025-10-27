#import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

def NpArrayDataLoader(image_array, sino_array, config, image_size = 90, sino_size=90, image_channels=1, sino_channels=1, augment=False, index=0):
    '''
    Function to load an image and a sinogram. Returns 4 pytorch tensors: the original dataset sinogram and image,
    and scaled and (optionally) normalized sinograms and images.

    image_array:    image numpy array
    sino_array:     sinogram numpy array
    config:         configuration dictionary with hyperparameters
    image_size:     shape to resize image to (for output)
    image_channels: number of channels for output images
    sino_size:      shape to resize sinograms to (for output)
    sino_channels:  number of channels in output sinograms
    augment:        perform data augmentation?
    index:          index of the image/sinogram pair to grab
    '''
    ## Set Normalization Variables ##
    if (train_type=='GAN') or (train_type=='SUP'):
        if train_SI==True:
            SI_normalize=config['SI_normalize']
            SI_scale=config['SI_scale']
            IS_normalize=False     # If the Sinogram-->Image network (SI) is being trained, don't waste time normalizing sinograms
            IS_scale=1             # If the Sinogram-->Image network (SI) is being trained, don't waste time scaling sinograms
        else:
            IS_normalize=config['IS_normalize']
            IS_scale=config['IS_scale']
            SI_normalize=False
            SI_scale=1
    else: # If a cycle-consistent network, normalize & scale everything
        IS_normalize=config['IS_normalize']
        SI_normalize=config['SI_normalize']
        IS_scale=config['IS_scale']
        SI_scale=config['SI_scale']

    ## Data Augmentation Functions ##
    def RandRotate(image_multChannel, sinogram_multChannel):
        '''
        Function for randomly rotating an image and its sinogram. If the image intersects the edge of the FOV, no rotation is applied.

        image_multChannel:    image to rotate. Shape: (C, H, W)
        sinogram_multChannel: sinogram to rotate. Shape: (C, H, W)
        '''

        def IntersectCircularBorder(image):
            '''
            Function for determining whether an image itersects a circular boundary inscribed within the square FOV.
            This function is not currently used.
            '''
            y_max = image.shape[1]
            x_max = image.shape[2]

            r_max = y_max/2.0
            x_center = (x_max-1)/2.0 # the -1 comes from the fact that the coordinates of a pixel start at 0, not 1
            y_center = (y_max-1)/2.0

            margin_sum = 0
            for y in range(0, y_max):
                for x in range(0, x_max):
                    if r_max < ((x-x_center)**2 + (y-y_center)**2)**0.5 :
                        margin_sum += torch.sum(image[:,y,x]).item()

            return_value = True if margin_sum == 0 else False
            return return_value

        def IntersectSquareBorder(image):
            '''
            Function for determining whether the image intersects the edge of the square FOV. If it does not, then the image
            is fully specified by the sinogram and data augmentation can be performed. If the image does
            intersect the edge of the image then some of it may be cropped outside the FOV. In this case,
            augmentation via rotation should not be performed as the rotated image may not be fully described by the sinogram.
            Looks at all channels in the image.
            '''
            max_idx = image.shape[1]-1
            margin_sum = torch.sum(image[:,0,:]).item() + torch.sum(image[:,max_idx,:]).item() \
                        +torch.sum(image[:,:,0]).item() + torch.sum(image[:,:,max_idx]).item()
            return_value = False if margin_sum == 0 else True
            return return_value
       
        if IntersectSquareBorder(image_multChannel) == False:
            bins = sinogram_multChannel.shape[2]
            bins_shifted = np.random.randint(0, bins)
            angle = int(bins_shifted * 180/bins)

            image_multChannel = transforms.functional.rotate(image_multChannel, angle, fill=0) # Rotate image. Fill in unspecified pixels with zeros.
            sinogram_multChannel = torch.roll(sinogram_multChannel, bins_shifted, dims=(2,)) # Cycle (or 'Roll') sinogram by that angle along dimension 2.
            sinogram_multChannel[:,:, 0:bins_shifted] = torch.flip(sinogram_multChannel[:,:,0:bins_shifted], dims=(1,)) # flip the cycled portion of the sinogram vertically

        return image_multChannel, sinogram_multChannel

    def VerticalFlip(image_multChannel, sinogram_multChannel):
        image_multChannel = torch.flip(image_multChannel,dims=(1,)) # Flip image vertically
        sinogram_multChannel = torch.flip(sinogram_multChannel,dims=(1,2)) # Flip sinogram horizontally and vertically
        return image_multChannel, sinogram_multChannel

    def HorizontalFlip(image_multChannel, sinogram_multChannel):
        image_multChannel = torch.flip(image_multChannel, dims=(2,)) # Flip image horizontally
        sinogram_multChannel = torch.flip(sinogram_multChannel, dims=(2,)) # Flip sinogram horizontally
        return image_multChannel, sinogram_multChannel

    ## Select Data, Convert to Pytorch Tensors ##
    image_multChannel = torch.from_numpy(image_array[index,:]) # image_multChannel.shape = (C, X, Y)
    sinogram_multChannel = torch.from_numpy(sino_array[index,:]) # sinogram_multChannel.shape = (C, X, Y)

    ## Run Data Augmentation on Selected Data. ##
    if augment==True:
        image_multChannel, sinogram_multChannel = RandRotate(image_multChannel, sinogram_multChannel)           # Always rotates image by a random angle
        if np.random.choice([True, False]): # Half of the time, flips the image vertically
            image_multChannel, sinogram_multChannel = VerticalFlip(image_multChannel, sinogram_multChannel)
        if np.random.choice([True, False]): # Half of the time, flips the image horizontally
            image_multChannel, sinogram_multChannel = HorizontalFlip(image_multChannel, sinogram_multChannel)

    ## Create A Set of Resized Outputs ##
    sinogram_multChannel_resize = transforms.Resize(size = (sino_size, sino_size), antialias=True)(sinogram_multChannel)
    image_multChannel_resize    = transforms.Resize(size = (image_size, image_size), antialias=True)(image_multChannel)

    ## (Optional) Normalize Resized Outputs Along Channel Dimension ##
    if SI_normalize:
        a = torch.reshape(image_multChannel_resize, (image_channels,-1))
        a = nn.functional.normalize(a, p=1, dim = 1)
        image_multChannel_resize = torch.reshape(a, (image_channels, image_size, image_size))
    if IS_normalize:
        a = torch.reshape(sinogram_multChannel_resize, (sino_channels,-1))                     # Flattens each sinogram. Each channel is normalized.
        a = nn.functional.normalize(a, p=1, dim = 1)                      # Normalizes along dimension 1 (values for each of the 3 channels)
        sinogram_multChannel_resize = torch.reshape(a, (sino_channels, sino_size, sino_size))  # Reshapes sinograms back into squares.

    ## Adjust Output Channels of Resized Outputs ##
    if image_channels==1:
        image_out = image_multChannel_resize                 # For image_channels = 1, the image is just left alone
    else:
        image_out = image_multChannel_resize.repeat(image_channels,1,1)   # This chould be altered to account for RGB images, etc.

    if sino_channels==1:
        sino_out = sinogram_multChannel_resize[0:1,:]        # Selects 1st sinogram channel only. Using 0:1 preserves the channels dimension.
    else:
        sino_out = sinogram_multChannel_resize               # Keeps full sinogram with all channels

    # Returns both original and altered sinograms and images, assigned to CPU or GPU
    return sinogram_multChannel.to(device), IS_scale*sino_out.to(device), image_multChannel.to(device), SI_scale*image_out.to(device)

class NpArrayDataSet(Dataset):
    '''
    Class for loading data from .np files, given file directory strings and set of optional transformations.
    In the dataset used in our first two conference papers, the data repeat every 17500 steps but with different augmentations.
    For the dataset with FORE rebinning, the dataset contains no augmented examples; all augmentation is performed on the fly.
    '''
    def __init__(self, image_path, sino_path, config, image_size = 90, sino_size=90, image_channels=1, sino_channels=1,
                 augment=False, offset=0, num_examples=-1, sample_division=1):
        '''
        image_path:         path to images in data set
        sino_path:          path to sinograms in data set
        config:             configuration dictionary with hyperparameters
        image_size:         shape to resize image to (for output)
        image_channels:     number of channels in images
        sino_size:          shape to resize sinograms to (for output)
        sino_channels:      number of channels in sinograms (for photopeak sinograms, this is 1)
        augment:            Set True to perform on-the-fly augmentation of data set. Set False to not perform augmentation.
        offset:             To begin dataset at beginning of the datafile, set offset=0. To begin on the second image, offset = 1, etc.
        num_examples:       Max number of examples to load into dataset. Set to -1 to load the maximum number from the numpy array.
        sample_division:    set to 1 to use every example, 2 to use every other example, etc. (Ex: if sample_division=2, the dataset will be half the size.)
        '''

        ## Load Data to Arrays ##
        image_array = np.load(image_path, mmap_mode='r')       # We use memmaps to significantly speed up the loading.
        sino_array = np.load(sino_path, mmap_mode='r')

        ## Set Instance Variables ##
        if num_examples==-1:
            self.image_array = image_array[offset:,:]
            self.sino_array = sino_array[offset:,:]
        else:
            self.image_array = image_array[offset : offset + num_examples, :]
            self.sino_array = sino_array[offset : offset + num_examples, :]

        self.config = config
        self.image_size = image_size
        self.sino_size = sino_size
        self.image_channels = image_channels
        self.sino_channels = sino_channels
        self.augment = augment
        self.sample_division = sample_division

    def __len__(self):
        length = int(len(self.image_array)/sample_division)
        return length

    def __getitem__(self, idx):

        idx = idx*self.sample_division

        sino_ground, sino_ground_scaled, image_ground, image_ground_scaled = NpArrayDataLoader(self.image_array, self.sino_array, self.config, self.image_size,
                                                                                self.sino_size, self.image_channels, self.sino_channels,
                                                                                augment=self.augment, index=idx)

        return sino_ground, sino_ground_scaled, image_ground, image_ground_scaled
        # Returns both original, as well as altered, sinograms and images