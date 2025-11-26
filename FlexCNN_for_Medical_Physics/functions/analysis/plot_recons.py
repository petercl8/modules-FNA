import os
import torch
import numpy as np

from FlexCNN_for_Medical_Physics.classes.dataset import NpArrayDataLoader
from FlexCNN_for_Medical_Physics.classes.generators import Generator


def BuildImageSinoTensors(image_array, sino_array, config, indexes, device):
    """
    Build image and sinogram torch tensors (channel-first) from numpy arrays using selected indexes.

    Parameters
    ----------
    image_array : np.ndarray
        Shape (N, H, W) or (N, C, H, W). Will be interpreted channel-first.
    sino_array : np.ndarray
        Shape (N, H, W) or (N, C, H, W) matching image index dimension.
    config : dict
        Must contain at least: 'train_SI', 'SI_scale', 'IS_scale', 'SI_normalize', 'IS_normalize',
        'image_size', 'sino_size', 'image_channels', 'sino_channels'.
    indexes : Sequence[int]
        List/iterable of integer indices to extract.
    device : torch.device or str
        Device to allocate output tensors and place loaded samples.

    Returns
    -------
    image_tensor : torch.Tensor
        Float32 tensor of shape (len(indexes), C_img, H, W).
    sino_tensor : torch.Tensor
        Float32 tensor of shape (len(indexes), C_sino, H, W).

    Notes
    -----
    - All outputs are torch.float32.
    - Assumes NpArrayDataLoader already performs any resizing/normalization based on config.
    - Index errors will raise naturally if an index is out of bounds.
    """
    first = True
    i = 0
    for idx in indexes:
        # augment=False for deterministic extraction
        sino_ground, sino_ground_scaled, image_ground, image_ground_scaled = NpArrayDataLoader(
            image_array, sino_array, config, augment=False, index=idx, device=device
        )

        if first:
            image_tensor = torch.zeros(
                len(indexes),
                image_ground_scaled.shape[0],
                image_ground_scaled.shape[1],
                image_ground_scaled.shape[2],
                dtype=torch.float32,
                device=device
            )
            sino_tensor = torch.zeros(
                len(indexes),
                sino_ground_scaled.shape[0],
                sino_ground_scaled.shape[1],
                sino_ground_scaled.shape[2],
                dtype=torch.float32,
                device=device
            )
            first = False

        image_tensor[i, :] = image_ground_scaled
        sino_tensor[i, :] = sino_ground_scaled
        i += 1

    return image_tensor, sino_tensor


def CNN_reconstruct(sino_tensor, config, checkpoint_path, device):
    """
    Run a trained CNN generator to reconstruct images from a sinogram tensor.

    Parameters
    ----------
    sino_tensor : torch.Tensor
        Shape (N, C_sino, H, W), float32, already on desired device.
    config : dict
        Must include generator architecture keys and:
        'image_size', 'sino_size', 'sino_channels', 'image_channels',
        normalization/scale keys: 'SI_normalize', 'SI_scale' (if supervisory / SI path).
    checkpoint_path : str
        Full path to a saved checkpoint file containing 'gen_state_dict'.
    device : torch.device or str
        Device on which to place the model before inference.

    Returns
    -------
    recon_tensor : torch.Tensor
        Reconstructed images, shape (N, C_img, H, W), float32, detached.

    Notes
    -----
    - Expects checkpoint to contain 'gen_state_dict'.
    - Inference is performed under torch.no_grad().
    - Any mismatch between checkpoint weights and current config will raise an error.
    """
    gen = Generator(config=config, gen_SI=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen.eval()
    with torch.no_grad():
        return gen(sino_tensor).detach()
    

'''
OLDER VERSION BELOW - TO BE DEPRECATED

## CNN Outputs ##
def CNN_reconstruct(sino_tensor, config, checkpoint_dirPath, checkpoint_fileName):

    #Construct CNN reconstructions of images of a sinogram tensor.
    #Config must contain: sino_size, sino_channels, image_channels.

    gen = Generator(config=config, gen_SI=True).to(device)
    checkpoint_path = os.path.join(checkpoint_dirPath, checkpoint_fileName)
    checkpoint = torch.load(checkpoint_path)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen.eval()
    return gen(sino_tensor).detach()
'''