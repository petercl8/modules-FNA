import torch

def weights_init(m): # 'm' represents layers in the generator or discriminator.

    #Function for initializing network weights to normal distribution, with mean 0 and s.d. 0.02
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0, 0.02)
        torch.nn.init.constant_(m.bias, 0)