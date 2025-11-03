import torch

def get_supervisory_loss(fake_X, real_X, sup_criterion):
    '''
    Function to calculate the supervisory loss.

    fake_X:         fake image tensor (Terminology from GANs. For supervisory networks, it's arbitrary whether fake_X or real_X are ground truths or reconstructions)
    real_X:         real image tensor
    sup_criterion   loss function. Will be a Pytorch object.
    '''
    #print('Calc supervisory loss')
    sup_loss = sup_criterion(fake_X, real_X)
    return sup_loss

def get_disc_loss(fake_X, real_X, disc_X, adv_criterion):
    '''
    Function to calculate the discriminator loss. Used to train the discriminator.
    '''
    disc_fake_pred = disc_X(fake_X.detach()) # Detach generator from fake batch
    disc_fake_loss = adv_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred)) # Good fakes shoudl yield predictions = 0.
    disc_real_pred = disc_X(real_X)
    disc_real_loss = adv_criterion(disc_real_pred, torch.ones_like(disc_real_pred)) # Good fakes shoudl yield predictions = 1.
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss

def get_gen_adversarial_loss(real_X, gen_XY, disc_Y, adv_criterion):
    '''
    Function to calculate the adversarial loss (for gen_XY) and fake_Y (from real_X).
    '''
    fake_Y = gen_XY(real_X)
    disc_fake_pred = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) # generator is penalized for discriminmator getting it right
    return adversarial_loss, fake_Y

def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    '''
    Function to calculate the cycle-consistency loss (for gen_YX).
    '''
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(cycle_X, real_X)
    return cycle_loss, cycle_X

def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, config):
    '''
    Function to calculate the total generator loss. Used to train the generators.
    '''
    supervisory_criterion = config['sup_criterion']
    cycle_criterion = config['cycle_criterion']
    gen_adversarial_criterion = config['gen_adv_criterion']
    lambda_adv = config['lambda_adv']
    lambda_sup = config['lambda_sup']
    lambda_cycle = config['lambda_cycle']

    # Adversarial Loss
    if lambda_adv != 0: # To save resources, we only run this code if lambda_adv != 0
        adv_loss_AB, fake_B = get_gen_adversarial_loss(real_A, gen_AB, disc_B, gen_adversarial_criterion)
        adv_loss_BA, fake_A = get_gen_adversarial_loss(real_B, gen_BA, disc_A, gen_adversarial_criterion)
        adv_loss = adv_loss_AB+adv_loss_BA
    else: # Even if we don't compute adversarial losses, we still need fake_A and fake_B for later code
        fake_A = gen_BA(real_B)
        fake_B = gen_AB(real_A)

    # Supervisory Loss
    if lambda_sup != 0: # To save resources, we only run this code if lambda_sup != 0
        sup_loss_AB = get_supervisory_loss(fake_B, real_B, supervisory_criterion)
        sup_loss_BA = get_supervisory_loss(fake_A, real_A, supervisory_criterion)
        sup_loss = sup_loss_AB+sup_loss_BA

    # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
    cycle_loss_AB, cycle_B = get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
    cycle_loss_BA, cycle_A = get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
    cycle_loss = cycle_loss_AB+cycle_loss_BA

    # Total Generator Loss
    if lambda_sup == 0:
        gen_loss = lambda_adv*adv_loss+lambda_cycle*cycle_loss
        return gen_loss, adv_loss.item(), 0, cycle_loss.item(), cycle_A, cycle_B
    elif lambda_adv == 0:
        gen_loss = lambda_sup*sup_loss+lambda_cycle*cycle_loss
        return gen_loss, 0, sup_loss.item(), cycle_loss.item(), cycle_A, cycle_B
    else:
        gen_loss = lambda_adv*adv_loss+lambda_sup*sup_loss+lambda_cycle*cycle_loss
        return gen_loss, adv_loss.item(), sup_loss.item(), cycle_loss.item(), cycle_A, cycle_B

### Functons for Assymmetric/Separate (Older) ###
'''
def get_gen_adv_loss(fake_X, disc_X, adv_criterion):
    print('Calc generative adversarial loss')
    disc_fake_pred = disc_X(fake_X)
    adversarial_loss = adv_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) # Called only from get_gen_loss
    return adversarial_loss

def get_sup_loss(fake_X, real_X, sup_criterion):
    print('Calc supervisory loss')
    sup_loss = sup_criterion(fake_X, real_X)
    return sup_loss

def get_cycle_loss(fake_I, gen_IS, low_rez_S, cycle_criterion):
    print('Calc cycle loss')
    cycle_S = gen_IS(fake_I)
    cycle_loss = cycle_criterion(cycle_S, low_rez_S)
'''