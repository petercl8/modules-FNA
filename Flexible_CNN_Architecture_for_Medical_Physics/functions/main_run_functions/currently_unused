'''
Note: It makes no sense to "test" a GAN or use SSIM since there is nothing to compare it to. Therefore, this functionality is left out here.
Also, now that you've defined assigned the checkpoint_dirPath and test_dataframe_dirPath in the "User Parameters cell", you can get rid of the path constructions below.

'''
def train_test_GAN(config, checkpoint_dirPath=None, load_state=False, save_state=False):
    '''
    Note: Arguments are set to False/None to ensure that when RayTune calles train(), states are not saved/loaded
    Note: you may want to use 'model.train()' to put model back into training mode if you put it into eval mode at some point...
    '''
    print('Training GAN only!!')

    ## Grab from Config ##

    batch_size=config['batch_size']
    gen_adv_criterion=config['gen_adv_criterion']
    scale=config['SI_scale'] if train_SI==True else config['IS_scale']

    ## Tensorboard ##
    writer=SummaryWriter(tensorboard_dir)

    # Generators/Discriminators #

    ## These are the original networks, and work great with 71x71 images ##
    #disc = Disc_I_Orig(config=config).to(device)
    #gen =  Gen_SI_Orig(config=config).to(device)

    ## These are the modified networks, for 90x90, and also work great ##
    #disc = Disc_I_Orig_90(config=config).to(device)
    #gen = Gen_SI_Orig_90(config=config).to(device)

    if train_SI==True:
        ## Now let's try a flex generator and Gen_SI_Orig_90 discriminator ##
        disc_adv_criterion=config['SI_disc_adv_criterion']
        disc = Disc_I_90(config=config, input_channels=image_channels).to(device)
        gen =  Gen_90(config=config, gen_SI=True, input_channels=sino_channels, output_channels=image_channels).to(device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=config['gen_lr'], betas=(config['gen_b1'], config['gen_b2'])) #betas are optional inputs
        disc_opt = torch.optim.Adam(disc.parameters(), lr=config['SI_disc_lr'], betas=(config['SI_disc_b1'], config['SI_disc_b2']))
    else:
        disc_adv_criterion=config['IS_disc_adv_criterion']
        disc = Disc_S_90(config=config, input_channels=sino_channels).to(device)
        gen =  Gen_90(config=config, gen_SI=False, input_channels=image_channels, output_channels=sino_channels).to(device)
        gen_opt = torch.optim.Adam(gen.parameters(), lr=config['gen_lr'], betas=(config['gen_b1'], config['gen_b2'])) #betas are optional inputs
        disc_opt = torch.optim.Adam(disc.parameters(), lr=config['IS_disc_lr'], betas=(config['IS_disc_b1'], config['IS_disc_b2']))

    ## Load Data ##
    dataloader = DataLoader(
        NpArrayDataSet(image_path=image_path, sino_path=sino_path, config=config, resize_size=resize_size, image_channels=image_channels, sino_channels=sino_channels, offset=True),
        batch_size=batch_size,
        shuffle=shuffle
    )

    ## Load Checkpoint ##
    if checkpoint_dirPath and load_state:
        # Load dictionary
        checkpoint_path = os.path.join(checkpoint_dirPath, checkpoint_file)
        checkpoint = torch.load(checkpoint_path)
        # Load values from dictionary
        start_epoch = checkpoint['epoch'] #If interrupted, this epoch may be trained more than once
        end_epoch = start_epoch + num_epochs
        batch_step = checkpoint['batch_step']
        gen.load_state_dict(checkpoint['gen_state_dict'])
        gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        disc_opt.load_state_dict(checkpoint['disc_opt_state_dict'])
    else:
        print('Starting from scratch')
        start_epoch=0
        end_epoch=num_epochs
        batch_step = 0
        gen = gen.apply(weights_init)
        disc = disc.apply(weights_init) # Both gen & disc inherit nn.Module functionality (.apply())

    ## Loop Over Epochs ##
    for epoch in range(start_epoch, end_epoch):
        pix_dist_real_array = np.array([]) # Reset every epoch
        mean_gen_loss = 0  # Reset every display step, but I define it here so it's available later
        mean_disc_loss = 0 # Reset every display step
        mean_pix_metric = 0  # Reset every display step
        time_init_full = time.time()

        ## Loop Over Batches ##
        for sino, sino_ground_scaled, image, image_ground_scaled in iter(dataloader): # Dataloader returns the batches. Loop over batches within epochs.

            print(f'FULL step (time): {(time.time()-time_init_full)*1000}')
            time_init_full = time.time()

            if train_SI==True:
                real=image_ground_scaled
                noise=sino_ground_scaled
            else:
                real=sino_ground_scaled
                noise=image_ground_scaled

            #print(f'Real Type: {real.dtype}, Real Shape:  {real.shape}')
            #print(f'Noise Type: {noise.dtype}, Noise Shape:  {noise.shape}')
            #cur_batch_size = len(real)

            ## UPDATE DISCRIMINATOR ##
            disc_opt.zero_grad()                    # Zero gradients before every batch #
            disc_real_pred = disc(real)             # Predictions on Real Images #

            with torch.no_grad(): # We won't be optmizing generator here, so disabling gradients saves on resources
                fake = gen(noise)
            disc_fake_pred = disc(fake.detach())

            a = torch.ones_like(disc_real_pred)

            disc_real_loss = disc_adv_criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_fake_loss = disc_adv_criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True) # retain_graph=True is set so that we can perform gradient calculations using "backward" twice:
                                                  # you need to compute gradients of discriminator in order to obtain gradients of generator, later.
                                                  # Otherwise, for performance reasons, you can't do this.
            disc_opt.step()

            # Keep track of the average discriminator loss
            mean_disc_loss += disc_loss.item() / display_step

            ## UPDATE GENERATOR ##
            gen_opt.zero_grad()
            # Generator adversarial loss
            disc_fake_pred = disc(gen(noise))
            gen_loss = gen_adv_criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            # Update gradients
            gen_loss.backward()
            gen_opt.step()
            # Keep track of the average generator loss
            mean_gen_loss += gen_loss.item() / display_step #gen_loss.item() reduces tensor to scalar. It updates loss per display step

            ## PIXEL DISTANCE METRIC ##
            pix_dist_fake = pixel_dist(fake)
            pix_dist_real = pixel_dist(real)
            pix_dist_real_array = np.append(pix_dist_real_array, pix_dist_real)
            pix_dist_real_avg = np.mean(pix_dist_real_array)
            #pix_dist_real_avg = 0.00029 # determined experimentally
            pix_metric = abs((pix_dist_real_avg-pix_dist_fake)/pix_dist_real_avg)
            mean_pix_metric += pix_metric / display_step

            ## visualization CODE ##
            if batch_step % display_step == 0 and batch_step > 0: # runs if batch_step is a multiple of the display step

                # Calculate Individual Loss Terms #
                loss_balance=abs(mean_gen_loss-mean_disc_loss)
                r_metric= range_metric(real, fake)
                a_metric= avg_metric(real, fake)

                # Metric Loss #
                optim_metric=0.5*loss_balance+mean_pix_metric+a_metric #+r_metric

                ## REPORT AND SAVE STATE ##
                # Report #
                if run_mode=='tune':
                    tune.report(batch_step=batch_step, epoch=epoch,
                                mean_gen_loss=mean_gen_loss, mean_disc_loss=mean_disc_loss, loss_balance=loss_balance,
                                range_metric=r_metric, avg_metric = a_metric, mean_pix_metric=mean_pix_metric, optim_metric=optim_metric
                                )
                else:
                    # Display Stats #
                    print(f'===========================================\nEPOCH: {epoch}, STEP: {batch_step}')

                    print(f'Real Image Batch Min: {torch.min(real)} // Max: {torch.max(real)} // Mean: {torch.mean(real)} // Sum: {torch.sum(real).item()}')
                    print(f'Fake Image Batch Min: {torch.min(fake)} // Max: {torch.max(fake)} // Mean: {torch.mean(fake)} // Sum: {torch.sum(fake).item()}')
                    print(f'mean_gen_loss: {mean_gen_loss} // mean_disc_loss: {mean_disc_loss}')
                    print(f'loss_balance: {loss_balance}')
                    print(f'mean_pixel_metric: {mean_pix_metric}')
                    print(f'range_metric: {r_metric}')
                    print(f'avg_metric: {a_metric}')
                    print(f'optim_metric: {optim_metric}')

                    # visualize Images #
                    print('Reals: ')
                    show_single_unmatched_tensor(real)
                    print('Fakes: ')
                    show_single_unmatched_tensor(fake)

                    writer.add_scalar('generator loss', mean_gen_loss, batch_step)
                    writer.add_scalar('discriminator loss', mean_disc_loss, batch_step)
                    writer.add_scalar('loss balance', loss_balance, batch_step)
                    writer.add_scalar('pixel distance loss', mean_pix_metric, batch_step)
                    #writer.add_image("real", make_grid(real_image_tensor[:25], nrow=5, normalize=True)) # [:num_images]=[0:num_images]
                    #writer.add_image("fake", make_grid(fake_image_tensor[:25], nrow=5, normalize=True))
                    writer.flush()

                # Save State #
                if checkpoint_dirPath and save_state:
                    path = os.path.join(checkpoint_dirPath, checkpoint_file)
                    torch.save({
                        'epoch': epoch,
                        'batch_step': batch_step,
                        'gen_state_dict': gen.state_dict(),
                        'gen_opt_state_dict': gen_opt.state_dict(),
                        'disc_state_dict': disc.state_dict(),
                        'disc_opt_state_dict': disc_opt.state_dict(),
                        }, path)

                # Zero Stats #
                mean_disc_loss = 0
                mean_gen_loss = 0
                mean_pix_metric = 0

    ## And the end of the epoch loop, we do a final save of the model ##
    if checkpoint_dirPath and save_state:
        path = os.path.join(checkpoint_dirPath, checkpoint_file)
        torch.save({
            'epoch': epoch,
            'batch_step': batch_step,
            'gen_state_dict': gen.state_dict(),
            'gen_opt_state_dict': gen_opt.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'disc_opt_state_dict': disc_opt.state_dict(),
            }, path)