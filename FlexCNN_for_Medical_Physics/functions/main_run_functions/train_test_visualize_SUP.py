def train_Supervisory_Sym(config, offset=0, num_examples=-1, sample_division=1):
    '''
    Function to train or test a network with supervisory loss only. Also used for visualizing data in the dataset.
    '''
    print('Dataset offset:', offset)
    print('Dataset num_examples:', num_examples)
    print('Dataset sample_division:', sample_division)

    ############################
    ### Initialize Variables ###
    ############################

    ## Grab some values and assign to local variables ##
    sup_criterion=config['sup_criterion']
    scale=config['SI_scale'] if train_SI==True else config['IS_scale']

    ## If Tuning ##
    if run_mode=='tune':
        batch_size=config['batch_size']   #config['batch_size']=tune.choice([32, 64, 128, 256, 512, 1024])
        batch_mult = 512/batch_size if tune_even_reporting == True else 1
        display_step = tune_iter_per_report*batch_mult # Larger batch size --> fewer training iterations per report to RayTune

        if tune_restore==False:
            tune_dataframe = pd.DataFrame({'SI_dropout': [], 'SI_exp_kernel': [], 'SI_gen_fill': [], 'SI_gen_hidden_dim': [], 'SI_gen_neck': [], 'SI_layer_norm': [], 'SI_normalize': [],'SI_pad_mode': [], 'batch_size': [], 'gen_lr': [], 'num_params': [], 'mean_CNN_MSE': [], 'mean_CNN_SSIM': [], 'mean_CNN_CUSTOM': []})
            tune_dataframe.to_csv(tune_dataframe_path, index=False)
        else:
            tune_dataframe = pd.read_csv(tune_dataframe_path)

    ## If Training ##
    elif run_mode=='train':
        batch_size=config['batch_size']
        display_step = train_display_step

    ## If Testing ##
    elif run_mode=='test':
        batch_size = config['batch_size'] = test_batch_size  # If we don't override the batch size in the config dictionary, the same batch size will be used as was used to train the network. Therefore, we override it.
        display_step = test_display_step
        test_dataframe = pd.DataFrame({'MSE (Network)' : [],  'MSE (FBP)': [],  'MSE (ML-EM)': [],'SSIM (Network)' : [], 'SSIM (FBP)': [], 'SSIM (ML-EM)': []})

    ## If Visualizeing ##
    elif run_mode=='visualize':
        batch_size = config['batch_size'] = visualize_batch_size # If we don't override the batch size in the config dictionary, the same batch size will be used as was used to train the network. Therefore, we override it.
        display_step = 1

    ## Define running variables ##
    mean_gen_loss = 0; mean_CNN_SSIM = 0 ; mean_CNN_MSE = 0 ; mean_CNN_CUSTOM = 0; report_num = 1  # First report to RayTune is report_num=1.

    ###########################
    ### Instantiate Classes ###
    ###########################

    # Generator #
    if train_SI==True:
        gen =  Generator(config=config, gen_SI=True,  input_size=sino_size, input_channels=sino_channels,  output_channels=image_channels).to(device)
    else:
        gen =  Generator(config=config, gen_SI=False, input_size=image_size, input_channels=image_channels, output_channels=sino_channels ).to(device)

    # Optimizer #
    gen_opt = torch.optim.Adam(gen.parameters(), lr=config['gen_lr'], betas=(config['gen_b1'], config['gen_b2']))

    # Dataloader #
    dataloader = DataLoader(
        NpArrayDataSet(image_path=image_path, sino_path=sino_path, config=config, image_size=image_size, image_channels=image_channels,
                       sino_size=sino_size, sino_channels=sino_channels, augment=augment, offset=offset, num_examples=num_examples, sample_division=sample_division),
        batch_size=batch_size,
        shuffle=shuffle
    )


    ##############################
    ### Set Initial Conditions ###
    ##############################

    ## If loading checkpoint (training, testing or visualizing). For tuning, load_state=False (always). ##
    if load_state==True:
        checkpoint = torch.load(checkpoint_path) # checkpoint is a dictionary of dictionaries
        gen.load_state_dict(checkpoint['gen_state_dict'])
        gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])

        # If testing or visualizing, start from the beginning #
        if run_mode=='test' or run_mode=='visualize':
            gen.eval()  # Evaluation mode-->don't run backprojection
            start_epoch=0; end_epoch=1; batch_step = 0

        # If training, pick up where we left off #
        elif run_mode=='train':
            start_epoch = checkpoint['epoch'] # Note: if interrupted, this epoch may be trained more than once
            end_epoch = train_epochs
            batch_step = checkpoint['batch_step'] # Note: because training is done with shuffling (unless you alter it), stopping partway through a training epoch will result in the network seeing some training examples more than once, and some not at all.

    ## If starting from scratch ##
    else:
        gen = gen.apply(weights_init)
        start_epoch=0 ; batch_step = 0
        end_epoch=num_epochs  # =1000 for tuning (Ray Tune terminates before you hit this), =train_epochs for training, =1 for testing or visualizing

    ## Initialize timestamps to keep track of calculation times ##
    time_init_full = time.time()   # This is reset at the display time so that the full step time is displayed (see below).
    time_init_loader = time.time()  # This is reset at the display time, but also reset at the end of the inner "for loop", so that only displays the data loading time.

    ########################
    ### Loop over Epochs ###
    ########################

    ### Loop over Epochs ###
    for epoch in range(start_epoch, end_epoch):

        #########################
        ### Loop Over Batches ###
        #########################
        for sino_ground, sino_ground_scaled, image_ground, image_ground_scaled in iter(dataloader): # Dataloader returns the batches. Loop over batches within epochs.

            # Show times #
            current_time = display_times('loader time', time_init_loader, show_times) # current_time is a dummy variable that isn't used in this loop
            time_init_full = display_times('FULL STEP TIME', time_init_full, show_times) # This step resets time_init_full after displaying the time so this displays the full time to fun the loop over a batch.

            # Assign inputs and targets #
            if train_SI==True:
                target=image_ground_scaled
                input=sino_ground_scaled
            else:
                target=sino_ground_scaled
                input=image_ground_scaled

            #######################
            ## Calculate Outputs ##
            #######################

            ## If Tuning or Training, train one step ##
            if run_mode=='tune' or run_mode=='train':
                time_init_train = time.time() # Initialize timestamp for training duration

                gen_opt.zero_grad()
                CNN_output = gen(input)

                if run_mode=='train' and torch.sum(CNN_output[1,0,:]) < 0: # Let's you know if the network starts outputing predominantly negative values.
                    print('PIXEL VALUES SUM TO A NEGATIVE NUMBER. IF THIS CONTINUES FOR AWHILE, YOU MAY NEED TO RESTART')

                # Update gradients
                gen_loss = sup_criterion(CNN_output, target)
                gen_loss.backward()
                gen_opt.step()
                # Keep track of the average generator loss
                mean_gen_loss += gen_loss.item() / display_step

                current_time = display_times('training time', time_init_train, show_times)

            ## If Testing or Vizualizing, calculate output only ##
            else:
                CNN_output=gen(input).detach()

            # Increment batch_step
            batch_step += 1

            ####################################
            ### Run-Type Specific Operations ###
            ####################################
            time_init_metrics=time.time()


            ## If Tuning or Training ##
            # We only calculate the mean value of the metrics, but not dataframes or reconstructions. Mean values are used to calculate the optimization metrics #
            if (run_mode == 'tune') or (run_mode=='train'):

                mean_CNN_SSIM += calculate_metric(target, CNN_output, SSIM)/ display_step # The SSIM function can only take single images as inputs, not batches, so we use a wrapper function and pass batches to it.
                mean_CNN_MSE +=  calculate_metric(target, CNN_output, MSE) / display_step # The MSE function can take either single images or batches. We use the wrapper for consistency.

                time_init_custom=time.time()
                # Custom metrics can take a long time to calculate, so we don't use a wrapper (which would loop through individual images in calculations.)
                mean_CNN_CUSTOM += custom_metric(target, CNN_output) / display_step
                current_time = display_times('Custom metric time', time_init_custom, show_times)

            ## If Testing ##
            # We reconstruct images and we calculate metric dataframes #
            if run_mode == 'test':
                test_dataframe, mean_CNN_MSE, mean_CNN_SSIM, mean_FBP_MSE, mean_FBP_SSIM, mean_MLEM_MSE, mean_MLEM_SSIM, FBP_output, MLEM_output =  reconstruct_images_and_update_test_dataframe(
                    input, image_size, CNN_output, image_ground_scaled, test_dataframe, config)

            ## If Visualizing ##
            if run_mode=='visualize':
                # We calculate reconstructions but not metric values. #
                FBP_output =  reconstruct(input, config, image_size=image_size, recon_type='FBP')
                MLEM_output = reconstruct(input, config, image_size=image_size, recon_type='MLEM')


            # Show metric calculation time #
            current_time = display_times('metrics time', time_init_metrics, show_times)

            ######################################
            ### VISUALIZATION / REPORTING CODE ###
            ######################################

            if batch_step % display_step == 0: # and (batch_step > 0 or run_mode != 'tune'):

                time_init_visualization=time.time()

                example_num = batch_step*batch_size

                ## If Tuning ##
                if run_mode=='tune':

                    session.report({'MSE':mean_CNN_MSE, 'SSIM':mean_CNN_SSIM, 'CUSTOM':mean_CNN_CUSTOM, 'example_number': example_num, 'batch_step':batch_step, 'epoch':epoch}) # Report to RayTune multiple times per trial

                    if int(tune_dataframe_fraction*tune_max_t) == report_num: # We only update tune_dataframe once per trial
                        tune_dataframe = update_tune_dataframe(tune_dataframe, tune_dataframe_path, gen, config, mean_CNN_MSE, mean_CNN_SSIM, mean_CNN_CUSTOM)

                    report_num +=1

                ## If Training ##
                if run_mode == 'train':
                    # Display Batch Metrics #
                    print('================Training===================')
                    print(f'CURRENT PROGRESS: epoch: {epoch} / batch_step: {batch_step} / image #: {example_num}')
                    print(f'mean_gen_loss:', mean_gen_loss)
                    print(f'mean_CNN_MSE :', mean_CNN_MSE)
                    print(f'mean_CNN_SSIM:', mean_CNN_SSIM)
                    print(f'mean-CNN_CUSTOM', mean_CNN_CUSTOM)
                    print('===========================================')
                    print('Last Batch MSE: ', calculate_metric(target, CNN_output, MSE))
                    print('Last Batch SSIM: ', calculate_metric(target, CNN_output, SSIM))

                    # Display Inputs & Reconstructions#
                    print('Input:')
                    show_single_unmatched_tensor(input[0:9])
                    print('Target/Output:')
                    show_multiple_matched_tensors(target[0:9], CNN_output[0:9])

                ## If Testing ##
                if run_mode == 'test':
                    # Display Batch Metrics #
                    print('==================Testing==================')
                    print(f'mean_CNN_MSE/mean_MLEM_MSE/mean_FBP_MSE : {mean_CNN_MSE}/{mean_MLEM_MSE}/{mean_FBP_MSE}')
                    print(f'mean_CNN_SSIM/mean_MLEM_SSIM/mean_FBP_SSIM: {mean_CNN_SSIM}/{mean_MLEM_SSIM}/{mean_FBP_SSIM}')
                    print('===========================================')

                    # Display Inputs & Reconstructions #
                    print('Input')
                    show_single_unmatched_tensor(input[0:9])
                    print('Target/Output/MLEM/FBP:')
                    show_multiple_matched_tensors(target[0:9], CNN_output[0:9], MLEM_output[0:9], FBP_output[0:9])

                ## If Visualizing ##
                if run_mode == 'visualize':
                    if visualize_batch_size==120:
                        print(f'visualize_offset: {visualize_offset}, Image Number (batch_step*120): {batch_step*120}')
                        show_single_unmatched_tensor(target, grid=True, cmap='inferno', fig_size=1)
                    else:
                        print('Input:')
                        show_single_unmatched_tensor(input[0:visualize_batch_size])
                        print('Target/ML-EM/FBP/Output:')
                        show_multiple_matched_tensors(target[0:visualize_batch_size], MLEM_output[0:visualize_batch_size], FBP_output[0:visualize_batch_size], CNN_output[0:visualize_batch_size])


                # Save State -- This does not occur with every batch used in training so save resources #
                if save_state:
                    print('Saving model!')
                    torch.save({
                        'epoch': epoch,
                        'batch_step': batch_step,
                        'gen_state_dict': gen.state_dict(),
                        'gen_opt_state_dict': gen_opt.state_dict(),
                        }, checkpoint_path)

                # Zero running stats -- occurs once per visualization step #
                mean_gen_loss = 0 ; mean_CNN_SSIM = 0 ; mean_CNN_MSE = 0 ; mean_CNN_CUSTOM=0

                # Show visualization time #
                current_time = display_times('visualization time', time_init_visualization, show_times)


            # Time step to display loader time
            time_init_loader = time.time()


    ############################################
    ### Complete end of Train Function Tasks ###
    ############################################

    # Save Network State (Training) #
    if save_state:
        print('Saving model!')
        path = os.path.join(checkpoint_dirPath, checkpoint_file)
        torch.save({
            'epoch': epoch+1, # If we are saving after an epoch is completed, and we pick up training later, we have to start at the next epoch.
            'batch_step': batch_step,
            'gen_state_dict': gen.state_dict(),  # dictionary of dictionaries!
            'gen_opt_state_dict': gen_opt.state_dict(),
            }, path)

    # If testing, return dataframe #
    if run_mode=='test':
        return test_dataframe