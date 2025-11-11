from FlexCNN_for_Medical_Physics.functions.main_run_functions import run_SUP


def tune(tune_max_t=40, trainable='SUP', grace_period=1):
    '''
    This function is called to tune the "trainable" function, given:

    tune_max_t:     maximum number of time units (in this case, number of reports) per trial.
    grace_period:   minimum number of raytune reports to run before aborting a trial due to poor performance
    '''

    ## What am I tuning for? ##
    if tune_for=='MSE':     # Values for these metric labels are passed to RayTune in the training function: session.report(.)
        optim_metric='MSE'
        min_max='min' # minimise MSE
    elif tune_for=='SSIM':
        optim_metric='SSIM'
        min_max='max' # maximize SSIM
    elif tune_for=='CUSTOM':
        optim_metric='CUSTOM'
        min_max='min'

    print('===================')
    print('tune_max_t:', tune_max_t)
    print('optim_metric:',optim_metric)
    print('min_max:', min_max)
    print('grace_period:', grace_period)
    print('tune_minutes', tune_minutes)  # Set in "User Parameters".
    print('===================')

    ## Reporters ##
    reporter = CLIReporter(
        metric_columns=[optim_metric,'batch_step'])

    reporter1 = JupyterNotebookReporter(
        overwrite=True,                                           # Overwrite subsequent reporter tables in output (so there is no scrolling)
        metric_columns=[optim_metric,'batch_step','example_num'], # Values for both 'batch_step' and 'example_num' are passed to RayTune
        metric=[optim_metric],                                    # Which metric is used to determine best trial?
        #mode=['min'],
        sort_by_metric=True,                                      # Order reporter table by metric
    )

    ## Trial Scheduler and Run Config ##
    if tune_scheduler == 'ASHA':
        scheduler = ASHAScheduler(
            time_attr='training_iteration', # "Time" is measured in training iterations. 'training_iteration' is a RayTune keyword (not passed in session.report(...)).
            max_t=tune_max_t, # (default=40). Maximum time units per trial (units = time_attr). Note: Ray Tune will by default run a maximum of 100 display steps (reports) per trial
            metric=optim_metric, # This is the label in a dictionary passed to RayTune (in session.report(...))
            mode=min_max,
            grace_period=grace_period, # Train for a minumum number of time_attr. Set in Tune() arguments.
            #reduction_factor=2
            )
        run_config=air.RunConfig(       # How to perform the run
            name=tune_exp_name,         # Ray checkpoints saved to this file, relative to tune_storage_dirPath. Set in "User Parameters"
            storage_path=tune_storage_dirPath,     # Tune search directory. Set in "User Parameters"
            progress_reporter=reporter, # Specified above
            failure_config=air.FailureConfig(fail_fast=False), # default = False. Keeps running if there is an error.
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=10,         # Maximum number of checkpoints that are kept per run (for each trial)
                checkpoint_score_attribute=optim_metric,  # Determines which checkpoints are kept on disk.
                checkpoint_score_order=min_max
                )
            )
    else:
        scheduler = FIFOScheduler()     # First in/first out scheduler
        run_config=train.RunConfig(
            stop={'training_iteration': tune_max_t}, # When using the FIFO scheduler, we must explicitly specify the stopping criterian.
            name=tune_exp_name,         # Ray checkpoints saved to this file, relative to tune_storage_dirPath
            storage_path=tune_storage_dirPath,     # Local directory
            progress_reporter=reporter,
            failure_config=air.FailureConfig(fail_fast=False), # default = False
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=10,         # Maximum number of checkpoints that are kept per run.
                checkpoint_score_attribute=optim_metric,  # Determines which checkpoints are kept on disk.
                checkpoint_score_order=min_max)
        )
        '''
        run_config=train.RunConfig(       # How to perform the run
            name=tune_exp_name,              # Ray checkpoints saved to this file, relative to tune_storage_dirPath
            storage_path=tune_storage_dirPath,     # Local directory
            progress_reporter=reporter,
            failure_config=air.FailureConfig(fail_fast=False), # default = False
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=10,         # Maximum number of checkpoints that are kept per run.
                checkpoint_score_attribute=optim_metric,  # Determines which checkpoints are kept on disk.
                checkpoint_score_order=min_max,
                stop={"time_total_s": 5})
            #    stop={"training_iteration": tune_max_t}) # The FIFO scheduler does not have a stopping criterian, so this stops the trial.
            )
        '''

    ## HyperOpt Search Algorithm ##
    search_alg = HyperOptSearch(metric=optim_metric, mode=min_max)  # It's also possible to pass the search space directly to the search algorithm here.
                                                                    # But then the search space needs to be defined in terms of the specific search algorithm methods, rather than letting RayTune translate.

    ## Which trainable do you want to use? ##
    if trainable=='SUP':
        trainable_with_resources = tune.with_resources(run_SUP, {"CPU":num_CPUs,"GPU":num_GPUs}) # train_Supervisory_Sym is a function of the config dictionary, but we don't state that explicitly.
    elif trainable=='GAN':
        trainable_with_resources = tune.with_resources(run_GAN, {"CPU":num_CPUs,"GPU":num_GPUs})
    elif trainable=='CYCLE':
        trainable_with_resources = tune.with_resources(run_CYCLE, {"CPU":num_CPUs,"GPU":num_GPUs})

    ## If starting from scratch ##
    if tune_restore==False:

        # Initialize a blank tuner object
        tuner = tune.Tuner(
                trainable_with_resources,       # The objective function w/ resources
                param_space=config,             # Let RayTune know what parameter space (dictionary) to search over.
                tune_config=tune.TuneConfig(    # How to perform the search
                    num_samples=-1,
                    time_budget_s=tune_minutes*60, # time_budget is in seconds
                    scheduler=scheduler,
                    search_alg=search_alg,
                    ),
                run_config=run_config
                )

    ## If loading from a checkpoint ##
    else:
        # Load the tuner
        tuner = tune.Tuner.restore(
            path=os.path.join(tune_storage_dirPath, tune_exp_name), # Path where previous run is checkpointed
            trainable=trainable_with_resources,
            resume_unfinished = False
            )

    result_grid: ResultGrid = tuner.fit()
