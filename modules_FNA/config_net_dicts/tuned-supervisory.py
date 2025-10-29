from torch import nn

## Supervisory
'''
In this module, set the correct hyperparameter dictionary for config_SUP_SI.
This is the dictionary of hyperparameters that determines the form of a the network that will be trained, tested, or visualized (when doing supervised learning, Sinogram-->Image).
You will usually find these hyperparameters by performing tuning and examining the best performing networks in tensorboard.

If training supervisory loss networks only, you don't need to worry about the other dictionaries in this section (GANs, Cycle-Consistent).
You also don't need to worry about "Search Spaces", as this is simply a dictionary of the search space that Ray Tune uses when tuning.
Feel free to look at it though, to see how I set up the search space. The last section (Set Correct Config) is where the configuration dictionary gets assigned.
The dictionary is either a searchable space, if tuning, or a set of fixed hyperparameters, if training, testing, or visualizing the data set.
'''

### Below networks were tuned on whole dataset ###
# 1x90x90, Tuned for MSE - fc6 #
'''
config_SUP_SI={
  "SI_dropout": False,
  "SI_exp_kernel": 4,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": None,
  "SI_gen_hidden_dim": 14,
  "SI_gen_mult": 2.3737518721494038,
  "SI_gen_neck": 5,
  "SI_gen_z_dim": 300,
  "SI_layer_norm": "instance",
  "SI_normalize": True,
  "SI_pad_mode": "zeros",
  "SI_scale": 8100,
  "batch_size": 266,
  "gen_b1": 0.5194977285709309,
  "gen_b2": 0.4955647195661826,
  "gen_lr": 0.0006569034263698925,
  "sup_criterion": nn.MSELoss()
}
'''
'''
# 1x90x90, Tuned for MAE (mean absolute error) - b08 #
config_SUP_SI={
  "SI_dropout": True,
  "SI_exp_kernel": 3,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": nn.Tanh(),
  "SI_gen_hidden_dim": 29,
  "SI_gen_mult": 3.4493572412953926,
  "SI_gen_neck": 5,
  "SI_gen_z_dim": 92,
  "SI_layer_norm": "instance",
  "SI_normalize": True,
  "SI_pad_mode": "zeros",
  "SI_scale": 8100,
  "batch_size": 184,
  "gen_b1": 0.41793988944151467,
  "gen_b2": 0.15133808988276928,
  "gen_lr": 0.0012653525173041019,
  "sup_criterion": nn.L1Loss()
}
'''

# 1x90x90, Tuned for SSIM - 14d #
config_SUP_SI = {
  "SI_dropout": False,
  "SI_exp_kernel": 4,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": nn.Tanh(),
  "SI_gen_hidden_dim": 23,
  "SI_gen_mult": 1.6605902406330195,
  "SI_gen_neck": 5,
  "SI_gen_z_dim": 789,
  "SI_layer_norm": "instance",
  "SI_normalize": True,
  "SI_pad_mode": "zeros",
  "SI_scale": 8100,
  "batch_size": 71,
  "gen_b1": 0.2082092731474774,
  "gen_b2": 0.27147903136187507,
  "gen_lr": 0.0005481469822215635,
  "sup_criterion": nn.MSELoss()
}

'''
# 1x90x90, Tuned for Local Distributions Metric, 5x5 window, stride 2
config_SUP_SI={
  "SI_dropout": True,
  "SI_exp_kernel": 3,
  "SI_gen_fill": 2,
  "SI_gen_final_activ": nn.Sigmoid(),
  "SI_gen_hidden_dim": 18,
  "SI_gen_mult": 2.4691388140182475,
  "SI_gen_neck": 11,
  "SI_gen_z_dim": 444,
  "SI_layer_norm": "instance",
  "SI_normalize": True,
  "SI_pad_mode": "zeros",
  "SI_scale": 8100,
  "batch_size": 33,
  "gen_b1": 0.8199882799898334,
  "gen_b2": 0.1207854128656507,
  "gen_lr": 0.0001095057659925285,
  "sup_criterion": nn.BCEWithLogitsLoss()
}
'''
'''
# 1x90x90, Tuned for Local Distributions Metric, 10x10 window, stride 8 (b5c)
config_SUP_SI={
  "SI_dropout": False,
  "SI_exp_kernel": 4,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": None,
  "SI_gen_hidden_dim": 9,
  "SI_gen_mult": 2.1547197646081444,
  "SI_gen_neck": 5,
  "SI_gen_z_dim": 344,
  "SI_layer_norm": "batch",
  "SI_normalize": False,
  "SI_pad_mode": "zeros",
  "SI_scale": 8100,
  "batch_size": 47,
  "gen_b1": 0.31108788447029295,
  "gen_b2": 0.3445239707919786,
  "gen_lr": 0.0007561178182660596,
  "sup_criterion": nn.L1Loss()
}
'''


### Below networks were tuned on 1/4 of dataset (high MSE or low MSE) ####
'''
# 1x90x90, Tuned for SSIM, highSSIM quartile, - c867539
config_SUP_SI = {
  "SI_dropout": False,
  "SI_exp_kernel": 3,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": nn.Tanh(),
  "SI_gen_hidden_dim": 14,
  "SI_gen_mult": 3.1366081867376066,
  "SI_gen_neck": 5,
  "SI_gen_z_dim": 1235,
  "SI_layer_norm": "instance",
  "SI_normalize": True,
  "SI_pad_mode": "reflect",
  "SI_scale": 8100,
  "batch_size": 512,
  "gen_b1": 0.36092827701745117,
  "gen_b2": 0.2959809747063715,
  "gen_lr": 0.0003914885622973457,
  "sup_criterion": nn.MSELoss()
}
'''
'''
# 1x90x90, Tuned for MSE, lowMSE quartile - d3c
config_SUP_SI = {
  "SI_dropout": False,
  "SI_exp_kernel": 3,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": nn.Tanh(),
  "SI_gen_hidden_dim": 10,
  "SI_gen_mult": 3.5952046080348117,
  "SI_gen_neck": 5,
  "SI_gen_z_dim": 1144,
  "SI_layer_norm": "batch",
  "SI_normalize": True,
  "SI_pad_mode": "zeros",
  "SI_scale": 8100,
  "batch_size": 338,
  "gen_b1": 0.21119520045946658,
  "gen_b2": 0.3219437242478679,
  "gen_lr": 0.0012228287967471555,
  "sup_criterion": nn.L1Loss()
}
'''
'''
# 1x90x90, Tuned for MSE, highMSE quartile - 66e
config_SUP_SI = {
  "SI_dropout": False,
  "SI_exp_kernel": 4,
  "SI_gen_fill": 0,
  "SI_gen_final_activ": nn.Tanh(),
  "SI_gen_hidden_dim": 13,
  "SI_gen_mult": 2.427097790975542,
  "SI_gen_neck": 1,
  "SI_gen_z_dim": 1943,
  "SI_layer_norm": "instance",
  "SI_normalize": True,
  "SI_pad_mode": "zeros",
  "SI_scale": 8100,
  "batch_size": 399,
  "gen_b1": 0.5173104983713961,
  "gen_b2": 0.5269533977675209,
  "gen_lr": 0.00042406256400739315,
  "sup_criterion": nn.MSELoss()
}
'''