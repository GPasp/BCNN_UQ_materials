############### Configuration file for Bayesian Unet fiber scripts###############
data_path = "../data/data_fiber"
random_seed = 1
img_size = 128
# %% ------------------ General NN optimization parameters ------------------
lr_start = 0.005  # optimal for the input output fiber
train_ens = 1
valid_ens = 1
num_workers = 1  # number of workers for GPU implementations
# %% ------------------ Unet architecture parameters ------------------
n_target_ch = 1  # number of channels in the output
# , 256, 512]#, 1024, 2048] # number of filters
nfilters = [n_target_ch, 16, 32, 64, 128]
kernel_size = 3  # kernel size
# %% ------------------ HMC parameters ------------------
normalizing_const = 1.
burn = 0  # GPU: 3000
bnorm_axis = -1
learning_rate = lr_start
store_on_GPU = False
preload = False
num_ts_data = 40
# %% ------------------ Tunable parameters/  ------------------
n_epochs = 2000  # number of epochs
batch_size = 1  # number of batches
num_samples = 1000  # these are the number of data points used in the dataset
patience = 50
num_samples_bayes = 1000  # number of posterior samples
beta_type = 10**(-8)  # 'Blundell', 'Standard', etc. Use float for const value
num_samples_hmc = 1000  # these are the number of samples from the weights posterior
# used for the prediction and plots before all the HMC samples have been completed
num_samples_hmc_pred = 1000
method = 'MCD'  # options: 'Deterministic', 'MCD', 'BBB', 'BBB_LRT', 'HMC'
device = "mps"
step_size = 0.0005
L = 300  # number of steps per sample
tau_out = 10000
tau = 100  # ./100. # 1/50
case = '4'
drop_rate = 0.2
# %% ------------------ MCD parameters ------------------
drop_idx_en = [0, 0, 0, 1, 1]
drop_idx_dec = [0, 0, 0, 0, 0]
# %% ------------------ Bayes by backprop parameters ------------------
priors = {
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}
lr_start_fiber = 0.01  # optimal for the input output fiber
layer_type = 'BBB'
# %% ------------------ Parameter to be monitored ------------------
variable_names = [
    'patience', 'num_samples', 'batch_size', 'n_target_ch', 'nfilters',
    'krnl_size', 'num_samples_bayes', 'num_samples_hmc', 'L', 'step_size', 'case', 'beta_type', 'preload', 'img_size', 'tau', 'tau_out', 'cuda', 'drop_idx_dec', 'drop_idx_en'
]
