# Scripts_fiber

This set of scripts implements UQ on microstructure to stress prediction. The relevant scripts are in scripts_fiber. This package uses the layers for BayesbyBackprop and HMC_torch for Hamiltonian Monte Carlo,respectively.

Use the config_fiber to specify the relevant parameters. The most important ones are:

- mode: determines Frequentist, MCD, BBB, BBB_LRT, HMC
- case : creates subdirectories in the data and trained_models
- patience : number of epochs for termination after the error does not change
- num_samples : number of data points used in the dataset
- batch_size : to be used only in MCD, Frequentist, BBB, BBB_LRT
- n_target_ch : number of target output stress (set to 1 by default)
- nfilters : encoding-decoding
- krnl_size
- num_samples_bayes : number of realizations of the network for the MCD, BBB at UQ prediction time
- num_samples_hmc : number of HMC samples
- L : HMC trajectory length
- step_size : HMC step size
- device: cuda, mps, cpu etc depending on whether you want to parallelize

The scripts implement

- MCD : Monte Carlo Dropout
- BBB : BayesByBackprop
- HMC : Hamiltonian Monte Carlo

- For MCD/BBB run:
  - fiber_Bayesian_train : training
  - fiber_Bayesian_pred : prediction
  - fiber_Bayesian_plot : plotting
- For Frequentist/HMC run:
  - fiber_HMC_train : training
  - fiber_HMC_pred : prediction
  - fiber_HMC_plot : plotting

### General polycrystalline optimization parameters

- num_samples = 700-1000 samples
- epochs = 500-1000
- nfilters = np.array([64, 128, 256, 512, 1024, 2048])
- kernel_size = 3

### BBB

- learning_rate = 0.0005

### HMC

- num_samples_hmc = 50000
- L = 3 # trajectory length
- step_size = 0.00001 for num_samples = 1000

Dont forget that the preloaded models if downloaded should be used in the same device as the user
