# %% # -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from scripts.ModelPlot import Plotting
from utils.utils_process import Config
from utils.utils_process import config_from_json
# %% Specify the dataset -------------------------
config_type = config_from_json()
# %% Load the configuration file -------------------------
cfg = Config(config_type)
cfg.config_type = config_type
print(f"Using device: {cfg.device}")
Plotting = Plotting(cfg)
Plotting.run_plotting()
# %%