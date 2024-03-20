import tempfile
import os
import numpy as np
# import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
from utils_process import downsample_data_torch
# import pandas as pd
import pickle
sys.path.append(".")
# %%

CONFIG_TYPE = os.environ.get('CONFIG_TYPE')
if CONFIG_TYPE == 'polycrystalline':
    import configs.config_polycrystalline as cfg
elif CONFIG_TYPE == 'fiber':
    import configs.config_fiber as cfg
print("CONFIG_TYPE:", CONFIG_TYPE)

if 'cfg' in globals():
    print("Config module loaded:", cfg.__name__)
else:
    print("No config module has been loaded.")
device = cfg.device


def get_ch_path(path):
    # path = specifications['data_path's]
    # split the string at each underscore
    split_path = path.split('_')
    # take the last part of the split string
    ch_path = '_'.join(split_path[-2:])
    return ch_path


def load_test_data(data_path):
    # Load the datasets
    X_ts = np.load(data_path+"/X_ts.npy")
    Y_ts = np.load(data_path+"/Y_ts.npy")
    # Load the datasets
    X_ts = np.load(data_path+"/X_ts.npy")
    Y_ts = np.load(data_path+"/Y_ts.npy")
    # Bring to torch tensor format
    if CONFIG_TYPE == 'polycrystalline':
        X_ts = X_ts.transpose(0, 3, 1, 2)
        Y_ts = Y_ts.transpose(0, 3, 1, 2)
        X_ts = downsample_data_torch(
            X_ts, target_size=(cfg.img_size, cfg.img_size))
        Y_ts = downsample_data_torch(
            Y_ts, target_size=(cfg.img_size, cfg.img_size))
    elif CONFIG_TYPE == 'fiber':
        X_ts = X_ts.transpose(3, 2, 0, 1)
        Y_ts = Y_ts.transpose(3, 2, 0, 1)

    # Select number of samples from configuration
    X_ts = X_ts[:cfg.num_ts_data, :cfg.n_target_ch, :, :]
    Y_ts = Y_ts[:cfg.num_ts_data, :cfg.n_target_ch, :, :]
    # #!!!!!!!!!!!!!!!!
    # Y_ts = X_ts[:,:1,:,:]
    #!!!!!!!!!!!!!!!!
    Ntest = len(X_ts)
    # Load datasete names
    # df = pd.read_csv(data_path+"/train_val_test_split.csv")
    # names_ts = df[df['split']=='test']['name'].tolist()
    names_ts = []
    out_channels = Y_ts.shape[1]
    in_channels = X_ts.shape[1]

    X_ts_tensor = torch.from_numpy(
        X_ts).float().requires_grad_(False).to(device)
    Y_ts_tensor = torch.from_numpy(
        Y_ts).float().requires_grad_(False).to(device)

    test_set = TensorDataset(X_ts_tensor, Y_ts_tensor)
    test_loader = DataLoader(test_set, 1, shuffle=False, num_workers=0)

    return test_set, test_loader, out_channels, X_ts, Y_ts, names_ts


def save_to_tempfile(data):
    temp_filename = tempfile.mktemp()
    np.save(temp_filename, data)
    return temp_filename


def load_from_tempfile(temp_filename):
    return np.load(temp_filename + '.npy')
