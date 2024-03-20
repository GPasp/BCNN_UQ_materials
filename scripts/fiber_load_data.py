#!/usr/bin/env python
# coding: utf-8
#%%
"""
This script loads the training, validation and testing data from the data_fiber folder 
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits import axes_grid1
import scipy.io
import scipy
from skimage.transform import resize
import sys
sys.path.append("..")
from utils.utils_process import Config
from utils.utils_process import config_from_json
# %% Specify the dataset -------------------------
config_type = config_from_json()
# %% Load the configuration file -------------------------
cfg = Config(config_type)
cfg.config_type = config_type
#%%
def downsample_4d_array(array, target_size=(64, 64)):
    _, _, channels, num_samples = array.shape
    downsampled_array = np.empty((target_size[0], target_size[1], channels, num_samples))
    for i in range(num_samples):
        for j in range(channels):
            downsampled_array[:, :, j, i] = resize(array[:, :, j, i], target_size, anti_aliasing=True)
    
    return downsampled_array

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def normalize_images(X, Y, channels, Nimages):
    # print(f"Processing image {i}...")
    X_norm = np.zeros_like(X)
    Y_norm = np.zeros_like(Y)
    for i in range(Nimages):
        X_min = np.amin(X[:, :, 0, i])
        X_max = np.amax(X[:, :, 0, i])
        X_norm[:, :, 0, i] = (X[:, :, 0, i] - X_min) / (X_max - X_min)
        for c in channels:
            Y_min = np.amin(Y[:, :, c - 1, i])
            Y_max = np.amax(Y[:, :, c - 1, i])
            Y_norm[:, :, c - 1, i] = (Y[:, :, c - 1, i] -
                                  Y_min) / (Y_max - Y_min)
    return  X_norm, Y_norm

#%%
# load the images
inp_folder = '../data_fiber/Input/Input_20Fiber_grid128_2000cases_generalized.mat'
out_folder = '../data_fiber/Output/Output_20Fiber_grid128_2000cases_UnitAxialAndShear.mat'

#%%
# target_height = cfg.img_size
# target_width = cfg.img_size
channels = [1, 2, 3, 4, 5, 6, 7, 8]
# target_depth = 8
Nimages = 2000
img_filenames = [f"img_{i+1}" for i in range(Nimages)]
X = scipy.io.loadmat(inp_folder)
X = X['input_data_4D']
Y = scipy.io.loadmat(out_folder)
Y = Y['output_data_4D']
#!!!!
X = downsample_4d_array(X,target_size=(cfg.img_size,cfg.img_size))
Y = downsample_4d_array(Y,target_size=(cfg.img_size,cfg.img_size))
#%% Optional image normalization
# X_norm, Y_norm = normalize_images(X, Y, channels, Nimages)
#%% Optional plotting of the input-output data
i = 20
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams.update({"font.size": 8})
fig, ax = plt.subplots(1, 9, figsize=(18, 18))
fig.subplots_adjust(wspace=0.8)
ax[0].imshow(X[:, :, 0, i])
ax[0].set_title(img_filenames[i] + "_microstucture")
add_colorbar(ax[0].images[0])
for c in channels:
    ax[c].imshow(Y[:, :, c - 1, i])
    add_colorbar(ax[c].images[0])
    ax[c].set_title(img_filenames[i] + "_stress_map_" + str(c))
# fig.tight_layout(pad=0)
save_path = "../figures_fiber/example.png"
plt.savefig(save_path, dpi=100, bbox_inches="tight")
#%%
# Split into training and validation sets
ix = np.arange(Nimages)
Nsamples = cfg.num_samples
# ix_tr, ix_val_ts = train_test_split(ix, train_size=0.8, random_state=0)
# ix_val, ix_ts = train_test_split(ix_val_ts, train_size=0.2,random_state=0)
# create 200 indices for the first 200 training images
ix_tr = np.arange(Nsamples)
# create the next 40 indices for the testing images
ix_ts = np.arange(Nsamples, Nsamples + 40)
# create the next 40 indices for the validation images
ix_val = np.arange(Nsamples + 40, Nsamples+140)
# sanity check, no overlap between train, validation and test sets
assert len(np.intersect1d(ix_tr, ix_val)) == 0
assert len(np.intersect1d(ix_tr, ix_ts)) == 0
assert len(np.intersect1d(ix_val, ix_ts)) == 0
X_tr = X[:,:,:,ix_tr]; Y_tr = Y[:,:,:,ix_tr]
X_val = X[:,:,:,ix_val]; Y_val = Y[:,:,:,ix_val]
X_ts = X[:,:,:,ix_ts]; Y_ts = Y[:,:,:,ix_ts]
#%%
# Create an array of names named wells for the the number of images
# wells = np.array([f"img_{i+1}" for i in range(Nimages)])
# Save the training and validation sample indices of img_filenames
fnames_tr = np.array(img_filenames)[ix_tr].tolist()
fnames_val = np.array(img_filenames)[ix_val].tolist()
fnames_ts = np.array(img_filenames)[ix_ts].tolist()
fname_split = (
    ["train"] * len(fnames_tr)
    + ["validation"] * len(fnames_val)
    + ["test"] * len(fnames_ts))
df = pd.DataFrame({"name": fnames_tr + fnames_val +
                  fnames_ts, "split": fname_split})
df.to_csv("../data_fiber/train_val_test_split.csv", index=False)
#%%
# Create an array of names named wells for the the number of images
# wells = np.array([f"img_{i+1}" for i in range(Nimages)])
# Save the training and validation sample indices
fnames_tr = np.array(img_filenames)[ix_tr].tolist()
fnames_val = np.array(img_filenames)[ix_val].tolist()
fnames_ts = np.array(img_filenames)[ix_ts].tolist()
fname_split = (
   ["train"] * len(fnames_tr)
   + ["validation"] * len(fnames_val)
   + ["test"] * len(fnames_ts)
)
df = pd.DataFrame({"img": fnames_tr + fnames_val + fnames_ts, "split": fname_split})
df.to_csv("../data_fiber/training_validation_test_splits.csv", index=False)
#%%
# Save to disk
np.save("../data_fiber/X_tr.npy", X_tr)
np.save("../data_fiber/X_val.npy", X_val)
np.save("../data_fiber/X_ts.npy", X_ts)
np.save("../data_fiber/Y_tr.npy", Y_tr)
np.save("../data_fiber/Y_val.npy", Y_val)
np.save("../data_fiber/Y_ts.npy", Y_ts)
# %%
