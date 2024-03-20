import os
import numpy as np
import torch
from torch.nn import Conv2d
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import importlib


class Config:
    def __init__(self, config_type):
        # Dynamically import the correct configuration module
        if config_type == 'fiber':
            cfg = importlib.import_module('configs.config_fiber')
        elif config_type == 'polycrystalline':
            cfg = importlib.import_module('configs.config_polycrystalline')
        else:
            raise ValueError("Invalid configuration type")

        # Load variables from the imported module
        self.load_config_variables(cfg)

    def load_config_variables(self, cfg):
        # Iterate through the module's attributes and set them to this class
        for attr in dir(cfg):
            # Skip built-in attributes and methods
            if not attr.startswith("__"):
                setattr(self, attr, getattr(cfg, attr))

def downsample_data_torch(data, target_size=(64, 64)):

    data_tensor = torch.from_numpy(data).float()
    downsampled_tensor = F.adaptive_avg_pool2d(data_tensor, target_size)
    downsampled_array = downsampled_tensor.numpy()

    return downsampled_array

def unflatten_with_layer_info(model, flattened_params):
    if flattened_params.dim() != 1:
        raise ValueError('Expecting a 1d flattened_params')
    params_list = []
    layer_info = []
    i = 0
    for name, val in model.named_parameters():
        length = val.nelement()
        param = flattened_params[i:i + length].view_as(val)
        params_list.append(param)
        layer_info.append((name, val.shape, i, i + length))
        i += length

    return params_list, layer_info

def get_largest_params(flattened_params, num_params=10):
    _, max_indices = torch.topk(flattened_params, num_params)
    return max_indices

def get_largest_conv_params(net, num_params=20):
    conv_params = []
    conv_indices = []
    layer_names = []  # To store the names of layers corresponding to parameters
    idx = 0
    
    # Iterate over all modules and their names in the model
    for module_name, module in net.named_modules():
        # Check if the module is a convolutional layer
        if isinstance(module, Conv2d):
            # Then iterate over parameters of this module
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    # Flatten the parameter and append it to the list
                    flat_param = param.view(-1)
                    conv_params.append(flat_param)
                    # Record the adjusted indices and the layer names
                    conv_indices.extend([idx + i for i in range(flat_param.numel())])
                    layer_names.extend([f"{module_name}.{param_name}"] * flat_param.numel())
                    idx += flat_param.numel()
    
    # Concatenate all the flattened convolutional parameters into a single tensor
    all_conv_params = torch.cat(conv_params)
    # Find the top k values and their indices within this tensor
    _, topk_conv_indices = torch.topk(all_conv_params.abs(), num_params)

    # Map back the indices of the top k values to their original positions and get corresponding layer names
    original_indices = [conv_indices[i] for i in topk_conv_indices.cpu().numpy()]
    original_layer_names = [layer_names[i] for i in topk_conv_indices.cpu().numpy()]

    return original_indices, original_layer_names


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


def comp_tau_list(Net, tau_value, device):
    tau_list = [tau_value for _ in Net.parameters()]
    return torch.tensor(tau_list).to(device)


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_array_to_file(numpy_array, filename):
    file = open(filename, 'a')
    shape = " ".join(map(str, numpy_array.shape))
    np.savetxt(file, numpy_array.flatten(), newline=" ", fmt="%.3f")
    file.write("\n")
    file.close()


def get_random_samples(input_data, output_data, num_samples=5):
    indices = np.random.choice(len(input_data), num_samples, replace=False)
    X_batch = input_data[indices]
    Y_batch = output_data[indices]
    return X_batch, Y_batch


def get_ch_path(path):
    # path = specifications['data_path's]
    # split the string at each underscore
    split_path = path.split('_')
    # take the last part of the split string
    ch_path = '_'.join(split_path[-2:])
    return ch_path


def normalize(Y):
    Nimages = Y.shape[0]
    Y_norm = np.zeros_like(Y)
    for i in range(Nimages):
        for c in range(Y.shape[-1]):  # Iterate over all channels
            Y_min = np.amin(Y[i, :, :, c])
            Y_max = np.amax(Y[i, :, :, c])
            Y_norm[i, :, :, c] = (Y[i, :, :, c] - Y_min) / (Y_max - Y_min)
    return Y_norm


def print_min_max(array, array_name):
    array_min = np.amin(array)
    array_max = np.amax(array)
    print(f"{array_name} min: {array_min}, max: {array_max}")


def load_data(cfg):
    data_path = cfg.data_path
    X_tr = np.load(data_path+"/X_tr.npy")
    Y_tr = np.load(data_path+"/Y_tr.npy")
    X_val = np.load(data_path+"/X_val.npy")
    Y_val = np.load(data_path+"/Y_val.npy")

    # Bring to torch tensor format
    if cfg.config_type == 'polycrystalline':
        X_tr = X_tr.transpose(0, 3, 1, 2)
        Y_tr = Y_tr.transpose(0, 3, 1, 2)
        X_val = X_val.transpose(0, 3, 1, 2)
        Y_val = Y_val.transpose(0, 3, 1, 2)
    elif cfg.config_type == 'fiber':
        X_tr = X_tr.transpose(3, 2, 0, 1)
        Y_tr = Y_tr.transpose(3, 2, 0, 1)
        X_val = X_val.transpose(3, 2, 0, 1)
        Y_val = Y_val.transpose(3, 2, 0, 1)

    X_tr = downsample_data_torch(
        X_tr, target_size=(cfg.img_size, cfg.img_size))
    Y_tr = downsample_data_torch(
        Y_tr, target_size=(cfg.img_size, cfg.img_size))
    X_val = downsample_data_torch(
        X_val, target_size=(cfg.img_size, cfg.img_size))
    Y_val = downsample_data_torch(
        Y_val, target_size=(cfg.img_size, cfg.img_size))

    # Select number of samples from configuration
    X_tr = X_tr[:cfg.num_samples, :cfg.n_target_ch, :, :]
    Y_tr = Y_tr[:cfg.num_samples, :cfg.n_target_ch, :, :]
    X_val = X_val[:, :cfg.n_target_ch, :, :]
    Y_val = Y_val[:, :cfg.n_target_ch, :, :]

    out_channels = Y_tr.shape[1]
    in_channels = X_tr.shape[1]
    num_train_data = X_tr.shape[0]
    num_val_data = X_val.shape[0]
    np.save(data_path + "/X_tr_sam.npy", X_tr)
    np.save(data_path + "/Y_tr_sam.npy", Y_tr)

    for i in range(num_train_data):
        # Convert to tensor, add dimension, convert back to numpy
        X_tr_sample = torch.from_numpy(X_tr[i, :, :, :]).unsqueeze(0).numpy()
        # Convert to tensor, add dimension, convert back to numpy
        Y_tr_sample = torch.from_numpy(Y_tr[i, :, :, :]).unsqueeze(0).numpy()
        np.save(os.path.join(data_path, f"X_tr_{i}.npy"), X_tr_sample)
        np.save(os.path.join(data_path, f"Y_tr_{i}.npy"), Y_tr_sample)

    for i in range(num_val_data):
        X_val_sample = torch.from_numpy(X_val[i, :, :, :]).unsqueeze(
            0).numpy()  # Convert to tensor, add dimension, convert back to numpy
        Y_val_sample = torch.from_numpy(Y_val[i, :, :, :]).unsqueeze(
            0).numpy()  # Convert to tensor, add dimension, convert back to numpy
        np.save(os.path.join(data_path, f"X_val_{i}.npy"), X_val_sample)
        np.save(os.path.join(data_path, f"Y_val_{i}.npy"), Y_val_sample)

    X_tr_tensor = torch.from_numpy(
        X_tr).float().requires_grad_(False).to(cfg.device)
    Y_tr_tensor = torch.from_numpy(
        Y_tr).float().requires_grad_(False).to(cfg.device)

    trainset = TensorDataset(X_tr_tensor, Y_tr_tensor)
    train_loader = DataLoader(trainset, cfg.batch_size,
                              shuffle=False, num_workers=0)

    X_val_tensor = torch.from_numpy(
        X_val).float().requires_grad_(False).to(cfg.device)
    Y_val_tensor = torch.from_numpy(
        Y_val).float().requires_grad_(False).to(cfg.device)
    valset = TensorDataset(X_val_tensor, Y_val_tensor)
    val_loader = DataLoader(valset, cfg.batch_size,
                            shuffle=False, num_workers=0)

    return trainset, train_loader, val_loader, out_channels, in_channels, X_tr, Y_tr, X_val, Y_val, X_tr_tensor, Y_tr_tensor, X_val_tensor, Y_val_tensor, num_train_data, num_val_data



def reverse_transpose(X_tr, Y_tr, X_val, Y_val, cfg):
    # Reverse transpose operation
    X_tr = X_tr.transpose(0, 2, 3, 1)
    Y_tr = Y_tr.transpose(0, 2, 3, 1)
    X_val = X_val.transpose(0, 2, 3, 1)
    Y_val = Y_val.transpose(0, 2, 3, 1)

    # Convert numpy arrays back to torch tensors
    X_tr_tensor = torch.from_numpy(
        X_tr).float().requires_grad_(False).to(device)
    Y_tr_tensor = torch.from_numpy(
        Y_tr).float().requires_grad_(False).to(cfg.device)

    X_val_tensor = torch.from_numpy(
        X_val).float().requires_grad_(False).to(cfg.device)
    Y_val_tensor = torch.from_numpy(
        Y_val).float().requires_grad_(False).to(cfg.device)

    return X_tr, Y_tr, X_val, Y_val, X_tr_tensor, Y_tr_tensor, X_val_tensor, Y_val_tensor


class TensorData:
    def __init__(self, tensors, path):
        self.tensors = tensors
        self.path = path


def load_test_data(cfg):
    data_path = cfg.data_path
    # Load the datasets
    X_ts = np.load(data_path+"/X_ts.npy")
    Y_ts = np.load(data_path+"/Y_ts.npy")
    # Load the datasets
    X_ts = np.load(data_path+"/X_ts.npy")
    Y_ts = np.load(data_path+"/Y_ts.npy")
    # Bring to torch tensor format
    if cfg.config_type == 'polycrystalline':
        X_ts = X_ts.transpose(0, 3, 1, 2)
        Y_ts = Y_ts.transpose(0, 3, 1, 2)
    elif cfg.config_type == 'fiber':
        X_ts = X_ts.transpose(3, 2, 0, 1)
        Y_ts = Y_ts.transpose(3, 2, 0, 1)

    X_ts = downsample_data_torch(
        X_ts, target_size=(cfg.img_size, cfg.img_size))
    Y_ts = downsample_data_torch(
        Y_ts, target_size=(cfg.img_size, cfg.img_size))

    # Select number of samples from configuration
    X_ts = X_ts[:cfg.num_ts_data, :cfg.n_target_ch, :, :]
    Y_ts = Y_ts[:cfg.num_ts_data, :cfg.n_target_ch, :, :]

    names_ts = []
    out_channels = Y_ts.shape[1]
    in_channels = X_ts.shape[1]

    X_ts_tensor = torch.from_numpy(
        X_ts).float().requires_grad_(False).to(cfg.device)
    Y_ts_tensor = torch.from_numpy(
        Y_ts).float().requires_grad_(False).to(cfg.device)

    test_set = TensorDataset(X_ts_tensor, Y_ts_tensor)
    test_loader = DataLoader(test_set, cfg.num_ts_data, shuffle=False, num_workers=0)

    return test_set, test_loader, out_channels, X_ts, Y_ts, names_ts


def load_from_tempfile(temp_filename):
    return np.load(temp_filename + '.npy')


def construct_det_path(cfg):
    if cfg.config_type == 'fiber':
        dir_name = 'trained_models_fiber'
    elif cfg.config_type == 'polycrystalline':
        dir_name = 'trained_models_polycrystalline_2D'
    # Construct directory and checkpoint paths
    train_path = os.path.abspath(os.path.join(
        '..', dir_name, 'case_' + cfg.case))
    ckpt_name = os.path.join(train_path, 'Det_unet.pt')
    return ckpt_name


def construct_paths(cfg):
    filepath = get_filepath(cfg.method)
    if cfg.config_type == 'fiber':
        dir_name = 'trained_models_fiber'
    elif cfg.config_type == 'polycrystalline':
        dir_name = 'trained_models_polycrystalline_2D'
    # Construct directory and checkpoint paths
    train_path = os.path.abspath(os.path.join(
        '..', dir_name, 'case_' + cfg.case))
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if cfg.method in ['MCD', 'BBB', 'BBB_LRT', 'Deterministic']:
        ckpt_name = os.path.join(train_path, filepath + '.pt')
    elif cfg.method in ['HMC']:
        ckpt_name = os.path.join(train_path, filepath + '.pkl')
    loss_filename = os.path.join(train_path, filepath+'_loss_acc.npz')
    return ckpt_name, loss_filename, train_path


def construct_fig_paths(cfg, filename):
    if cfg.config_type == 'fiber':
        dir_name = 'figures_fiber'
    elif cfg.config_type == 'polycrystalline':
        dir_name = 'figures_polycrystalline_2D'

    # Construct directory path for the figures
    dir_path = os.path.abspath(os.path.join(
        '..', dir_name, 'case_' + cfg.case, cfg.method))

    # Create the directory if it does not exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Construct full figure path (directory path + filename)
    fig_path = os.path.join(dir_path, filename)

    return fig_path, dir_path

def construct_dir_path(cfg,save_name):    
    if cfg.config_type == 'polycrystalline':
        save_path = os.path.join('../figures_polycrystalline_2D/', save_name)
    elif cfg.config_type == 'fiber':
        save_path = os.path.join('../figures_fiber/', save_name)    
    return save_path    

def save_tests(plt, cfg, filepath, sample_label='sample_all'):
    # Use construct_fig_paths to get the correct figure path
    if cfg.method == 'MCD':
        # Convert drop_rate to string and remove the dot
        drop_rate_str = str(cfg.drop_rate).replace('.', '')
        fig_filename = f'{sample_label}_{filepath}_{drop_rate_str}.pdf'
    else:
        fig_filename = f'{sample_label}_{filepath}.pdf'

    fig_path, dir_path = construct_fig_paths(cfg, fig_filename)
    plt.savefig(fig_path, bbox_inches='tight', dpi=100)


def save_test_individual(plt, cfg, filepath, sample_id):
    # Use construct_fig_paths to get the correct figure path
    if cfg.method == 'MCD':
        # Convert drop_rate to string and remove the dot
        drop_rate_str = str(cfg.drop_rate).replace('.', '')
        fig_filename = f'sample_{sample_id}_{filepath}_{drop_rate_str}.pdf'
    else:
        fig_filename = f'sample_{sample_id}_{filepath}.pdf'
    fig_path, _ = construct_fig_paths(cfg, fig_filename)
    plt.savefig(fig_path, bbox_inches='tight', dpi=100)


def get_filepath(method):
    if method == 'MCD':
        filepath = 'MCD_unet'
    elif method == 'BBB':
        filepath = 'BBB_unet'
    elif method == 'BBB_LRT':
        filepath = 'BBB_LRT_unet'
    elif method == 'HMC':
        filepath = 'HMC_unet'
    elif method == 'Deterministic':
        filepath = 'Det_unet'
    return filepath


def write_config(cfg, filepath):
    if cfg.config_type == 'fiber':
        dir_name = 'trained_models_fiber'
    elif cfg.config_type == 'polycrystalline':
        dir_name = 'trained_models_polycrystalline_2D'
    dir_path = os.path.abspath(os.path.join(
        '..', dir_name, 'case_' + cfg.case))
    config_filename = 'config_' + filepath + '.py'
    config_file_path = os.path.join(dir_path, config_filename)
    # Write selected configuration variables to the file
    with open(config_file_path, 'w') as file:
        for key in cfg.variable_names:
            value = getattr(cfg, key, None)
            file.write(f"{key} = {repr(value)}\n")
    return config_file_path

import json

def config_from_json():
    config_file_path = 'config.json'
    with open(config_file_path, 'r') as file:
        config_data = json.load(file)
    return config_data['config_type']