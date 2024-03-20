import sys
sys.path.append("../")
from utils.utils_process import construct_paths, load_test_data
import os
import pickle
import torch
from torch.nn import functional as F
import numpy as np
from ModelTrain import Training
import pickle
import copy
import os
import torch
import numpy as np
import pickle
from architectures.u_net import BayesByBackprop, Deterministic
from architectures.u_net.layers.BBBConv import BBBConv2d
import glob


class BayesianModelPredict:
    def __init__(self, cfg):
        self.device = cfg.device
        self.cfg = cfg
        training_manager = Training(cfg)
        _, self.test_loader, self.out_channels, self.X_ts, self.Y_ts, _ = load_test_data(cfg)
        self.X_ts_tensor = torch.tensor(self.X_ts, dtype=torch.float32)
        self.Y_ts_tensor = torch.tensor(self.Y_ts, dtype=torch.float32)
        self.train_path = construct_paths(cfg)[-1]
        self.net = training_manager.select_network(
            cfg.method, cfg.nfilters, cfg.kernel_size, cfg.layer_type, cfg.drop_rate).to(cfg.device)
        if self.cfg.method in ['BBB', 'BBB_LRT', 'MCD', 'Deterministic']:
            # load the Deterministic model for MCD since all the keys match. In this way there is no nee for retraining
            if self.cfg.method == 'MCD':
                cfg2 = copy.deepcopy(self.cfg)
                cfg2.method = 'Deterministic'
                ckpt_name, _, _ = construct_paths(cfg2)
            else:
                ckpt_name, _, _ = construct_paths(self.cfg)
            self.net.load_state_dict(torch.load(ckpt_name, map_location=self.device))
            self.net.train()  # Set the network to training so you have to dropout and BBB sampling 

    def load_output_from_path(self, output_path):
        with open(output_path, 'rb') as f:
            return pickle.load(f)

    def compute_and_save_predictions(self):
        _, test_loader, _, _, _, _ = load_test_data(self.cfg)
        _, _, train_path = construct_paths(self.cfg)
        n_samples, output_arrrays, mse_history = self.compute_output()
        mean_output, std_output = self.compute_mean_and_std(output_arrrays)
        self.save_results(mean_output, std_output, train_path, mse_history)
        
    def compute_output(self):
        if self.cfg.method in ['BBB', 'BBB_LRT', 'MCD', 'Deterministic']:
            n_samples = self.cfg.num_samples_bayes if self.cfg.method in ['BBB', 'BBB_LRT', 'MCD'] else 2
            outputs = []
            with torch.no_grad():
                for i in range(n_samples):
                    for X_batch, _ in self.test_loader:
                        X_batch = X_batch.to(self.device)
                        # from torchsummary import summary
                        # X_batch = X_batch.to('cpu')   
                        # summary(self.net.to('cpu'), X_batch.shape[1:])
                        X_batch_output = self.net(X_batch)
                        outputs.append(X_batch_output.detach().cpu().numpy())  # Convert to numpy array and append
            outputs_array = np.stack(outputs, axis=0)
            return n_samples, outputs_array, _
        elif self.cfg.method == 'HMC':
            from HMC_torch.hamiltorch import util 
            n_samples = self.cfg.num_samples_hmc_pred
            outputs = []
            mse_history = np.zeros((n_samples,self.Y_ts.shape[0],))
            for i in range(n_samples):
                model_path = os.path.join(self.train_path, f"HMC_unet_{i}.pkl")
                with open(model_path, 'rb') as f:
                    ret_params = pickle.load(f)
                fmodel = util.make_functional(self.net)
                params_unflattened = util.unflatten(self.net, ret_params[0])
                output = fmodel(self.X_ts_tensor, params=params_unflattened)
                se = ((output - self.Y_ts_tensor)**2) 
                se2 = se.detach().cpu().numpy()
                mse = np.mean(se2, axis=(2, 3), keepdims=True)
                mse = np.squeeze(mse, axis=(1, 2, 3))
                mse_history[i,:] = mse.T
                outputs.append(output.detach().cpu())
            outputs_array = np.stack(outputs, axis=0)
            return n_samples, outputs_array, mse_history                        
        
    def compute_mean_and_std(self, outputs):
        sum_of_predictions = None
        sum_of_predictions_squared = None
        n_samples = 0

        for i in range(outputs.shape[0]):
            # Initialize accumulators
            if sum_of_predictions is None:
                sum_of_predictions = np.zeros_like(outputs[i,:,:,:,:])
                sum_of_predictions_squared = np.zeros_like(outputs[i,:,:,:,:])

            # Accumulate predictions
            sum_of_predictions += outputs[i]
            sum_of_predictions_squared += outputs[i] ** 2
            n_samples = n_samples + 1

        # Compute mean and standard deviation
        mean_prediction = sum_of_predictions / n_samples    
        variance_prediction = (sum_of_predictions_squared / n_samples) - (mean_prediction ** 2) # Estimate only the epistemic uncertainty
        # variance_prediction = self.cfg.tau_out + (sum_of_predictions_squared / n_samples) - (mean_prediction ** 2)
        std_prediction = np.sqrt(variance_prediction)

        return mean_prediction, std_prediction

        # # Calculate the mean and standard deviation along the first axis (samples)
        # mean_output = np.mean(outputs, axis=0)
        # std_output = np.std(outputs, axis=0)
        
        # return mean_output, std_output

    def save_results(self, mean_output, std_output, train_path, mse_history):
        suffix = f'_{str(self.cfg.drop_rate).replace(".", "")}' if self.cfg.method == 'MCD' else ''
        mean_filename = os.path.join(train_path, f'{self.cfg.method}{suffix}_mean.pkl')
        std_filename = os.path.join(train_path, f'{self.cfg.method}{suffix}_std.pkl')
        mse_history_filename = os.path.join(train_path, f'{self.cfg.method}{suffix}_mse_history.pkl')
        with open(mean_filename, 'wb') as f:
            pickle.dump(mean_output, f)
        with open(std_filename, 'wb') as f:
            pickle.dump(std_output, f)
        if mse_history is not None:
            with open(mse_history_filename, 'wb') as f:
                pickle.dump(mse_history, f)

class SampleNetworks:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = 'cpu'
        self.BBB_net = BayesByBackprop(cfg.nfilters, cfg.kernel_size, cfg.layer_type).to(self.device)
        self.HMC_net = Deterministic(cfg.nfilters, cfg.kernel_size).to(self.device)
        self.train_path = construct_paths(cfg)[-1]
        self.hmc_files_pattern = os.path.join(self.train_path, 'HMC_unet_*.pkl')
        self.W_mu, self.W_rho, self.bayesian_layer_name, self.layer_number = self._extract_conv_layer_BBB()
        
    def _extract_conv_layer_BBB(self):
        # Dictionary to keep track of layers and whether we've seen W_mu and W_rho
        layer_tracker = {}
        for name, param in self.BBB_net.named_parameters():
            # Split the parameter name into layer name and parameter type
            layer_name, param_type = name.rsplit('.', 1)
            # Initialize the layer entry if not already present
            if layer_name not in layer_tracker:
                layer_tracker[layer_name] = {'W_mu': None, 'W_rho': None}
            # Store the parameter object if it's of type W_mu or W_rho
            if param_type in ['W_mu', 'W_rho']:
                layer_tracker[layer_name][param_type] = param
        # Now filter layers that have both W_mu and W_rho
        bayesian_layers = [layer for layer, params in layer_tracker.items() if params['W_mu'] is not None and params['W_rho'] is not None]
        # Check if there is at least two such layers and return the second one's parameters
        if len(bayesian_layers) >= 2:
            second_layer_name = bayesian_layers[1]  # Get the second layer's name
        for index, (name, _) in enumerate(self.BBB_net.named_children()):
            if name == second_layer_name:
                layer_number = index
            # Return the parameters of the second Bayesian layer
        return layer_tracker[second_layer_name]['W_mu'], layer_tracker[second_layer_name]['W_rho'], second_layer_name, layer_number
        
    def _sample_weights_BBB(self):
        num_samples = self.cfg.num_samples_hmc # sample for the same number that we have generated the HMC samples 
        sampled_bbb_weights = []
        W_sigma = torch.log1p(torch.exp(self.W_rho))
        for i in range(num_samples):
            W_eps = torch.randn_like(self.W_mu).to(self.device)
            weight = self.W_mu + W_eps * W_sigma
            sampled_bbb_weights.append(weight.cpu().detach().numpy().flatten())
        
        sampled_bbb_weights_array = np.array(sampled_bbb_weights)
        save_path = os.path.join(self.train_path, 'BBB_weight_samples.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(sampled_bbb_weights_array, f)
    
    def _extract_conv_layer_HMC(self, model, unflattened_params, layer_name):
        param_dict = dict(model.named_parameters())
        layer_weights = None
        # Iterate over the model's named parameters
        for name, param in model.named_parameters():
            if name.startswith(layer_name) and name.endswith('.weight'):
                # Return the weights for the layer if the name matches
                return param.data  # O
        return layer_weights


    def _unflatten(self ,model, flattened_params):
        if flattened_params.dim() != 1:
            raise ValueError('Expecting a 1d flattened_params')
        params_list = []
        i = 0
        for val in list(model.parameters()):
            length = val.nelement()
            param = flattened_params[i:i+length].view_as(val)
            params_list.append(param)
            i += length
        i = 0    
        for name, param in model.named_parameters():
            layer_name, param_type = name.rsplit('.', 1)
            if layer_name == self.bayesian_layer_name:
                if param_type == 'weight':
                    layer_weights = params_list[i]
            i = i + 1
        return layer_weights    
    
    def _collect_weights_HMC(self):
        # Assumes unflatten function is correctly defined to match the structure of HMC_net
        hmc_weights = []
        for file_path in glob.glob(self.hmc_files_pattern): 
            with open(file_path, 'rb') as file:
                hmc_params = pickle.load(file)
            layer_weights = self._unflatten(self.HMC_net, hmc_params[0])  # Assuming unflatten works correctly
            # layer_weights = unflattened_params[self.layer_number]
            hmc_weights.append(layer_weights.cpu().detach().numpy().flatten()[np.newaxis, :])
        if hmc_weights == []:
            hmc_weights_array = []    
        else:
            hmc_weights_array = np.concatenate(hmc_weights, axis=0)  # Shape will be [N, num_parameters]
        save_path = os.path.join(self.train_path, 'HMC_weight_samples.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(hmc_weights_array, f)

        return hmc_weights

