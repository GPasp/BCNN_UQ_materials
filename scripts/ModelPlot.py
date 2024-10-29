#%%
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
import fnmatch
from utils.utils_process import construct_paths, load_test_data, get_filepath, save_tests, construct_fig_paths, construct_dir_path
from skimage.metrics import structural_similarity as ssim
from matplotlib import gridspec
from ModelPredict import SampleNetworks
import pandas as pd
#%%
rc('text', usetex=True)
rc('font', size=14)
rc('legend', fontsize=14)
rc('text', usetex=True)
rc('font', size=14)
rc('legend', fontsize=14)

class Plotting:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_path = cfg.data_path
        self.method = cfg.method
        self.train_path = os.path.join(construct_paths(cfg)[2])
        self.filepath = get_filepath(cfg.method)
        _, _, _, self.X_ts, self.Y_ts, self.names_ts = load_test_data(cfg)
        self.mean_pred, self.std_pred = self.load_mean_and_std(self)
        self.ae = np.abs(self.Y_ts - self.mean_pred)

    def load_test_data(self):
        _, _, _, X_ts, Y_ts, names_ts = load_test_data(self.data_path)
        return X_ts, Y_ts, names_ts
    
    @staticmethod
    def load_mean_and_std(self):
        mean_filename = os.path.join(self.train_path, self.method + '_mean.pkl')
        std_filename = os.path.join(self.train_path, self.method + '_std.pkl')
        if self.method == 'MCD':
            drop_rate_str = str(self.cfg.drop_rate).replace('.', '')
            mean_filename = os.path.join(self.train_path, self.method + f'_{drop_rate_str}_mean.pkl')
            std_filename = os.path.join(self.train_path, self.method + f'_{drop_rate_str}_std.pkl')
        if os.path.exists(mean_filename) and os.path.exists(mean_filename):
            with open(mean_filename, 'rb') as f:
                mean_pred = pickle.load(f)
            with open(std_filename, 'rb') as f:
                std_pred = pickle.load(f)
        else:
            mean_pred = None
            std_pred = None
            
        return mean_pred, std_pred
    
    def _plot_predictions(self, Nsamp=40, channel=0):
        ae = np.abs(self.Y_ts - self.mean_pred)
        if self.std_pred is not None and len(self.std_pred) > 0 and self.cfg.method != 'Deterministic':
            n_columns = 5
        else:
            n_columns = 4
        if self.cfg.config_type == 'polycrystalline':
            colormap = 'plasma'
            colormap2 = 'bwr'
        elif self.cfg.config_type == 'fiber':
            colormap = 'viridis'
            colormap2 = 'viridis'
            
        fig, axes = plt.subplots(Nsamp, n_columns, figsize=(4 * n_columns, 20 * 4),squeeze = False)
        fig.subplots_adjust(wspace=0.2, hspace=-0.9)
        for i in range(Nsamp):
            vmin_stress = 0.8*np.min(self.Y_ts[i-1, channel])
            vmax_stress = 1.2*np.max(self.Y_ts[i-1, channel])
            
            im = axes[i-1, 0].imshow(self.X_ts[i-1, channel, : , :], cmap=colormap2)
            axes[i-1, 0].set_xticks([])
            axes[i-1, 0].set_yticks([])
            axes[i-1, 0].set_title(r'$\rm{Microstructure}$', fontsize=14)
            
            if self.cfg.config_type == 'polycrystalline':
                # Create a mappable object with the same colormap
                sm = plt.cm.ScalarMappable(cmap=colormap2, norm=plt.Normalize(vmin=vmin_stress, vmax=vmax_stress))
                sm.set_array([])  # Only needed for matplotlib versions < 3.1
                cbar = fig.colorbar(sm, ax=axes[i-1, 0],fraction=0.046, pad=0.04)
                ticks = np.linspace(vmin_stress, vmax_stress, 5)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])

            im = axes[i-1, 1].imshow(self.Y_ts[i-1, channel, : , :], vmin=vmin_stress, vmax=vmax_stress, cmap=colormap)
            axes[i-1, 1].set_xticks([])
            axes[i-1, 1].set_yticks([])
            axes[i-1, 1].set_title(r'$\rm{Target\ stress}$', {'fontsize': 14})
            fig.colorbar(im, ax=axes[i-1, 1], fraction=0.046, pad=0.04)

            im = axes[i-1, 2].imshow(self.mean_pred[i-1, channel, : , :], vmin=vmin_stress, vmax=vmax_stress, cmap=colormap)
            axes[i-1, 2].set_xticks([])
            axes[i-1, 2].set_yticks([])
            if self.cfg.method == 'Deterministic':
                axes[i-1, 2].set_title(r'$\rm{Predicted \ stress}$', {'fontsize': 14})
            else:
                axes[i-1, 2].set_title(r'$\rm{Predicted \ mean \ stress}$', {'fontsize': 14})
            fig.colorbar(im, ax=axes[i-1, 2], fraction=0.046, pad=0.04)

            if self.std_pred is not None and len(self.std_pred) > 0 and self.cfg.method != 'Deterministic':
                im = axes[i-1, 3].imshow(self.std_pred[i-1, channel, : , :], cmap='hot')
                axes[i-1, 3].set_xticks([])
                axes[i-1, 3].set_yticks([])
                axes[i-1, 3].set_title(r'$\rm{Predicted \ stress \ \sigma}$', {'fontsize': 14})
                fig.colorbar(im, ax=axes[i-1, 3], fraction=0.046, pad=0.04)

                im = axes[i-1, 4].imshow(ae[i-1, channel], cmap='hot')
                axes[i-1, 4].set_xticks([])
                axes[i-1, 4].set_yticks([])
                axes[i-1, 4].set_title(r'$\rm{Absolute \ error}$', {'fontsize': 14})
                fig.colorbar(im, ax=axes[i-1, 4], fraction=0.046, pad=0.04)
            else:
                im = axes[i-1, 3].imshow(ae[i-1, channel, : , :], cmap='hot')
                axes[i-1, 3].set_xticks([])
                axes[i-1, 3].set_yticks([])
                axes[i-1, 3].set_title(r'$\rm{Absolute \ error}$', {'fontsize': 14})
                fig.colorbar(im, ax=axes[i-1, 3], fraction=0.046, pad=0.04)
        save_tests(plt, self.cfg, self.filepath,'sample_all')
    
    def _plot_predictions_individual(self, Nsamp=10, channel=0):
        ae = np.abs(self.Y_ts - self.mean_pred)
        colormap, colormap2 = ('plasma', 'bwr') if self.cfg.config_type == 'polycrystalline' else ('viridis', 'viridis')

        # Determine the number of columns based on the presence of Y_ts_std
        if self.std_pred is not None and len(self.std_pred) > 0 and self.cfg.method != 'Deterministic':
            n_columns = 5
        else:
            n_columns = 4     
            
        for i in range(Nsamp):
            relative_errors = np.abs(self.mean_pred[i] - self.Y_ts[i]) / np.abs(self.Y_ts[i])    
            max_relative_error = np.max(relative_errors) * 100  # in percentage
            # Calculate overall MAE for the current sample
            absolute_errors = np.abs(self.mean_pred[i] - self.Y_ts[i])
            overall_mae = np.mean(absolute_errors)
            # print(f"Sample {i+1}: Method: {self.cfg.method}, Maximum relative error: {max_relative_error:.2f}%, Overall MAE: {overall_mae:.4f}")
            fig, axes = plt.subplots(1, n_columns, figsize=(4 * n_columns, 4), squeeze=True)
            fig.subplots_adjust(wspace=0.2, hspace=-0.9)
            vmin = 0.8*np.min(self.Y_ts[i, channel])
            vmax = 1.2*np.max(self.Y_ts[i, channel])

            # Absolute Error plot
            if self.cfg.config_type == 'fiber':
                vmin_err = 0
                vmax_err = 2
            elif self.cfg.config_type == 'polycrystalline':
                vmin_err = 0
                vmax_err = 0.3

            # Microstructure plot
            im = axes[0].imshow(self.X_ts[i, channel, :, :], cmap=colormap2)
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            axes[0].set_title(r'$\rm{Microstructure}$', fontsize=14)
            cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
            cbar.ax.set_visible(False)  # Hide the colorbar

            if self.cfg.config_type == 'polycrystalline':
                # Create a mappable object with the same colormap
                sm = plt.cm.ScalarMappable(cmap=colormap2, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=axes[0], fraction=0.046, pad=0.04)
                ticks = np.linspace(vmin, vmax, 5)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])

            # True stress plot
            im = axes[1].imshow(self.Y_ts[i, channel, :, :], vmax=vmax, cmap=colormap)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            axes[1].set_title(r'$\rm{Target\ stress}$', fontsize=14)
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            # Predicted Mean stress plot
            im = axes[2].imshow(self.mean_pred[i, channel, :, :], vmin=vmin, vmax=vmax, cmap=colormap)
            axes[2].set_xticks([])
            axes[2].set_yticks([])
            if self.cfg.method == 'Deterministic':
                axes[2].set_title(r'$\rm{Predicted \ stress}$', fontsize=14)
            else:
                axes[2].set_title(r'$\rm{Predicted \ mean \ stress}$', fontsize=14)
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

            if self.std_pred is not None and len(self.std_pred) > 0 and self.cfg.method != 'Deterministic':
                # Predicted stress \sigma plot
                # vmin_var = 0.6*np.min(self.std_pred[i, channel])
                # vmax_var = 1.5*np.max(self.std_pred[i, channel])
                if self.cfg.config_type == 'fiber':
                    vmin_var = 0
                    vmax_var = 0.8
                elif self.cfg.config_type == 'polycrystalline':
                    vmin_var = 0.01
                    vmax_var = 0.1
                
                im = axes[3].imshow(self.std_pred[i, channel, :, :], cmap='hot',vmin = vmin_var,vmax = vmax_var)
                axes[3].set_xticks([])
                axes[3].set_yticks([])
                axes[3].set_title(r'$\rm{Predicted \ stress \ \sigma}$', fontsize=14)
                fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

                # Absolute Error plot
                im = axes[4].imshow(ae[i, channel], cmap='hot', vmin = vmin_err,vmax = vmax_err)
                axes[4].set_xticks([])
                axes[4].set_yticks([])
                axes[4].set_title(r'$\rm{Absolute \ error}$', fontsize=14)
                fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
            else:
                    
                im = axes[3].imshow(ae[i, channel, :, :], cmap='hot',vmin = vmin_err,vmax = vmax_err)
                axes[3].set_xticks([])
                axes[3].set_yticks([])
                axes[3].set_title(r'$\rm{Absolute \ error}$', fontsize=14)
                fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

            # Save each individual figure
            save_tests(plt, self.cfg, self.filepath, f'sample_{i+1}')
            plt.close(fig)

    def _plot_mean_error_history_HMC(self):
        # Construct the paths
        fig_filename = 'mse_error_history_HMC.pdf'
        fig_path, _ = construct_fig_paths(self.cfg, fig_filename)
        _, _, train_path = construct_paths(self.cfg)
        error_history_path = os.path.join(train_path, 'HMC_mse_history.pkl')
        with open(error_history_path, 'rb') as file:
            error_history = pickle.load(file)
        plt.figure()
        for i in range(10):
            plt.plot(error_history[:, i], label=f'{i+1}st microstructure')
        plt.xlabel(r'$\mathrm{HMC - Iteration}$')
        plt.ylabel(r'$\mathrm{Mean \ Squared \ Error (MSE)}$')        # plt.title('Mean Error History')
        # plt.grid(True)
        plt.legend()
        plt.savefig(fig_path)
        plt.close()

    def _plot_max_parameters_HMC(self):
        ft_size = 20
        _, _, train_path = construct_paths(self.cfg)
        file_pattern='max_params_*_HMC.pkl'
        # Find all files in the folder that match the pattern
        file_paths = [os.path.join(train_path, f) for f in os.listdir(train_path) if fnmatch.fnmatch(f, file_pattern)]
        # Initialize a list to do store all values
        all_values = np.zeros((10, len(file_paths)))
        # Load values from each file
        for i, file_path in enumerate(sorted(file_paths)):
            with open(file_path, 'rb') as file:
                values = pickle.load(file)
                all_values[:, i] = values.detach().cpu().numpy()
                
        _, dir_path = construct_fig_paths(self.cfg, get_filepath(self.cfg.method))
        save_path = os.path.join(dir_path, 'max_param_histogram_HMC.pdf')            
        fig, axs = plt.subplots(10, 1, figsize=(10, 40))
        rgb_color = (0, 0, 255/255)  # Normalize the RGB values
        for i in range(10):
            axs[i].hist(all_values[i, :], bins=80, edgecolor='black', density=True, color=rgb_color)
            axs[i].set_title(r'$\mathrm{Parameter} \ $' + rf'${i+1}$' + r'$ \ \mathrm{Histogram}$', fontsize=ft_size)
            axs[i].set_xlabel(r'$\mathrm{Value}$',fontsize=ft_size)
            axs[i].set_ylabel(r'$\mathrm{Count}$',fontsize=ft_size)
        plt.tight_layout()
        save_tests(plt, self.cfg, self.filepath, 'histogram_all')
        plt.close(fig)
        
        for i in range(10):
            plt.figure(figsize=(5, 3))  # Create a new figure for each histogram
            plt.hist(all_values[i, :], bins=80, edgecolor='black', density=True, color=rgb_color)
            # plt.title(r'$\mathrm{Parameter\ }' + rf'${i+1}$' + r'$\ \mathrm{Histogram}$', fontsize=14)
            plt.xlabel(r'$\mathrm{Value}$',fontsize=ft_size)
            plt.ylabel(r'$\mathrm{Frequency}$',fontsize=ft_size)  # Use 'Frequency' for density=True
            plt.tight_layout()
            save_tests(plt, self.cfg, self.filepath, f'histogram_{i+1}')
            plt.close()  # Close the current figur
        
        for i in range(10):
            plt.figure(figsize=(9,3))
            plt.plot(all_values[i, :], marker='.', linestyle='-', color=rgb_color)
            # plt.title(r'$\mathrm{Parameter} \ $' + rf'${i+1}$' + r'$ \ \mathrm{Histogram}$', fontsize=14)
            plt.xlabel(r'$\mathrm{HMC \ iteration}$',fontsize=ft_size)
            plt.ylabel(r'$\mathrm{Parameter \ value}$',fontsize=ft_size)
            plt.tight_layout()
            save_tests(plt, self.cfg, self.filepath, f'trajectory_{i+1}')
            plt.close()
        
    def _plot_training_history_BBB(self):
        _, history_path, _  = construct_paths(self.cfg)
        _, directory = construct_fig_paths(self.cfg, self.filepath)
        # Load the training history
        history = np.load(history_path)
        # Extract values
        loss_train = history['loss_train']
        loss_valid = history['loss_valid']
        KL_train = history['kl_train']
        beta_history = history['beta']
        
        # Plot combined losses
        fig1, ax1 = plt.subplots()
        ax1.plot(loss_train, 'o-', label='Training Loss')
        ax1.plot(loss_valid, 'o-', label=f'Validation Loss- Beta Type: {self.cfg.beta_type} - Samples: {self.cfg.num_samples}')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Losses over Epochs')
        ax1.legend()
        plt.savefig(directory + '/' + self.filepath + '_combined_losses.pdf')

        # Plot kl_train separately
        fig2, ax2 = plt.subplots()
        ax2.plot(KL_train, 'o-', color='red', label=f'KL Train Loss - Beta Type: {self.cfg.beta_type} - Samples: {self.cfg.num_samples}')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('KL Loss')
        ax2.set_title('KL Train Loss over Epochs')
        ax2.legend()
        plt.savefig(directory + '/' + self.filepath + '_kl_train_loss.pdf')
        
        # Plot beta history
        fig3, ax3 = plt.subplots()
        ax3.plot(beta_history, 'o-', color='blue', label=f'Beta History - Beta Type: {self.cfg.beta_type}')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Beta Value')
        ax3.set_title('Beta History over Epochs')
        ax3.legend()
        plt.savefig(directory + '/' + self.filepath + '_beta_history.pdf')

        # Plot ratio of losses kl_train separately
        fig4, ax4 = plt.subplots()
        ax4.plot(KL_train/loss_train, 'o-', color='green', label=f'KL Train/loss_train - Beta Type: {self.cfg.beta_type}')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Loss/ratio')
        ax4.legend()
        # Save the KL loss plot
        plt.savefig(directory + '/' + 'loss_ratio.pdf')

    def run_plotting(self):
        method = self.cfg.method
        Plotting._plot_predictions(self, Nsamp=40, channel=0)
        Plotting._plot_predictions_individual(self, Nsamp=40, channel=0)
        if method in ['BBB', 'BBB_LRT']:
            Plotting._plot_training_history_BBB(self)
        if method in ['HMC']:
            Plotting._plot_mean_error_history_HMC(self)
            Plotting._plot_max_parameters_HMC(self)

class PlottingComparisonMCD:
    def __init__(self, cfg):
        self.cfg = cfg
        self.index = 8
    def _plot_comparison(self, data, save_name, calculate_metric=None, data_hmc=None):
        fig, axes = plt.subplots(nrows=len(self.cases), ncols=len(self.drop_rates), figsize=(8, 6.4))
        fig.subplots_adjust(wspace=0.2, hspace=-0.9)
        ft_size = 10
        cbar_ft_size = 6

        for j, drop_rate in enumerate(self.drop_rates):
            temp_min = []
            temp_max = []

            # Find the min and max for each column (drop_rate)
            for case_i in self.cases:
                if calculate_metric:
                    _, err_image, _ = calculate_metric(data[case_i][drop_rate][self.index, 0, :, :], data_hmc[self.index, 0, :, :])
                    temp_min.append(np.min(err_image))
                    temp_max.append(np.max(err_image))
                else:
                    temp_min.append(np.min(data[case_i][drop_rate][self.index, 0, :, :]))
                    temp_max.append(np.max(data[case_i][drop_rate][self.index, 0, :, :]))

            # col_min = np.min(temp_min)
            # col_max = np.max(temp_max)
            if self.cfg.config_type == 'fiber':
                col_min = 0
                col_max = 1
            elif self.cfg.config_type == 'polycrystalline':
                col_min = 0.01
                col_max = 0.1
            
            for i, case_i in enumerate(self.cases):
                ax = axes[i][j]
                err_value = None
                if calculate_metric:
                    err_value, err_image, metric_type = calculate_metric(data[case_i][drop_rate][self.index, 0, :, :], data_hmc[self.index, 0, :, :])
                    im = ax.imshow(err_image, cmap='pink', aspect='auto', vmin=col_min, vmax=col_max)
                else:
                    im = ax.imshow(data[case_i][drop_rate][self.index, 0, :, :], cmap='hot', aspect='auto', vmin=col_min, vmax=col_max)

                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
                if calculate_metric:
                    ax.set_xlabel(rf'$\mathrm{{{metric_type}}} = \mathrm{{{err_value:.3f}}}$', fontsize=0.8*ft_size)
                else:
                    xlabel_text = ' a '
                    ax.set_xlabel(xlabel_text, fontsize=0.8*ft_size, color='white', alpha=0)

                ax.set_xticklabels([])
                ax.set_yticklabels([])

                if j == 0:
                    ax.set_ylabel(r'$\mathrm{Case \ }$' + rf'${self.case_names[i]}$', fontsize=ft_size, labelpad=10)

        for j, drop_rate in enumerate(self.drop_rates):
            axes[0][j].set_title(r'$\mathrm{p}=$' + rf'$\mathrm{self.drop_rates_names[j]}$', fontsize=ft_size)

        if self.cfg.config_type == 'polycrystalline':
            save_path = os.path.join('../figures_polycrystalline_2D/', save_name)
        elif self.cfg.config_type == 'fiber':
            save_path = os.path.join('../figures_fiber/', save_name)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        
    def _plot_MCD_stds(self, stds_MCD):
        self._plot_comparison(stds_MCD, 'MCD_std.pdf', calculate_metric=None)
    
    def _plot_MCD_HMC_ssim(self, stds_MCD, std_HMC):
        def calculate_ssim(data, data_hmc):
            mssim, ssi_image = ssim(data, data_hmc, full=True, data_range=data.max() - data.min(), win_size=3)
            return mssim, ssi_image, 'SSIM'
        self._plot_comparison(stds_MCD, 'MCD_HMC_ssim.pdf', calculate_metric=calculate_ssim, data_hmc=std_HMC)

    def _plot_MCD_HMC_ae(self, stds_MCD, std_HMC):
        def calculate_ae(data, data_hmc):
            ae = np.abs(((data - data_hmc)))
            return np.mean(ae), ae, 'MAE'
        self._plot_comparison(stds_MCD, 'MCD_HMC_ae.pdf', calculate_metric=calculate_ae, data_hmc=std_HMC)
    
    
    def _calculate_errors(self, Y_ts, mean_pred_HMC=None, mean_pred_MCD=None, mean_pred_BBB=None, std_pred_HMC=None, std_pred_MCD=None, std_pred_BBB=None):
        # Convert any NaN values to zero
        conversions = [mean_pred_HMC, mean_pred_MCD, mean_pred_BBB, std_pred_HMC, std_pred_MCD, std_pred_BBB]
        conversions = [np.nan_to_num(item) if item is not None else None for item in conversions]
        mean_pred_HMC, mean_pred_MCD, mean_pred_BBB, std_pred_HMC, std_pred_MCD, std_pred_BBB = conversions

        data = []

        # Compute errors and total variance for each method if available
        for method, mean_pred, std_pred in zip(["HMC", "MCD", "BBB"], conversions[:3], conversions[3:]):
            if mean_pred is not None:
                mean_stress_error = round(np.mean(np.abs(Y_ts - mean_pred)), 4)
                std_stress_error = round(np.std(np.abs(Y_ts - mean_pred)), 4)
                average_variance = round(np.mean(std_pred),4)
                if std_pred_HMC is not None:
                    mean_std_stress_error = round(np.mean(np.abs(std_pred - std_pred_HMC)), 4)
                    std_std_stress_error = round(np.std(np.abs(std_pred - std_pred_HMC)), 4)
                else:
                    mean_std_stress_error = None
                    std_std_stress_error = None
                
                data.append({
                    "Method": method,
                    "stress error - absolute mean": mean_stress_error,
                    "stress error - variance": std_stress_error,
                    "average variance": average_variance,
                    "sigma error - absolute mean": mean_std_stress_error,
                    "sigma error - variance": std_std_stress_error
                })

        df = pd.DataFrame(data)

        return df


    def run_plotting_comparison(self, cases, case_names, drop_rates, drop_rates_names):
        data_path = self.cfg.data_path
        self.method = 'HMC'
        # Load the HMC data
        self.train_path = os.path.join(construct_paths(self.cfg)[2])
        mean_HMC, std_HMC = Plotting.load_mean_and_std(self)
        _, _, _, X_ts, Y_ts, self.names_ts = load_test_data(self.cfg)
        # Switch to MCD method configurations
        self.cfg.method = 'MCD'
        self.cases = cases
        self.case_names = case_names
        self.drop_rates = drop_rates
        self.drop_rates_names = drop_rates_names

        means_MCD = {}
        stds_MCD = {}
        results_df = pd.DataFrame()


        # Initialize nested dictionaries for each case and drop rate
        for case in self.cases:
            case_i = f'{case}'  
            means_MCD[case_i] = {}  
            stds_MCD[case_i] = {} 
            if self.cfg.config_type == 'fiber':  
                data_path = os.path.join('../trained_models_fiber', f'case_{case_i}')
            elif self.cfg.config_type == 'polycrystalline':
                data_path = os.path.join('../trained_models_polycrystalline_2D', f'case_{case_i}')
                    
                
            for drop_rate_key_i in self.drop_rates:
                mean_file = os.path.join(data_path, f'MCD_{drop_rate_key_i}_mean.pkl')
                std_file = os.path.join(data_path, f'MCD_{drop_rate_key_i}_std.pkl')
                
                # Checking if files exist before attempting to open them
                if os.path.exists(mean_file) and os.path.exists(std_file):
                    with open(mean_file, 'rb') as f:
                        means_MCD[case_i][drop_rate_key_i] = pickle.load(f)
                    with open(std_file, 'rb') as f:
                        stds_MCD[case_i][drop_rate_key_i] = pickle.load(f)

                mean_pred_MCD = means_MCD[case_i].get(drop_rate_key_i, None)
                std_pred_MCD = stds_MCD[case_i].get(drop_rate_key_i, None)
                mean_pred_BBB, std_pred_BBB = PlottingComparisonAll._load_mean_and_std(self,'BBB')
                
                df = self._calculate_errors(Y_ts, mean_HMC, mean_pred_MCD, mean_pred_BBB, std_HMC, std_pred_MCD, std_pred_BBB)
                df['Case'] = case_i
                df['Drop Rate'] = drop_rate_key_i
                results_df = pd.concat([results_df, df])
                
                    
        
        drop_colums = ['stress error - absolute mean', 'stress error - variance', 'sigma error - absolute mean', 'sigma error - variance', 'Drop Rate']
        results_df = results_df.drop_duplicates(subset=drop_colums)
        # determining the name of the file
        file_name = self.cfg.data_path + '/comparison_results.csv' 
        results_df.to_csv(file_name,index=False)
        print(results_df.iloc[:, :].to_latex(index=False,float_format="%.5f"))
        # Plotting functions can be called here, assuming they are defined elsewhere in your class
        self._plot_MCD_stds(stds_MCD)
        if std_HMC is not None:
            self._plot_MCD_HMC_ae(stds_MCD, std_HMC)
            self._plot_MCD_HMC_ssim(stds_MCD, std_HMC)

        return results_df

class PlottingComparisonAll:
    def __init__(self, cfg):
        train_path = os.path.join(construct_paths(cfg)[2])
        self.cfg = cfg
        self.train_path = train_path
        self.index = 4  
        _, _, _, self.X_ts, self.Y_ts, self.names_ts = load_test_data(cfg)
        self.mean_pred_MCD, self.std_pred_MCD = self._load_mean_and_std(self,'MCD', '0.2')
        self.mean_pred_BBB, self.std_pred_BBB = self._load_mean_and_std(self,'BBB')
        self.mean_pred_HMC, self.std_pred_HMC = self._load_mean_and_std(self,'HMC')
        self.mean_pred_Det, self.std_pred_Det = self._load_mean_and_std(self,'Deterministic')
        
    def _load_weights(self, filename):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
            # Convert to NumPy array if not already
            # weights = [np.array(weight) for weight in weights]
        return weights
    
    @staticmethod
    def _load_mean_and_std(self, method, drop_rate=None):
        mean_filename = os.path.join(self.train_path, method + '_mean.pkl')
        std_filename = os.path.join(self.train_path, method + '_std.pkl')
        if method == 'MCD':
            drop_rate_str = str(drop_rate).replace('.', '')
            mean_filename = os.path.join(self.train_path, method + f'_{drop_rate_str}_mean.pkl')
            std_filename = os.path.join(self.train_path, method + f'_{drop_rate_str}_std.pkl')
        
        # check if filename exists before attempting to open it
        if os.path.exists(mean_filename) and os.path.exists(std_filename):
            with open(mean_filename, 'rb') as f:
                mean = pickle.load(f)
            with open(std_filename, 'rb') as f:
                std = pickle.load(f)
        else:
            mean = None
            std = None
        return mean, std
        
    def _plot_comparison_all(self, Nsamp=40, channel = 0):
        ae_MCD = np.abs(self.Y_ts - self.mean_pred_MCD)
        ae_BBB = np.abs(self.Y_ts - self.mean_pred_BBB)

        if self.std_pred_HMC is not None:
            ae_HMC = np.abs(self.Y_ts - self.mean_pred_HMC)
            n_rows = 4
            n_columns = 3
        else:
            ae_HMC = None
            n_rows = 3 
            n_columns = 3
        
        if self.cfg.config_type == 'polycrystalline':
            colormap = 'plasma'
            colormap2 = 'bwr'
        elif self.cfg.config_type == 'fiber':
            colormap = 'viridis'
            colormap2 = 'viridis'
        for i in range(Nsamp):            
            fig, axes = plt.subplots(n_rows, n_columns, figsize=(4 * n_columns, 3.75 * n_rows))
            fig.subplots_adjust(wspace=0.15, hspace=0.02)
            ft_size = 18
            cbar_ft_size = 8
            
            vmin = 0.8 * np.min(self.Y_ts[i, channel])
            vmax = 1.2 * np.max(self.Y_ts[i, channel])
            
            if self.std_pred_HMC is not None:
                vmin_std = np.min([self.std_pred_HMC[i, channel, :, :], 
                                self.std_pred_BBB[i, channel, :, :], 
                                self.std_pred_MCD[i, channel, :, :]])

                vmax_std = np.max([self.std_pred_HMC[i, channel, :, :], 
                                self.std_pred_BBB[i, channel, :, :], 
                                self.std_pred_MCD[i, channel, :, :]])
                vmin_ae = np.min([ae_HMC[i, channel, :, :], 
                                ae_BBB[i, channel, :, :], 
                                ae_MCD[i, channel, :, :]])
                vmax_ae = np.max([ae_HMC[i, channel, :, :], 
                                ae_BBB[i, channel, :, :], 
                                ae_MCD[i, channel, :, :]])
            else:
                vmin_std = np.min([self.std_pred_BBB[i, channel, :, :], 
                                self.std_pred_MCD[i, channel, :, :]])

                vmax_std = np.max([self.std_pred_BBB[i, channel, :, :], 
                                self.std_pred_MCD[i, channel, :, :]])
                
                vmin_ae = np.min([ae_BBB[i, channel, :, :], 
                                ae_MCD[i, channel, :, :]])
                vmax_ae = np.max([ae_BBB[i, channel, :, :],
                                  ae_MCD[i, channel, :, :]])
                
            j = 0
            im0 = axes[j, 0].imshow(self.Y_ts[i, channel], cmap=colormap, vmin=vmin, vmax=vmax)
            axes[j, 0].set_title(r'$\mathrm{Target \ stress}$', fontsize=ft_size)
            axes[j, 0].set_xticks([])
            axes[j, 0].set_yticks([])
            cbar = fig.colorbar(im0, ax=axes[j, 0], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here

            if self.mean_pred_HMC is not None:                        
                j = j + 1
                im1 = axes[j, 0].imshow(self.mean_pred_HMC[i, channel], cmap=colormap, vmin=vmin, vmax=vmax)
                axes[j,0].set_title(r'$\rm{Mean \ stress \ - HMC}$', fontsize=ft_size)
                axes[j,0].set_xticks([])
                axes[j,0].set_yticks([])
                cbar = fig.colorbar(im1, ax=axes[j, 0], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
            
            j = j + 1
            im2 = axes[j,0].imshow(self.mean_pred_BBB[i, channel], cmap=colormap, vmin=vmin, vmax=vmax)
            axes[j,0].set_title(r'$\rm{Mean \ stress \ - BBB}$',fontsize=ft_size)
            axes[j,0].set_xticks([])
            axes[j,0].set_yticks([])
            cbar = fig.colorbar(im2, ax=axes[j, 0], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
            
            j = j + 1
            im3 = axes[j, 0].imshow(self.mean_pred_MCD[i, channel], cmap=colormap, vmin=vmin, vmax=vmax)
            axes[j,0].set_title(r'$\rm{Mean \ stress \ - MCD}$', fontsize=ft_size)
            axes[j,0].set_xticks([])
            axes[j,0].set_yticks([])
            cbar = fig.colorbar(im3, ax=axes[j, 0], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
            
            j = 0
            im00 = axes[j, 1].imshow(self.X_ts[i, channel], cmap=colormap2)
            axes[j, 1].set_title(r'$\rm{Microstructure}$', fontsize=ft_size)
            axes[j, 1].set_xticks([])
            axes[j, 1].set_yticks([])
            cbar = fig.colorbar(im00, ax=axes[j, 1], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
            if self.cfg.config_type == 'fiber':
                cbar.ax.set_visible(False)  # Hide the colorbar
            # axes[0,1].axis('off')
            
            if ae_HMC is not None:
                j = j + 1
                im4 = axes[j, 1].imshow(ae_HMC[i, channel], cmap='hot',vmin = vmin_ae, vmax = vmax_ae)
                axes[j, 1].set_title(r'$\rm{Absolute \ error  \ - \ HMC}$', fontsize=ft_size)
                axes[j, 1].set_xticks([])
                axes[j, 1].set_yticks([])
                cbar = fig.colorbar(im4, ax=axes[j, 1], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
            
            if ae_BBB is not None:
                j = j + 1
                im5 = axes[j, 1].imshow(ae_BBB[i, channel], cmap='hot', vmin = vmin_ae, vmax = vmax_ae)
                axes[j, 1].set_title(r'$\rm{Absolute \ error  \ - \ BBB}$', fontsize=ft_size)
                axes[j, 1].set_xticks([])
                axes[j, 1].set_yticks([])
                cbar = fig.colorbar(im5, ax=axes[j, 1], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
            
            if ae_MCD is not None:
                j = j + 1
                im6 = axes[j, 1].imshow(ae_MCD[i, channel], cmap='hot', vmin = vmin_ae, vmax = vmax_ae)
                axes[j, 1].set_title(r'$\rm{Absolute \ error  \ - \ MCD}$')
                axes[j, 1].set_xticks([])
                axes[j, 1].set_yticks([])
                cbar = fig.colorbar(im6, ax=axes[j, 1], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here

            if n_columns == 3:                
                axes[0, 2].set_xticks([])
                axes[0, 2].set_yticks([])
                axes[0, 2].axis('off')

            j = 0 
            if self.std_pred_HMC is not None:
                j = j + 1
                im7 = axes[j, 2].imshow(self.std_pred_HMC[i, channel], cmap='hot', vmin=vmin_std, vmax=vmax_std)
                axes[j, 2].set_title(r'$\rm{\sigma \ - HMC}$', fontsize=ft_size)
                axes[j, 2].set_xticks([])
                axes[j, 2].set_yticks([])
                cbar = fig.colorbar(im7, ax=axes[j, 2], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
                
            j = j + 1
            im8 = axes[j, 2].imshow(self.std_pred_BBB[i, channel], cmap='hot', vmin=vmin_std, vmax=vmax_std)
            axes[j, 2].set_title(r'$\rm{\sigma \ - BBB}$', fontsize=ft_size)
            axes[j, 2].set_xticks([])
            axes[j, 2].set_yticks([])
            cbar = fig.colorbar(im8, ax=axes[j, 2], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
            
            j = j + 1
            im9 = axes[j, 2].imshow(self.std_pred_MCD[i, channel], cmap='hot', vmin=vmin_std, vmax=vmax_std)
            axes[j, 2].set_title(r'$\rm{\sigma \ - MCD}$', fontsize=ft_size)
            axes[j, 2].set_xticks([])
            axes[j, 2].set_yticks([])
            cbar = fig.colorbar(im9, ax=axes[j, 2], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
            
            save_name = f'comparison_all_{i}.pdf'
            save_path = construct_dir_path(self.cfg,save_name)
            plt.savefig(save_path, dpi=100)
            plt.close(fig)  # Close the figure to free memory

    def _plot_std_comparison(self, Nsamp=40, channel = 0):
        val_error_MCD = []
        val_error_BBB = []
        for i in range(1, Nsamp + 1):  # Assuming i starts from 1
            n_columns = 3
            n_rows = 2
            fig, axes = plt.subplots(2, 3, figsize=(4 * n_columns, 3.75 * n_rows))
            fig.subplots_adjust(wspace=0.15, hspace=0.02)
            ft_size = 18
            cbar_ft_size = 8
            
            # Standard deviations
            std_HMC = self.std_pred_HMC[i - 1,channel,:,:]  
            std_MCD = self.std_pred_MCD[i - 1,channel,:,:]
            std_BBB = self.std_pred_BBB[i - 1,channel,:,:]
            
            vmin_std = np.min([std_HMC, std_BBB, std_MCD])
            vmax_std = np.max([std_HMC, std_BBB, std_MCD])
            
            # Calculate the differences/errors
            error_HMC_MCD = np.abs(std_HMC - std_MCD)
            error_HMC_BBB = np.abs(std_HMC - std_BBB)
            
            # Print the error 
            val_error_BBB.append(error_HMC_BBB/std_HMC)
            val_error_MCD.append(error_HMC_MCD/std_HMC)

            vmin_error = np.min([error_HMC_MCD, error_HMC_BBB])
            vmax_error = np.max([error_HMC_MCD, error_HMC_BBB])
            
            axes[0, 0].set_xticks([])
            axes[0, 0].set_yticks([])
            axes[0, 0].axis('off')

            # Plot HMC std in the first column, centered
            im0 = axes[1, 0].imshow(std_HMC, cmap='hot', vmin = vmin_std, vmax = vmax_std)
            axes[1, 0].set_title(r'$\mathrm{\sigma \ - \ HMC}$', fontsize = ft_size)
            axes[1, 0].axis('off')  # Turn off axis
            cbar = fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
            # fig.colorbar(im0, ax=axes[1, 0])

            # Plot MCD and BBB std in the second column
            im1 = axes[0, 1].imshow(std_MCD, cmap='hot', vmin = vmin_std, vmax = vmax_std)
            axes[0, 1].set_title(r'$\mathrm{\sigma \ - \ MCD}$', fontsize = ft_size)
            axes[0, 1].axis('off')
            cbar = fig.colorbar(im0, ax=axes[0,1], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
            # fig.colorbar(im1, ax=axes[0, 1])

            im2 = axes[1, 1].imshow(std_BBB, cmap='hot', vmin = vmin_std, vmax = vmax_std)
            axes[1, 1].set_title(r'$\mathrm{\sigma \ - \ BBB}$', fontsize = ft_size)
            axes[1, 1].axis('off')
            cbar = fig.colorbar(im0, ax=axes[1, 1], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
            # fig.colorbar(im2, ax=axes[1, 1])

            # Plot the errors in the third column
            im3 = axes[0, 2].imshow(error_HMC_MCD, cmap='hot', vmin = vmin_error, vmax = vmax_error)
            axes[0, 2].set_title(r'$\mathrm{Relative \ error \ MCD}$', fontsize = ft_size)
            axes[0, 2].axis('off')
            cbar = fig.colorbar(im0, ax=axes[0, 2], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
            # fig.colorbar(im3, ax=axes[0, 2])

            im4 = axes[1, 2].imshow(error_HMC_BBB, cmap='hot', vmin = vmin_error, vmax = vmax_error)
            axes[1, 2].set_title(r'$\mathrm{Relative \ error  \ BBB}$', fontsize = ft_size)
            axes[1, 2].axis('off')
            cbar = fig.colorbar(im0, ax=axes[1, 2], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=cbar_ft_size)  # Set the font size here
            # fig.colorbar(im4, ax=axes[1, 2])

            # Save the figure
            save_name = f'std_comparison_{i}.pdf'
            save_path = construct_dir_path(self.cfg,save_name)
            plt.savefig(save_path, dpi=100)
            plt.close(fig)  # Close the figure to free memory                                           

        print(f"The average validation error for the std in BBB is {np.mean(np.array(val_error_BBB)):.3f}")
        print(f"The average validation error for the std in MCD is {np.mean(np.array(val_error_MCD)):.3f}")

    def _plot_weights_histogram_BBB_HMC(self):
        bbb_weights_file = os.path.join(self.train_path, 'BBB_weight_samples.pkl')
        hmc_weights_file = os.path.join(self.train_path, 'HMC_weight_samples.pkl')
        rgb_color = (0, 0, 255/255)  # Normalize the RGB values
        # Load the weights
        ft_size = 8
        ft_size_legend = 5
        bbb_weights = self._load_weights(bbb_weights_file)
        hmc_weights = self._load_weights(hmc_weights_file)
        Nsamp = 40
        # Plot for the first 10 samples
        for i in range(Nsamp):
            
            fig, axes = plt.subplots(1, 2, figsize=(5, 2.15))  # Create a figure with 1 row and 2 columns of subplots

            # Flatten the weights for histogram if they are multidimensional
            bbb_sample_weights = bbb_weights[:,i].flatten()
            hmc_sample_weights = hmc_weights[:,i].flatten()

            axes[0].hist(bbb_sample_weights, bins=30, alpha=0.8, label=r'$\mathrm{NN \ parameter  \ - \ BBB}$', density=True, color=rgb_color)
            axes[1].hist(hmc_sample_weights, bins=30, alpha=0.8, label=r'$\mathrm{NN \ parameter \ - \ HMC}$', density=True, color=rgb_color)

            axes[0].set_xlabel(r'$\mathrm{Value}$',fontsize=ft_size)
            axes[1].set_xlabel(r'$\mathrm{Value}$',fontsize=ft_size)
            axes[0].set_ylabel(r'$\mathrm{Frequency}$',fontsize=ft_size) 
            
            axes[0].legend(loc='upper right', fontsize = ft_size_legend)
            axes[1].legend(loc='upper right', fontsize = ft_size_legend)
            
            axes[0].tick_params(axis='both', labelsize=5)
            axes[1].tick_params(axis='both', labelsize=5)

    
            plt.tight_layout()
            save_name = f'weight_hist_BBB_HMC_{i}.pdf'
            save_path = construct_dir_path(self.cfg,save_name)
            save_path = os.path.join('../figures_fiber/', save_name)
            plt.savefig(save_path, dpi=100)
            plt.close(fig)
            
        for i in range(Nsamp):
            plt.figure(figsize=(10,2.5))
            plt.plot(hmc_weights[:,i].flatten(), marker='.', linestyle='-', color=rgb_color, alpha=0.8, label=r'$\mathrm{NN \ parameter\ -  \ HMC}$')
            plt.legend(loc='upper right', fontsize=0.6* ft_size)
            plt.ylabel(r'$\mathrm{Parameter \ Value}$',fontsize=ft_size)
            plt.xlabel(r'$\mathrm{HMC \ - \ iteration}$',fontsize=ft_size)
            # plt.legend(r'$\mathrm{NN \ parameter \ HMC}$',fontsize=ft_size)
            plt.tight_layout()
            save_name = f'trajectory_{i+1}.pdf'
            save_path = construct_dir_path(self.cfg,save_name)
            plt.savefig(save_path, dpi=100)
            plt.close()
    
    def _plot_weights_mean_vs_std_HMC(self):
        hmc_weights_file = os.path.join(self.train_path, 'HMC_weight_samples.pkl')
        hmc_weights = self._load_weights(hmc_weights_file)
        Nsamp = 500  # Number of samples to consider

        # Initialize lists to hold the means and standard deviations
        means = []
        stds = []
        rgb_color = (0, 0, 255/255)  # Normalize the RGB values
        # Calculate the mean and standard deviation for each sample
        for i in range(Nsamp):
            sample_weights = hmc_weights[:, i]
            means.append(np.mean(sample_weights))
            stds.append(np.std(sample_weights))

        # Now plot mean vs standard deviation
        plt.figure(figsize=(10, 5))
        plt.scatter(stds, means, color=rgb_color)
        # plt.title('Mean vs. Standard Deviation of HMC Weights for First 100 Iterations')
        plt.xlabel(r'$\mathrm{\sigma \ of \ parameters}$')
        plt.ylabel(r'$\mathrm{Mean \ of \ parameters}$')
        plt.tight_layout()
        save_name = f'weights_mean_vs_std_HMC.pdf'
        save_path = construct_dir_path(self.cfg,save_name)
        save_path = os.path.join('../figures_fiber/', save_name)
        plt.savefig(save_path, dpi=100)
        
        
    def _calculate_comparison_all(self):
        mean_stress_error_HMC = round(np.mean(np.abs(self.Y_ts - self.mean_pred_HMC)),4)
        mean_stress_error_BBB = round(np.mean(np.abs(self.Y_ts - self.mean_pred_BBB)),4)
        mean_stress_error_MCD = round(np.mean(np.abs(self.Y_ts - self.mean_pred_MCD)),4)
        
        std_stress_error_HMC = round(np.std(np.abs(self.Y_ts - self.mean_pred_HMC)),4)
        std_stress_error_BBB = round(np.std(np.abs(self.Y_ts - self.mean_pred_BBB)),4)
        std_stress_error_MCD = round(np.std(np.abs(self.Y_ts - self.mean_pred_MCD)),4)
        
        mean_std_stress_error_MCD = round(np.mean(np.abs(self.std_pred_MCD - self.std_pred_HMC)),4)
        mean_std_stress_error_BBB = round(np.mean(np.abs(self.std_pred_BBB - self.std_pred_HMC)),4)
        std_std_stress_error_MCD = round(np.std(np.abs(self.std_pred_MCD - self.std_pred_HMC)),4)
        std_std_stress_error_BBB = round(np.std(np.abs(self.std_pred_BBB - self.std_pred_HMC)),4)

        # Create a DataFrame
        data = {
            "stress error - absolute mean": [mean_stress_error_HMC, mean_stress_error_MCD, mean_stress_error_BBB],
            "stress error - variance": [std_stress_error_HMC, std_stress_error_MCD, std_stress_error_BBB],
            "sigma error - absolute mean": [None, mean_std_stress_error_MCD, mean_std_stress_error_BBB],
            "sigma error - variance": [None, std_std_stress_error_MCD, std_std_stress_error_BBB]
        }

        index = ["HMC", "MCD", "BBB"]
        df = pd.DataFrame(data, index=index)

        # Export to LaTeX
        latex_table = df.to_latex(float_format="%.4f", na_rep="-", caption="Comparison of stress and Standard Deviation Errors", label="tab:error_comparison")
        print(latex_table)

        # save_name = "error_comparison_table.tex"
        # save_path = construct_dir_path(self.cfg,save_name)
        # with open(save_path, "w") as file:
        #     file.write(latex_table)

    def run_plotting_comparison_all(self):
        self._plot_comparison_all()
        if self.mean_pred_HMC is not None:
            self._plot_std_comparison()
            self._plot_weights_histogram_BBB_HMC()
            self._calculate_comparison_all()
            self._plot_weights_mean_vs_std_HMC()

# %%
