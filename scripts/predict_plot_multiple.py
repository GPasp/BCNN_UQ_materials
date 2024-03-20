# Import necessary modules
import sys
sys.path.append("..")
from scripts.ModelPredict import BayesianModelPredict, SampleNetworks
from scripts.ModelPlot import Plotting, PlottingComparisonMCD, PlottingComparisonAll, Plotting
from utils.utils_process import Config
from utils.utils_process import config_from_json
# %% Specify the dataset -------------------------
config_type = config_from_json()

# Function to run prediction and plotting
def run_process(config_type, cases, drop_rates):
    # Load the configuration file
    cfg = Config(config_type)
    cfg.config_type = config_type
    print(f"Using device: {cfg.device}")

    # Iterate over each case
    for case, drop_idx_config in cases.items():
        cfg.case = case
        cfg.drop_idx_en = drop_idx_config['en']
        cfg.drop_idx_dec = drop_idx_config['dec']

        # Iterate over each dropout rate
        for drop_rate in drop_rates:
            cfg.drop_rate = drop_rate
            cfg.method = 'MCD'  
            predictor = BayesianModelPredict(cfg)
            predictor.compute_and_save_predictions()
            plotter = Plotting(cfg)
            plotter.run_plotting()
            
if __name__ == "__main__":
    if config_type == 'polycrystalline':
        cases_run = {
            '1a': {'en': [0, 0, 0, 0, 1], 'dec': [0, 0, 0, 0, 0]},
            '1b': {'en': [0, 0, 0, 0, 0], 'dec': [0, 0, 0, 0, 1]},
            '1c': {'en': [0, 0, 0, 1, 1], 'dec': [0, 0, 0, 0, 0]},
            '1d': {'en': [0, 0, 0, 0, 0], 'dec': [0, 0, 0, 1, 1]},
        }
        cases_actual = ['1a', '1b', '1c', '1d']
        case_names = ['1', '2', '3', '4']
        drop_rates_actual = ['001', '005', '01', '02']
        drop_rates_names = ['0.01', '0.05', '0.1', '0.2']
        drop_rates = [float(rate) for rate in drop_rates_names]
                
    elif config_type == 'fiber':        
        cases_run = {
            '4a': {'en': [0, 0, 0, 0, 1], 'dec': [0, 0, 0, 0, 0]},
            '4b': {'en': [0, 0, 0, 0, 0], 'dec': [0, 0, 0, 0, 1]},
            '4c': {'en': [0, 0, 0, 1, 1], 'dec': [0, 0, 0, 0, 0]},
            '4d': {'en': [0, 0, 0, 0, 0], 'dec': [0, 0, 0, 1, 1]},
        }
        cases_actual = ['4a', '4b', '4c', '4d']
        case_names = ['1', '2', '3', '4']
        drop_rates_actual = ['001', '005', '01', '02']
        drop_rates_names = ['0.01', '0.05', '0.1', '0.2']
        drop_rates = [float(rate) for rate in drop_rates_names]
    
    cfg = Config(config_type)
    cfg.config_type = config_type
    
    run_process(config_type, cases_run, drop_rates)
    # Compare the MCD results in terms of dropout layers and cases
    plotting_comparison = PlottingComparisonMCD(cfg) 
    plotting_comparison.run_plotting_comparison(cases_actual,case_names,drop_rates_actual,drop_rates_names)
    
    # Collect the sampled weights from HMC and BBB and plot the histograms
    sample_networks = SampleNetworks(cfg)
    sample_networks._sample_weights_BBB()
    sample_networks._collect_weights_HMC()
    # Compare the results of MCD and BBB with HMC
    plotting_comparison_all = PlottingComparisonAll(cfg)
    plotting_comparison_all.run_plotting_comparison_all()
    

    
