import numpy as np
import pandas as pd
import re

import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio

from dnn_weight_noise import DnnLayerWeightExperiment, DNN

# device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

experiment_versions = ['v43','v44']
dnn_names = ['dnn2a','dnn2b','dnn2c','dnn2d']

  ################################
 ###### final epoch plots #######
################################ 

def create_final_epoch_layer_plots(experiment_versions, dnn_names):
    """Create final epoch versus layer plots, related to training"""

    for experiment_version in experiment_versions:
        full_model_path = f"experiment_plots/{experiment_version}/dnn_experiments_results.pth"

        print(f"loading model from {full_model_path}")
        dnn_experiments = torch.load(full_model_path, weights_only=False)

        for dnn_name in dnn_names:

            max_singular = []
            min_singular = []
            condition_numbers = []
            weight_means = []
            weight_stds = []
            max_abs_weight_mean_diffs = []
            layer_numbers = []

            ##          max_singular min_singular ... 
            ## layer 1
            ## layer 2

            ## count number of layers to determine last layer
            layer_counter = 0
            for name, module in dnn_experiments[dnn_name].model.named_modules():
                if isinstance(module, nn.modules.linear.Linear):
                    layer_counter += 1
            last_layer = layer_counter
            print(f"last_layer = {last_layer}")

            layer_counter = 1
            for name, module in dnn_experiments[dnn_name].model.named_modules():
                if isinstance(module, nn.modules.linear.Linear):
                    
                    # print(name)
                    ## calculate layer stats here
                    if (layer_counter != 1) & (layer_counter != last_layer):
                        layer_weights = module.weight.data.clone().numpy()
                        weight_mean = np.mean(layer_weights.flatten())
                        weight_std = np.std(layer_weights.flatten())

                        try:
                            cond_number = np.linalg.cond(layer_weights)
                        except Exception as e:
                            print(f"Warning: {str(e)}")

                        U,S,V = np.linalg.svd(layer_weights) 
                        min_singular_value = S.min()
                        max_singular_value = S.max()
                        max_abs_weight_mean_diff = np.max(np.abs(layer_weights.flatten())) - np.min(np.abs(layer_weights.flatten()))

                        max_singular.append(max_singular_value)
                        min_singular.append(min_singular_value)
                        condition_numbers.append(cond_number)
                        weight_means.append(weight_mean)
                        weight_stds.append(weight_std)
                        max_abs_weight_mean_diffs.append(max_abs_weight_mean_diff)
                        layer_numbers.append(layer_counter)
                    layer_counter += 1

            df = pd.DataFrame({
                'max_singular': max_singular, 
                'min_singular': min_singular,
                'condition_numbers': condition_numbers,
                'weight_means': weight_means,
                'weight_stds': weight_stds,
                'max_abs_weight_mean_diffs': max_abs_weight_mean_diffs,
                'layer_numbers': layer_numbers,
            })

            df = df.set_index('layer_numbers')
            df_normalized = df.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 0)
            feature_cols = df_normalized.columns
            df_normalized = df_normalized.reset_index()
            fig = px.line(df_normalized, x='layer_numbers', y=feature_cols)
            fig.update_layout(xaxis_title='Layer Number', yaxis_title='Normalized Value')
            figure_file_path = f"experiment_plots/{experiment_version}/{dnn_name}/final_epoch_vs_layer.html"
            print(f"Saving figure to {figure_file_path}")
            fig.write_html(figure_file_path)

  ##################################
 ####### side by side plots #######
################################## 

for experiment_version in experiment_versions:
    full_model_path = f"experiment_plots/{experiment_version}/dnn_experiments_results.pth"

    print(f"loading model from {full_model_path}")
    dnn_experiments = torch.load(full_model_path, weights_only=False)

    
    for dnn_name in dnn_names:
        
        model_layer_info = dnn_experiments[dnn_name].model_layer_info
        number_of_layers = len(model_layer_info.keys())

        subplot_titles = []
        for layer_name in model_layer_info.keys():
            for plot_type in ["Combined","Initial","Final"]:
                subplot_titles.append(f"{plot_type} {layer_name.replace('_',' ').capitalize()}")
        
        # fig_training_weights = make_subplots(rows=number_of_layers, cols=3, subplot_titles=subplot_titles)
        fig, axs = plt.subplots(nrows=number_of_layers, ncols=3, figsize=(10, 16))

        row = 0
        ## each layer goes on its own row, cols=0,1,2
        for layer_name, info in model_layer_info.items():
            initial_weights = info['initial_weights']
            final_weights = info['final_weights']

            _,S_init,_ = np.linalg.svd(initial_weights) 
            _,S_final,_ = np.linalg.svd(final_weights) 

            n_bins = 50
            axs[row, 0].hist(S_init, bins=n_bins, alpha=0.5, label="Initial", color='gold')
            axs[row, 0].hist(S_final, bins=n_bins, alpha=0.5, label="Final", color='royalblue')
            axs[row, 0].set_title(f"Combined {layer_name.replace('_',' ').capitalize()}")

            axs[row, 1].hist(S_init, bins=n_bins, alpha=0.5, label="Initial", color='gold')
            axs[row, 1].set_title(f"Initial {layer_name.replace('_',' ').capitalize()}")

            axs[row, 2].hist(S_final, bins=n_bins, alpha=0.5, label="Final", color='royalblue')
            axs[row, 2].set_title(f"Final {layer_name.replace('_',' ').capitalize()}")
            row += 1

        fig.suptitle('Initial and Final Singular Values')
        # plt.show()
        
        figure_file_path = f"experiment_plots/{experiment_version}/{dnn_name}/initial_and_final_singular_values.png"
        print(f"Saving figure to {figure_file_path}")
        plt.savefig(figure_file_path)
        plt.clf()
