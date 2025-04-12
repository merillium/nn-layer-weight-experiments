import numpy as np
import pandas as pd
import re
from io import StringIO

import torch
from torch import nn
import torch.nn.functional as F

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

from dnn_weight_noise import DnnLayerWeightExperiment, DNN

# device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

experiment_versions = ['vNoNorm-CIFAR10-fix','vBATCH-CIFAR10','vLAYER-CIFAR10','vWEIGHT-CIFAR10']
# experiment_versions = ['vNoNorm-MNIST-128-fix','vBATCH-MNIST-128','vLAYER-MNIST-128','vWEIGHT-MNIST-128']
dnn_names = ['dnn2a','dnn2b','dnn2c','dnn2d']

## change these later
dnn_experiment_map = {
    'dnn2a': '8 layer constant width DNN = 1024, uniform init',
    'dnn2b': '8 layer constant width DNN = 1024, gaussian init',
    'dnn2c': '8 layer tapered width DNN, uniform init',
    'dnn2d': '8 layer tapered width DNN, gaussian init',
}

experiment_version_map = {
    'vNoNorm-CIFAR10-fix': 'no norm',
    'vBATCH-CIFAR10': 'batch norm',
    'vLAYER-CIFAR10': 'layer norm',
    'vWEIGHT-CIFAR10': 'weight norm',
}

# experiment_version_map = {
#     'vNoNorm-MNIST-128-fix': 'no norm',
#     'vBATCH-MNIST-128': 'batch norm',
#     'vLAYER-MNIST-128': 'layer norm',
#     'vWEIGHT-MNIST-128': 'weight norm',
# }

_color_palette = px.colors.qualitative.Plotly
experiment_color_map = {
    key: _color_palette[i]
    for i,key in enumerate(experiment_version_map.keys())
}

## Create plots comparing test accuracies between different DNN experiments
## fix the noise, layer 1.... layer 8 on the xaxis, 
## pull information from model_layer_info 

fixed_noises = [0.5, 1.5, 2.5]
for dnn_name in dnn_names:
    
    fig = make_subplots(rows=1, cols=len(fixed_noises), subplot_titles=[f"noise = {noise}" for noise in fixed_noises])
    y_min = 1.00
    y_max = 0.00
    
    ## combine all experiments: no norm, batch norm, etc into one plot
    for experiment_version in experiment_versions:
        if experiment_version == 'vNoNorm-CIFAR10-fix':
            # dnn_noise_experiment_data = fixed_noise_experiments['vNoNorm-CIFAR10-fix'][dnn_name]
            full_model_path = f"experiment_plots/{experiment_version}/dnn_experiments_results.pth"
            # dnn_noise_experiment_data = pd.read_csv(data, sep='\s+')
            # print(dnn_noise_experiment_data)
            # raise Exception
        else:
            # full_model_path = f"experiment_plots/{experiment_version}/all_experiments_results.pth"

            full_model_path = f"experiment_plots/{experiment_version}/dnn_all_experiments_results.pth"
        
        print(f"loading model from {full_model_path}")
        dnn_experiments = torch.load(full_model_path, weights_only=False)
        dnn_noise_experiment_data = dnn_experiments[dnn_name].noise_experiment_data

        # print(experiment_version, dnn_name)
        # print(dnn_noise_experiment_data)
        layer_names = [f"layer_{i}" for i in range(1,9)]

        for i, noise in enumerate(fixed_noises, 1):
            accuracies_at_fixed_noise = dnn_noise_experiment_data.loc[dnn_noise_experiment_data['noise_vars'] == noise, layer_names].values.flatten()
            if accuracies_at_fixed_noise.min() < y_min:
                y_min = accuracies_at_fixed_noise.min()
            if accuracies_at_fixed_noise.max() > y_max:
                y_max = accuracies_at_fixed_noise.max()
            # print(f"at fixed noise of {noise}")
            # print(accuracies_at_fixed_noise)
            if i == 1:
                showlegend=True
            else:
                showlegend=False
            fig.add_trace(go.Scatter(
                x=layer_names,
                y=accuracies_at_fixed_noise,
                name=experiment_version_map[experiment_version],
                marker=dict(color=experiment_color_map[experiment_version]),
                showlegend=showlegend,
            ),row=1, col=i)
    
    fig_dnn_name = dnn_experiment_map[dnn_name]
    fig.update_layout(
        title=f"all experiments comparison {fig_dnn_name}: test accuracy vs layer at fixed noises",
        yaxis1 = dict(range=[y_min,y_max]),
        yaxis2 = dict(range=[y_min,y_max]), 
        yaxis3 = dict(range=[y_min,y_max]),             
    )
    fig.show()