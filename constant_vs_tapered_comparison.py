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

from dnn_weight_noise import DnnLayerWeightExperiment, DNN



dnn_names = ['dnn2a','dnn2c']

dnn_experiment_map = {
    'dnn2a': '8 layer constant width DNN = 1024, <br>uniform init',
    'dnn2b': '8 layer constant width DNN = 1024, <br>gaussian init',
    'dnn2c': '8 layer tapered width DNN, <br>uniform init',
    'dnn2d': '8 layer tapered width DNN, <br>gaussian init',
}
experiment_version_map = {
    'vNoNorm-CIFAR10-old': 'no norm',
}

_color_palette = px.colors.qualitative.Plotly
experiment_color_map = {
    key: _color_palette[i]
    for i,key in enumerate(dnn_experiment_map.keys())
}


fixed_noises = [0.5, 1.5, 2.5]
fig = make_subplots(rows=1, cols=len(fixed_noises), subplot_titles=[f"noise = {noise}" for noise in fixed_noises])

## set values that are guaranteed to be overwritten
y_min = 1.0
y_max = 0.0
for dnn_name in dnn_names:
    
    ## combine all experiments: no norm, batch norm, etc into one plot
    # dnn_noise_experiment_data = fixed_noise_experiments['vNoNorm-CIFAR10-fix'][dnn_name]
    # dnn_noise_experiment_data = pd.read_csv(data, sep='\s+')
    # print(dnn_noise_experiment_data)
    full_model_path = f"experiment_plots/vNoNorm-CIFAR10-fix/dnn_experiments_results.pth"
    dnn_experiments = torch.load(full_model_path, weights_only=False)
    dnn_noise_experiment_data = dnn_experiments[dnn_name].noise_experiment_data

    # print(experiment_version, dnn_name)
    # print(dnn_noise_experiment_data)
    layer_names = [f"layer_{i}" for i in range(1,9)]

    for i, noise in enumerate(fixed_noises, 1):
        accuracies_at_fixed_noise = dnn_noise_experiment_data.loc[dnn_noise_experiment_data['noise_vars'] == noise, layer_names].values.flatten()
        # print(f"at fixed noise of {noise}")
        print(accuracies_at_fixed_noise)
        if i == 1:
            showlegend=True
        else:
            showlegend=False
        fig.add_trace(go.Scatter(
            x=layer_names,
            y=accuracies_at_fixed_noise,
            name=dnn_experiment_map[dnn_name],
            marker=dict(color=experiment_color_map[dnn_name]),
            showlegend=showlegend,
        ),row=1, col=i)

        if min(accuracies_at_fixed_noise) < y_min:
            y_min = min(accuracies_at_fixed_noise)
        if max(accuracies_at_fixed_noise) > y_max:
            y_max = max(accuracies_at_fixed_noise)
        
y_min_final = np.floor(y_min * 100)/100.0
y_max_final = np.ceil(y_max * 100)/100.0

fig.update_layout(
    title=f"Constant width vs Tapered width for DNN (no normalization): test accuracy vs layer at fixed noises",
    yaxis1 = dict(range=[y_min_final,y_max_final]),
    yaxis2 = dict(range=[y_min_final,y_max_final]), 
    yaxis3 = dict(range=[y_min_final,y_max_final]),             
)
fig.show()