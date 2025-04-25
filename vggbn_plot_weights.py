import argparse
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from cifar10_models.vgg import vgg11_bn

model = vgg11_bn(pretrained=True)

_color_palette = px.colors.qualitative.Dark24

_layer_color_map = {
    f"layer_{i}": _color_palette[i-1]
    for i in range(1,8+1)
}

weights_by_layer = {}
n = 1
for layer_name in model.state_dict().keys():
    if ("features" in layer_name) and ("weight" in layer_name):
        weight_layer_string = layer_name
        bias_layer_string = weight_layer_string.replace("weight","bias")
        
        base_layer_name = weight_layer_string.split(".weight")[0]
        num_occurrences = len([layer for layer in model.state_dict() if base_layer_name in layer])
        if num_occurrences == 2:

            weights_layer = model.state_dict()[weight_layer_string]
            weights_by_layer[f"layer_{n}"] = {}
            weights_by_layer[f"layer_{n}"]['weights'] = weights_layer.flatten().numpy()
            weights_by_layer[f"layer_{n}"]['weights_mean'] = torch.mean(weights_layer.flatten()).item()
            weights_by_layer[f"layer_{n}"]['weights_std'] = torch.std(weights_layer.flatten()).item()
            n += 1
        else:
            pass

nrows = int(np.ceil(len(weights_by_layer)/2))
print(f"nrows = {nrows}")
fig = make_subplots(rows=nrows, cols=2, subplot_titles=tuple([f"layer_{i}" for i in range(1,8+1)]))

n = 1
for row in range(1,nrows+1):
    for col in [1,2]:
        weights_flat = weights_by_layer[f"layer_{n}"]['weights']

        n_bins = 100

        x_min, x_max = np.min(weights_flat), np.max(weights_flat)
        bin_values = np.linspace(x_min, x_max, n_bins)
        bins_for_barchart = 0.5 * (bin_values[:-1] + bin_values[1:])

        counts, bins = np.histogram(weights_flat, bins=bin_values)

        fig.append_trace(go.Bar(
            x=bins_for_barchart, y=counts, offset=0, showlegend=False, 
            marker=dict(color=_layer_color_map[f"layer_{n}"]), name=f"layer_{n}"
        ), row=row, col=col)

        
        weights_mean = weights_by_layer[f"layer_{n}"]['weights_mean']
        weights_std = weights_by_layer[f"layer_{n}"]['weights_std']
        fig.add_annotation(
            xref="x domain",yref="y domain",
            x=0.1, y=0.9, showarrow=False,
            text=f"mean={weights_mean:.5f} <br>stdev={weights_std:.5f}",
            bgcolor="rgba(255, 255, 255, 0.9)",
            row=row, col=col,
        )
        n+=1
    
fig.update_layout(title="VGG11_bn final weight distribution")
fig.show()