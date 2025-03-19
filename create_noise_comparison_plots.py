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

from dnn_weight_noise import DnnLayerWeightExperiment, DNN

# device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

experiment_versions = ['vNoNorm-CIFAR10-old','vBATCH-CIFAR10','vLAYER-CIFAR10','vWEIGHT-CIFAR10']
dnn_names = ['dnn2a','dnn2b','dnn2c','dnn2d']

## change these later
dnn_experiment_map = {
    'dnn2a': '8 layer constant width DNN = 1024, uniform init',
    'dnn2b': '8 layer constant width DNN = 1024, gaussian init',
    'dnn2c': '8 layer tapered width DNN, uniform init',
    'dnn2d': '8 layer tapered width DNN, gaussian init',
}
experiment_version_map = {
    'vNoNorm-CIFAR10-old': 'no norm',
    'vBATCH-CIFAR10': 'batch norm',
    'vLAYER-CIFAR10': 'layer norm',
    'vWEIGHT-CIFAR10': 'weight norm',
}

_color_palette = px.colors.qualitative.Plotly
experiment_color_map = {
    key: _color_palette[i]
    for i,key in enumerate(experiment_version_map.keys())
}

old_noise_experiments = {
    'vNoNorm-CIFAR10-old': {
        'dnn2a':{
            'variance':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
0.1	0.56936	0.566996	0.567417	0.568754	0.569869	0.570288	0.570337	0.570309
0.5	0.563925	0.550788	0.552044	0.56013	0.566256	0.568957	0.570297	0.569007
1	0.556053	0.529175	0.530234	0.548351	0.560688	0.567554	0.569786	0.566507
1.5	0.54817	0.504271	0.506603	0.533172	0.554224	0.565641	0.569231	0.565427
2	0.540021	0.479744	0.48401	0.518721	0.547987	0.563428	0.569046	0.563001
2.5	0.531556	0.455687	0.458002	0.502388	0.539693	0.561269	0.568477	0.560512"""),
        },
        'dnn2b':{
            'variance':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8
0.1	0.569507	0.567571	0.56966	0.570321	0.570229	0.570194	0.570274	0.570265
0.5	0.563787	0.555888	0.563846	0.569122	0.570727	0.570359	0.570406	0.56975
1	0.558005	0.538883	0.557391	0.567464	0.570792	0.570529	0.570525	0.568419
1.5	0.551364	0.519324	0.547953	0.565513	0.570406	0.570522	0.570589	0.567282
2	0.543982	0.501467	0.539818	0.563069	0.569929	0.570879	0.570588	0.566407
2.5	0.537422	0.482292	0.531675	0.561288	0.569449	0.570703	0.570705	0.565016"""),
        },
        'dnn2c':{
            'variance':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8	layer_9
0.1	0.56151	0.55821	0.556455	0.555474	0.555741	0.557119	0.559003	0.560891	0.560745
0.5	0.554872	0.536436	0.52966	0.524478	0.528201	0.536411	0.543975	0.554467	0.552409
1	0.546882	0.509216	0.496929	0.489318	0.496486	0.507271	0.525489	0.545995	0.54188
1.5	0.538399	0.483591	0.469796	0.456738	0.465734	0.481083	0.506148	0.536721	0.53057
2	0.527907	0.45671	0.442049	0.42244	0.438946	0.45746	0.487794	0.525341	0.516979
2.5	0.519935	0.432758	0.417785	0.397756	0.413165	0.436699	0.470186	0.517168	0.508114"""),
        },
        'dnn2d':{
            'variance':StringIO("""noise_vars	layer_1	layer_2	layer_3	layer_4	layer_5	layer_6	layer_7	layer_8	layer_9
0.1	0.560232	0.555726	0.554136	0.552794	0.553587	0.555161	0.557996	0.55986	0.559719
0.5	0.553702	0.534769	0.528062	0.520603	0.523272	0.531421	0.544009	0.552361	0.551777
1	0.543952	0.50895	0.496513	0.478256	0.484748	0.498077	0.524112	0.543904	0.542009
1.5	0.535152	0.484452	0.464662	0.437508	0.45146	0.470022	0.503922	0.535091	0.530149
2	0.524355	0.462814	0.440988	0.404715	0.424012	0.43882	0.482805	0.522839	0.515336
2.5	0.517119	0.440164	0.418807	0.375599	0.39551	0.418296	0.460313	0.51389	0.502425"""),
        },
    },
}

## Create plots comparing test accuracies between different DNN experiments
## fix the noise, layer 1.... layer 8 on the xaxis, 
## pull information from model_layer_info 

fixed_noises = [0.5, 1.5, 2.5]
for dnn_name in dnn_names:
    
    fig = make_subplots(rows=1, cols=len(fixed_noises), subplot_titles=[f"noise = {noise}" for noise in fixed_noises])

    ## combine all experiments: no norm, batch norm, etc into one plot
    for experiment_version in experiment_versions:
        if experiment_version == 'vNoNorm-CIFAR10-old':
            data = old_noise_experiments['vNoNorm-CIFAR10-old'][dnn_name]['variance']
            dnn_noise_experiment_data = pd.read_csv(data, sep='\s+')
            print(dnn_noise_experiment_data)
            # raise Exception
        else:
            full_model_path = f"experiment_plots/{experiment_version}/dnn_all_experiments_results.pth"
            print(f"loading model from {full_model_path}")

            dnn_experiments = torch.load(full_model_path, weights_only=False)
            dnn_noise_experiment_data = dnn_experiments[dnn_name].noise_experiment_data

        # print(experiment_version, dnn_name)
        # print(dnn_noise_experiment_data)
        layer_names = [f"layer_{i}" for i in range(1,9)]

        for i, noise in enumerate(fixed_noises, 1):
            accuracies_at_fixed_noise = dnn_noise_experiment_data.loc[dnn_noise_experiment_data['noise_vars'] == noise, layer_names].values.flatten()
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
        yaxis1 = dict(range=[0.33,0.58]),
        yaxis2 = dict(range=[0.33,0.58]), 
        yaxis3 = dict(range=[0.33,0.58]),             
    )
    fig.show()