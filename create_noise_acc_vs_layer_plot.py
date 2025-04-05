from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dnn_weight_noise import DnnLayerWeightExperiment, DNN

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

import plotly.graph_objects as go

## all of these are now fixed!
## need to repeat this for MNIST experiments
experiment_versions = ['vNoNorm-CIFAR10','vBATCH-CIFAR10','vLAYER-CIFAR10','vWEIGHT-CIFAR10']
# experiment_versions = ['vNoNorm-MNIST-128','vBATCH-MNIST-128','vLAYER-MNIST-128','vWEIGHT-MNIST-128']
dnn_names = ['dnn2a','dnn2b','dnn2c','dnn2d']

for experiment_version in experiment_versions:
    
    full_model_path = f"experiment_plots/{experiment_version}/dnn_experiments_results.pth"
    print(f"loading model from {full_model_path}")
    dnn_experiments = torch.load(full_model_path, weights_only=False)

    for dnn_name in dnn_names:

        noise_acc_vs_layer_file = Path(f"experiment_plots/{experiment_version}/{dnn_name}/noise_acc_vs_layer.html")
        
        if noise_acc_vs_layer_file.exists():
            print(f"noise_acc_vs_layer plot already exists for {experiment_version}, {dnn_name}... skipping!")
            continue
        else:
            
            df_accuracy_vs_noise = dnn_experiments[dnn_name].noise_experiment_data
            layer_names = [f"layer_{i}" for i in range(1,9)]
            fig_noise_acc_vs_layer = go.Figure()

            df_layer_accuracy_vs_noise = df_accuracy_vs_noise.set_index('noise_vars').T

            # print(f"at fixed noise of {noise}")
            # print(accuracies_at_fixed_noise)
            
            print(f"{experiment_version}, {dnn_name}")
            print(df_layer_accuracy_vs_noise)

            self_temp = dnn_experiments[dnn_name]

            for noise_var in df_layer_accuracy_vs_noise.columns:
                fig_noise_acc_vs_layer.add_trace(go.Scatter(
                    x=df_layer_accuracy_vs_noise.index.tolist(),
                    y=df_layer_accuracy_vs_noise[noise_var],
                    name=f"noise = {noise_var}",
                    marker=dict(color=self_temp._noise_color_map[f"noise_var_{noise_var}"]),
                ))
            try:
                fig_noise_acc_vs_layer.update_layout(title=f"""DNN with {self_temp.model._N_LAYERS} layers + layer widths of {self_temp.model._HIDDEN_LAYER_WIDTHS} with training accuracy = {self_temp.train_accuracy:.2%}
                                    <br>Test Accuracy vs Layer by (layer variance normalized) Noise ({self_temp.dataset_name}, seed = {self_temp.noise_random_seed})""")
            except:
                fig_noise_acc_vs_layer.update_layout(title=f"""DNN with 8 layers and training accuracy = {self_temp.train_accuracy:.2%}
                                    <br>Test Accuracy vs Layer by (layer variance normalized) Noise ({self_temp.dataset_name}, seed = {self_temp.noise_random_seed})""")

            # fig_noise_acc_vs_layer.show()
            fig_noise_acc_vs_layer.write_html(noise_acc_vs_layer_file)
            print(f"saved {noise_acc_vs_layer_file}")


