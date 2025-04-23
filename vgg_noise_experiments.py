import argparse
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from cifar10_models.vgg import vgg11_bn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pretrained VGG11_bn weight noise experiments")
    parser.add_argument("--debug-mode", type=str, help="Debug Mode or actual experiment", choices=["debug","experiment"], required=True)
    args = parser.parse_args()

    debug = True if args.debug_mode == "debug" else False
    np.random.seed(42)

    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = vgg11_bn(pretrained=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), 
            (0.2023, 0.1994, 0.2010)
        )
    ])

    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    if debug:
        num_workers = 0
    else:
        num_workers = 2
    
    testloader = DataLoader(test_data, batch_size=128, shuffle=True, num_workers=num_workers)

    loss_function = nn.CrossEntropyLoss()

    noise_vars=[0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    color_palette = px.colors.qualitative.Plotly
    noise_color_map = {
        f"noise_var_{x}": color_palette[i-1]
        for i,x in enumerate(noise_vars)
    }

    layer_color_map = {
        f"layer_{i}": color_palette[i-1]
        for i in range(1,8+1)
    }

    if debug:
        N_NOISE_SAMPLES=2
    else:
        N_NOISE_SAMPLES=50
    
    noise_types=["layer_variance"]

    conv_weight_layer_names = []
    model_layer_stdev_info = {}

    ## store results
    all_layer_noise_test_acc = {}
    all_layer_noise_test_acc['noise_vars'] = noise_vars

    # >>> model.state_dict().keys()
    # ['features.0.weight', 'features.0.bias', 
    # 'features.1.weight', 'features.1.bias', 
    # 'features.1.running_mean', 'features.1.running_var', 
    # 'features.1.num_batches_tracked', 
    # 'features.4.weight', 'features.4.bias', 
    # 'features.5.weight', 'features.5.bias', 
    # 'features.5.running_mean', 'features.5.running_var', 
    # 'features.5.num_batches_tracked', 
    # 'features.8.weight', 'features.8.bias', 
    # 'features.9.weight', 'features.9.bias', 'features.9.running_mean', 'features.9.running_var', 'features.9.num_batches_tracked', 'features.11.weight', 'features.11.bias', 'features.12.weight', 'features.12.bias', 'features.12.running_mean', 'features.12.running_var', 'features.12.num_batches_tracked', 'features.15.weight', 'features.15.bias', 'features.16.weight', 'features.16.bias', 'features.16.running_mean', 'features.16.running_var', 'features.16.num_batches_tracked', 'features.18.weight', 'features.18.bias', 'features.19.weight', 'features.19.bias', 'features.19.running_mean', 'features.19.running_var', 'features.19.num_batches_tracked', 'features.22.weight', 'features.22.bias', 'features.23.weight', 'features.23.bias', 'features.23.running_mean', 'features.23.running_var', 'features.23.num_batches_tracked', 'features.25.weight', 'features.25.bias', 'features.26.weight', 'features.26.bias', 'features.26.running_mean', 'features.26.running_var', 'features.26.num_batches_tracked', 'classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias'])

    # n = 0
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.modules.Conv2d):
    #         conv_layer_names.append(name)

    #         weight_layer_string = f"features.{n}.weight"
    #         bias_layer_string = f"features.{n}.bias"

    #         weight_layer_stdev = torch.std(model.state_dict()[weight_layer_string]).item()
    #         bias_layer_stdev = torch.std(model.state_dict()[bias_layer_string]).item()

    #         model_layer_stdev_info[f"features.{n}.weight"] = weight_layer_stdev
    #         model_layer_stdev_info[f"features.{n}.bias"] = bias_layer_stdev

    #     n += 1

    n = 0
    for layer_name in model.state_dict().keys():
        if ("features" in layer_name) and ("weight" in layer_name):
            weight_layer_string = layer_name
            bias_layer_string = weight_layer_string.replace("weight","bias")
            
            base_layer_name = weight_layer_string.split(".weight")[0]
            num_occurrences = len([layer for layer in model.state_dict() if base_layer_name in layer])
            if num_occurrences == 2:

                conv_weight_layer_names.append(weight_layer_string)

                weight_layer_stdev = torch.std(model.state_dict()[weight_layer_string]).item()
                bias_layer_stdev = torch.std(model.state_dict()[bias_layer_string]).item()

                model_layer_stdev_info[f"layer_{n+1}_weight_std"] = weight_layer_stdev
                model_layer_stdev_info[f"layer_{n+1}_bias_std"] = bias_layer_stdev
                n += 1
            else:
                pass

    n = 0
    for name, module in model.named_modules():
        ## for a particular layer (n_stack), keep track of the test_accuracy vs noise
        ## for figures and layers, we use n to refer to layer 1, ... layer N of the cnn

        if isinstance(module, nn.Conv2d):
            for noise_type in noise_types:
                layer_avg_test_accs = []
                
                for noise_var in noise_vars: 
                    test_acc_sum = 0

                    for _ in range(N_NOISE_SAMPLES):

                        correct = 0
                        total = 0   

                        with torch.no_grad():
                            # weight_layer_string = f"features.{n}.weight"
                            # bias_layer_string = f"conv_layers.{n}.bias"
                            
                            weight_layer_string = conv_weight_layer_names[n]

                            bias_layer_string = weight_layer_string.replace("weight","bias")

                            print(f"adding noise (unscaled value of {noise_var}) to {weight_layer_string} weights")

                            weight_layer_size = model.state_dict()[weight_layer_string].data.shape
                            bias_layer_size = model.state_dict()[bias_layer_string].data.shape

                            weight_layer_stdev = model_layer_stdev_info[f"layer_{n+1}_weight_std"]
                            weight_noise_var = weight_layer_stdev**2 * noise_var

                            bias_layer_stdev = model_layer_stdev_info[f"layer_{n+1}_bias_std"]
                            bias_noise_var = weight_layer_stdev**2 * noise_var

                            conv_layer = dict(model.named_modules())[name]

                            weight_noise = torch.normal(mean=0, std=np.sqrt(weight_noise_var), size=weight_layer_size).to(device)
                            bias_noise = torch.normal(mean=0, std=np.sqrt(bias_noise_var), size=bias_layer_size).to(device)

                            # model.state_dict()[weight_layer_string].data += weight_noise
                            # model.state_dict()[bias_layer_string].data += bias_noise
                            conv_layer.weight.add_(weight_noise)
                            conv_layer.bias.add_(bias_noise)

                            for images, labels in testloader:
                                if debug:
                                    correct = np.random.randint(0,10)
                                    total = 10
                                else:
                                    images = images.to(device)
                                    labels = labels.to(device)
                                    outputs = model(images)

                                    loss = loss_function(outputs, labels)

                                    # test_loss += loss.item()

                                    _, predicted = torch.max(outputs.data, 1)
                                    total += labels.size(0)
                                    correct += (predicted == labels).sum().item()
                            
                            conv_layer.weight.sub_(weight_noise)
                            conv_layer.bias.sub_(bias_noise)

                        test_acc = correct / total
                        # print(f"test_acc for (unscaled) noise {noise_var} = {test_acc}")

                        test_acc_sum += test_acc

                    avg_test_acc = test_acc_sum/N_NOISE_SAMPLES
                    print(f"avg test acc = {avg_test_acc} for {weight_layer_string}, noise var = {noise_var}")
            
                    layer_avg_test_accs.append(avg_test_acc)

            all_layer_noise_test_acc[f'layer_{n+1}'] = np.array(layer_avg_test_accs)
            n += 1
        else:
            pass

    ####################
    ## visualizations ##
    ####################

    row = 1
    df_accuracy_vs_noise = pd.DataFrame(all_layer_noise_test_acc)
    df_accuracy_vs_noise = df_accuracy_vs_noise.sort_values(by='noise_vars', ascending=True)
    print(df_accuracy_vs_noise)

    layer_cols = [col for col in df_accuracy_vs_noise.columns if "layer" in col]

    fig_noise_vs_accuracy = go.Figure()

    for layer_col in list(layer_cols):
        text_x = df_accuracy_vs_noise['noise_vars'].iloc[-1]
        text_y = df_accuracy_vs_noise[layer_col].iloc[-1]

        print(f"{layer_col}: {text_x}, {text_y}")

        fig_noise_vs_accuracy.add_annotation(
            xref="x",yref="y",
            x=text_x*1.05, y=text_y, showarrow=False,
            text=layer_col,
        )
        fig_noise_vs_accuracy.add_trace(
            go.Scatter(
                x=df_accuracy_vs_noise['noise_vars'],
                y=df_accuracy_vs_noise[layer_col],
                mode='lines+markers',
                marker=dict(color=layer_color_map[layer_col]),
                name=layer_col,
            )
        )

    fig_noise_vs_accuracy.update_traces(mode='lines+markers', opacity=0.8) 
    fig_noise_vs_accuracy.update_layout(title="Pretrained VGG11 Test Accuracy vs Noise by Layer", xaxis_title='noise vars (unscaled)', height=1200)

    df_layer_accuracy_vs_noise = df_accuracy_vs_noise.set_index('noise_vars').T
    print(df_layer_accuracy_vs_noise)
    fig_noise_acc_vs_layer = go.Figure()

    for noise_var in df_layer_accuracy_vs_noise.columns:
        fig_noise_acc_vs_layer.add_trace(go.Scatter(
            x=df_layer_accuracy_vs_noise.index.tolist(),
            y=df_layer_accuracy_vs_noise[noise_var],
            name=f"noise = {noise_var}",
            marker=dict(color=noise_color_map[f"noise_var_{noise_var}"]),
        ))
    fig_noise_acc_vs_layer.update_layout(title="Pretrained VGG11 Test Accuracy vs Layer by Noise", xaxis_title='noise vars (unscaled)', height=1200)

    if debug:
        folder = 'pretrained_VGG11_bn_TEST'
    else:
        folder = 'pretrained_VGG11_bn'
    
    fig_noise_vs_accuracy.write_html(f"experiment_plots/{folder}/fig_noise_vs_accuracy.html")
    fig_noise_acc_vs_layer.write_html(f"experiment_plots/{folder}/noise_acc_vs_layer.html")
