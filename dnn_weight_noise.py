import argparse
import copy
import os
import time
import numpy as np
import pandas as pd
import re

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from utils import fit_gaussian_curve, init_weights, equalize_axes_layout

# import detectors
# import timm

# from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

## torch info
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

# from torchvision.models import vgg16

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

class DNN(nn.Module):
    def __init__(self, N_CLASSES, HIDDEN_LAYER_WIDTHS, random_seed, init_type):
        super().__init__()
        self._N_CLASSES = N_CLASSES

        self._HIDDEN_LAYER_WIDTHS = HIDDEN_LAYER_WIDTHS
        self._N_HIDDEN_LAYERS = len(self._HIDDEN_LAYER_WIDTHS) 
        self._N_LAYERS = self._N_HIDDEN_LAYERS+1 # hidden layers + 1 output layer

        self.random_seed = random_seed
        self.init_type = init_type

        if device=="mps":
            torch.mps.manual_seed(self.random_seed)
        if device=="cuda":
            torch.cuda.manual_seed(self.random_seed)

        ## NOTE: ideally we should parametrize 28 x 28
        ## and construct the layers a more dynamic fashion
        hidden_layers = []
        for i, layer_width in enumerate(self._HIDDEN_LAYER_WIDTHS):
            if i == 0:
                first_hidden_layer = nn.Linear(28 * 28, layer_width)
                hidden_layers.append(first_hidden_layer)
            else:
                previous_layer_width = self._HIDDEN_LAYER_WIDTHS[i-1]
                hidden_layers.append(nn.Linear(previous_layer_width, layer_width))
        output_layer = [nn.Linear(self._HIDDEN_LAYER_WIDTHS[-1], self._N_CLASSES)]
        all_layers = hidden_layers + output_layer

        self.linear_relu_stack = nn.Sequential(*all_layers)
        self.initialize_weights()
    
    def initialize_weights(self, new_random_seed=None):
        """Convenience method to initialize or re-initialize weights"""
        if new_random_seed is not None:
            self.random_seed = new_random_seed
        self.linear_relu_stack.apply(lambda m: init_weights(m, self.init_type, self.random_seed))
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.linear_relu_stack(x)
        return x

class DnnLayerWeightExperiment():

    def __init__(self, model, dataset_name, noise_random_seed, preloaded=False):
        """Most attributes are initialized to None or empty dicts, and populated later
        random_seed: int for adding reproducible Gaussian noise following true variance = variance / (previous) layer width
        """
        
        ## data loading instance attributes
        self.dataset_name = dataset_name
        self.noise_random_seed = noise_random_seed

        ## model instance
        self.base_model = copy.deepcopy(model)
        device = next(model.parameters()).device  
        self.base_model = self.base_model.to(device)
        self.model = model
        self._N_EPOCHS = None

        self.max_weights_std = None

        self.preloaded = preloaded

        ## GUARANTEE THE SAME COLOR SCHEME ACROSS PLOTS
        if not self.preloaded:
            self._color_palette = px.colors.qualitative.Plotly
            self._layer_color_map = {
                f"layer_{i}": self._color_palette[i-1]
                for i in range(1,self.model._N_LAYERS+1)
            }
        else:
            ## initialize it later
            self._color_palette = px.colors.qualitative.Dark24
            self._layer_color_map = {}

        if self.preloaded:
            self.model_layer_info = {}
        
        else:
            self.model_layer_info = {
                f'layer_{i}': {
                    'initial_weights': None,
                    'initial_biases': None,
                    'final_weights': None,
                    'final_biases': None,
                    'weight_std_by_epoch': [],
                    'cond_number_by_epoch': []
                } for i in range(1,self.model._N_LAYERS+1)
            }

            n = 1
            for name, layer in self.model.named_parameters():
                if 'weight' in name:
                    self.model_layer_info[f'layer_{n}']['initial_weights'] = layer.detach().cpu().numpy()
                
                ## weights come before biases
                if 'bias' in name:
                    self.model_layer_info[f'layer_{n}']['initial_biases'] = layer.detach().cpu().numpy()
                    n += 1
            n = 1

        # model output instance attributes
        self.train_accuracies = []
        self.test_accuracies = []

        self.layer_summary_stats = {}
        self.all_layer_noise_test_acc = {}
        self.layer_condition_numbers = {}

        self.all_figures = {
            'initial_weights': None,
            'final_weights': None,
            'condition_numbers_by_epoch': None,
            'weight_stds_by_epoch': None,
            'accuracies_by_epoch': None,
            'noise_test_accuracies': None,
            'condition_numbers_by_layer': None
        }
    
    # Define the transformations for the data
    def load_dataset(self):
        # Download and load the training and test datasets
        if self.dataset_name == 'fashion_mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            self.train_data = datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
            self.test_data = datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
        elif self.dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            self.train_data = datasets.MNIST('./data', train=True, transform=transform, download=True)
            self.test_data = datasets.MNIST('./data', train=False, transform=transform, download=True)
        
        ## this hasn't been tested yet
        elif self.dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize( 
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) 
                )
            ])
            # transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.5,), (0.5,))
            # ])
            self.train_data = datasets.CIFAR10('./data', train=True, transform=transform, download=True)
            self.test_data = datasets.CIFAR10('./data', train=False, transform=transform, download=True)
        else:
            raise Exception("Dataset {self.dataset_name} not supported")
        

    def train_base_model(self, N_EPOCHS=30, regularizer=None):
        self._N_EPOCHS = N_EPOCHS
        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=100, shuffle=True, num_workers=8)
        device = next(self.base_model.parameters()).device

        ## data is expected to be in batches
        ## we can add regularizer term to CrossEntropyLoss using parameter weight_decay
        ## OR we can implement it out-of-the-box (this is useful if we intend to modify it)
        ## construct l2_norm one layer at a time for future use
        
        l2_lambda = 10**-5
        l2_loss = torch.tensor(0.)
        loss_function = nn.CrossEntropyLoss()

        if regularizer == 'l2':
            for name, layer in self.model.named_parameters():
                if 'weight' in name:
                    l2_loss = l2_loss + torch.linalg.norm(layer, 2).detach() ** 2
        elif regularizer is None:
            pass
        else:
            raise Exception(f"Regularizer {regularizer} not implemented!")
        
        ## try different learning rates but keep track of their success
        ## this ensures we choose the BEST learning rate, not the first that succeeds

        # best_train_accuracy = 0.0
        best_test_accuracy = 0.0 
        best_model_state = None
        train_accuracies = []
        test_accuracies = []

        ## we need to calculate these for each epoch AND each learning rate
        ## otherwise these will not be stored 
        model_layer_info = {
            f'layer_{i}': {
                'initial_weights': self.model_layer_info[f'layer_{i}']['initial_weights'].copy(),
                'initial_biases': self.model_layer_info[f'layer_{i}']['initial_biases'].copy(),
                'final_weights': None,
                'final_biases': None,
                'weight_std_by_epoch': [],
                'cond_number_by_epoch': []
            } for i in range(1,self.model._N_LAYERS+1)
        }

        ## try learning rates [0.0001, 0.001, 0.01]
        ## maybe scale up later if needed for more experiments
        ## learning rates for adam?
        # lr_adam = [3*10**-4, 5*10**-4, 10**-3]
        lr_sgd = [0.0001, 0.001, 0.01]

        lr_decay_factor = 0.5

        ## detect a plateau (e.g. an average change of < 0.01 over a period of 20 epochs)
        plateau_threshold = 0.01
        plateau_epochs = 20
        
        for lr in lr_sgd:

            print(f"Trying learning rate = {lr}")

            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=l2_lambda, nesterov=True)

            ## optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=l2_lambda)
            # scheduler = ExponentialLR(optimizer, gamma=0.9)

            for epoch in range(N_EPOCHS):

                loss = 0.0
                correct = 0
                total = 0
                for images, labels in trainloader:
                    images=images.to(device)
                    labels=labels.to(device)
                    self.model.train()

                    optimizer.zero_grad()
                    outputs=self.model(images)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = loss_function(outputs, labels)

                    # + l2_lambda*l2_loss ## this is where we use the l2 regularizer
                    # but this is redundant if we include it in the optimizer itself

                    loss.backward()
                    optimizer.step()
                
                train_accuracy = correct / total
                train_accuracies.append(train_accuracy)

                training_loss = loss.item()
                if np.isnan(training_loss):
                    print(f"Training did not converge for learning rate = {lr}")
                    break

                print(f"Epoch {epoch}: training_loss = {training_loss}, train accuracy = {train_accuracy:.2%}")
                
                ## for each epoch, we need to get layer weight std and condition number

                n = 1
                for name, layer in self.model.named_parameters():
                    if "weight" in name:
                        # print(f"saving layer n = {n}")
                        layer_weights = layer.detach().cpu().numpy()
                        # print(layer_weights.shape)
                        try:
                            cond_number = np.linalg.cond(layer_weights)
                            if np.isnan(cond_number):
                                print("Warning: condition number is null!")

                            weight_std = np.std(layer_weights.flatten())

                            ## we store the condition number and weight std by epoch
                            ## and set this to the actual model's info if this epoch does the best
                            model_layer_info[f'layer_{n}']['cond_number_by_epoch'].append(cond_number)
                            model_layer_info[f'layer_{n}']['weight_std_by_epoch'].append(weight_std)

                            n += 1

                        except Exception as e:
                            print(f"Warning: {str(e)}")
                    else:
                        continue

                # print(f"epoch {epoch}: standard deviation for {name} = {weight_std}")

                ## calculate test accuracy for the epoch
                test_accuracy = self.get_test_accuracy()
                test_accuracies.append(test_accuracy)

                ## if plateauing is occurring, we need to change the learning rate
                if epoch % plateau_epochs == 0:
                    avg_epoch_change = np.mean(np.diff(np.array(test_accuracies[-plateau_epochs:])))
                    if avg_epoch_change < plateau_threshold:
                        lr = lr * lr_decay_factor
                        print(f"test accuracy has plateaued at epoch {epoch}, new learning rate = {lr}")
                        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=l2_lambda, nesterov=True)

            ## model training is completed for some learning rate, lr
            ## check if the accuracy is above a certain threshold AND better than previous accuracy
            if (train_accuracy <= 0.10) | (np.isnan(training_loss)):
                print(f"Training did not converge for learning rate = {lr}")

            else:
                print(f"Training successfully converged for learning rate = {lr}")

                ## when we improve on the current best test accuracy
                ## we set all of the model parameters equal the current state of the model

                ## NOTE: this could happen mid-experiment, so the model state will be reset
                ## when subsequent learning rates are tried

                if (test_accuracy > best_test_accuracy):
                    print(f"Better test accuracy = {test_accuracy} found for learning rate {lr}, overwriting model")
                    best_test_accuracy = test_accuracy # primitives are immutable! if test_accuracy changes, won't impact best_test_accuracy
                    best_model_state = copy.deepcopy(self.model.state_dict())

                    # print("setting self.model_layer_info to model_layer_info (shown below)")
                    self.model_layer_info = copy.deepcopy(model_layer_info)
                    # print("\n")
                    # print(self.model_layer_info)
                    # print("\n")
                    self.train_accuracy = train_accuracy # saves train accuracy corresponding to best test accuracy
                    self.train_accuracies = train_accuracies
                    self.test_accuracy = test_accuracy
                    self.test_accuracies = test_accuracies
                
            ## reset accuracies for the next iteration of learning parameter + optimizer
            ## we need to reset the model too!

            print("--- Resetting model experiment parameters! ---")
            train_accuracies = []
            test_accuracies = []
            model_layer_info = {
            f'layer_{i}': {
                'initial_weights': self.model_layer_info[f'layer_{i}']['initial_weights'],
                'initial_biases': self.model_layer_info[f'layer_{i}']['initial_biases'],
                'final_weights': None,
                'final_biases': None,
                'weight_std_by_epoch': [],
                'cond_number_by_epoch': []
            } for i in range(1,self.model._N_LAYERS+1)
        }

            self.model = copy.deepcopy(self.base_model)
            device = next(self.base_model.parameters()).device  # Get device of base model
            self.model = self.model.to(device) # move copied model to the same device

        ## once the best model is determined after trying all learning rates + optimizers
        ## save the best model state as the model's actual state permanently
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        else:
            raise Exception("No valid model was found! (no learning rates + optimizers converged)")
        
        ## then loop through layers of the model:
        ## extract the initial and final weights, and calculate summary stats of each layer's weights 
        ## these can be accessed later for data visualization, debugging, and analysis

        # print(self.model)
        # print("\n")
        # for name, layer in self.model.named_parameters():
        #     print(name)
        
        n = 1
        for name, layer in self.model.named_parameters():
            print(name)
            if 'weight' in name:
                layer_weights = layer.detach().cpu().numpy()
                print(f"setting layer {n} final weights to weights from {name}")
                self.model_layer_info[f'layer_{n}']['final_weights'] = layer_weights
        
            ## weights come before biases
            if 'bias' in name:
                print(f"setting bias {n} final weights to biases from {name}")
                self.model_layer_info[f'layer_{n}']['final_biases'] = layer_weights

                ## since there will always be a bias corresponding to weight, only increment after bias
                n += 1
        
        print("training base_model completed:")
        # print(self.model_layer_info)

    def get_test_accuracy(self) -> float:
        # Ensure model is in evaluation mode
        self.model.eval()

        # Get the device and DataLoader
        device = next(self.model.parameters()).device
        valloader = torch.utils.data.DataLoader(self.test_data, batch_size=100, shuffle=False, pin_memory=True)
        
        # Initialize counters
        correct = 0
        total = 0

        # Disable gradient computation for efficiency
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        # Calculate accuracy
        test_accuracy = correct / total
        return test_accuracy

    ############################################################
    ####### EXPERIMENT 1: distribution of layer weights ########
    ############################################################

    def create_layer_weight_plots(self):
        """
        Create histogram of initial and final layer weights with fitted + scaled Gaussian
            - If initial layer weights are uniform, then do not fit a Gaussian
        Create histogram of singular values for each layer weight matrix
        Create a heatmap of initial and layer weights [LEGACY]

        Create a line chart of condition number versus epoch
        Create a line chart of train + test accuracy versus epoch
        """
        if not self.preloaded:
            for weight_type in ['Initial','Final']:
                weight_type_key = f"{weight_type.lower()}_weights"

                fig_histogram = make_subplots(
                    rows=int(np.ceil(self.model._N_LAYERS/2)), cols=2,
                    subplot_titles=[f"{weight_type} Layer {i} weights" for i in range(1,self.model._N_LAYERS+1)]
                )

                ## HEATMAP OF WEIGHTS MATRIX

                fig_heatmap = make_subplots(
                    rows=int(np.ceil(self.model._N_LAYERS/2)), cols=2,
                    subplot_titles=[f"Layer {i} weights" for i in range(1,self.model._N_LAYERS+1)]
                )

                fig_singular_values = make_subplots(
                    rows=int(np.ceil(self.model._N_LAYERS/2)), cols=2,
                    subplot_titles=[f"Singular Values for Layer {i} weights" for i in range(1,self.model._N_LAYERS+1)],
                    vertical_spacing=0.05
                )
                
                ## for each layer, get the weight matrix and populate subplots
                ## for the histogram, heatmap, and scatter plot of singular values
                zmin, zmax = 0,0
                for i in range(1, self.model._N_LAYERS+1):
                    weights_matrix = self.model_layer_info[f"layer_{i}"][weight_type_key]

                    ## print(f"layer {i}")
                    ## print(weights_matrix)

                    if np.min(weights_matrix) < zmin:
                        zmin = np.min(weights_matrix)
                    if np.max(weights_matrix) > zmax:
                        zmax = np.max(weights_matrix)

                for i in range(1, self.model._N_LAYERS+1):
                    row = int(np.ceil(i/2))
                    if i % 2 == 1: ## the count starts at 1
                        col = 1
                    else:
                        col = 2
                    
                    ## HISTOGRAM OF FLATTENED WEIGHTS DISTRIBUTION
                    weights_matrix = self.model_layer_info[f"layer_{i}"][weight_type_key]
                    weights_flat = weights_matrix.flatten()
                    weights_mean = np.mean(weights_flat)
                    weights_std = np.std(weights_flat)

                    self.layer_summary_stats[f'layer_{i}_mean'] = weights_mean
                    self.layer_summary_stats[f'layer_{i}_std'] = weights_std

                    # print(f"plotting layer_{i} weights on row={row}, col={col}")

                    n_bins = 100

                    x_min, x_max = np.min(weights_flat), np.max(weights_flat)
                    bin_values = np.linspace(x_min, x_max, n_bins)
                    bins_for_barchart = 0.5 * (bin_values[:-1] + bin_values[1:])

                    counts, bins = np.histogram(weights_flat, bins=bin_values)

                    ## histogram color scheme
                    fig_histogram.append_trace(go.Bar(
                        x=bins_for_barchart, y=counts, offset=0, showlegend=False, 
                        marker=dict(color=self._layer_color_map[f"layer_{i}"]),
                    ), row=row, col=col)
                    
                    ## try to fit a Gaussian to the final weights, not needed for initial weights
                    if weight_type == 'Final':
                        
                        ## extract the midpoints of the bins
                        x_train = np.array([(bins[i]+bins[i+1])/2 for i in range(n_bins-1)])
                        y_train = counts 

                        fitted_gaussian = fit_gaussian_curve(x_train, y_train, weights_mean, weights_std)

                        ## draw a curve for the entire distribution
                        x_domain = np.linspace(x_min, x_max, n_bins)
                        
                        fig_histogram.append_trace(go.Scatter(
                            x=x_domain, y=fitted_gaussian(x_domain),
                            marker=dict(color='#8222fc'), showlegend=False
                        ), row=row, col=col)
                    
                    ## HEATMAP OF LAYER WEIGHTS
                    fig_heatmap.append_trace(go.Heatmap(z=weights_matrix, zmin=zmin, zmax=zmax), row=row, col=col)
                    
                    fig_histogram.add_annotation(
                        xref="x domain",yref="y domain",
                        x=0.1, y=0.9, showarrow=False,
                        text=f"mean={weights_mean:.5f} <br>stdev={weights_std:.5f}",
                        bgcolor="rgba(255, 255, 255, 0.9)",
                        row=row, col=col,
                    )

                    ### HISTOGRAM OF SINGULAR VALUES
                    try:
                        U,S,V = np.linalg.svd(weights_matrix) 

                        r = S.max()

                        fig_singular_values.append_trace(
                            go.Histogram(x=S, marker=dict(color=self._layer_color_map[f"layer_{i}"]), name=f"layer_{i}"),
                            row=row, col=col,
                        )

                        singular_values_mean = np.mean(S)
                        singular_values_std = np.std(S)

                        fig_singular_values.add_annotation(
                            xref="x domain",yref="y domain",
                            x=0.1, y=0.9, showarrow=False,
                            text=f"mean={singular_values_mean:.5f} <br>stdev={singular_values_std:.5f}",
                            bgcolor="rgba(255, 255, 255, 0.9)",
                            row=row, col=col,
                        )

                    except Exception as e:
                        print(f"Exception {e}! Creating other plots")

                ## add descriptive titles with summary stats
                final_train_accuracy = self.train_accuracies[-1]
                final_test_accuracy = self.test_accuracies[-1]
                fig_heatmap.update_layout(title=f"""
                                DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}: final train acc = {final_train_accuracy:.2%}, final test_acc = {final_test_accuracy:.2%}
                                <br>Heatmap of {weight_type} Weights by Layer ({self.dataset_name}, seed = {self.model.random_seed})<br>""",
                                margin=dict(t=120, l=50, r=50, b=50))
                
                fig_histogram.update_layout(title=f"""
                                DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}: final train acc = {final_train_accuracy:.2%}, final test_acc = {final_test_accuracy:.2%}
                                <br>Distribution of {weight_type} Weights by Layer ({self.dataset_name}, seed = {self.model.random_seed})<br>""",
                                margin=dict(t=120, l=50, r=50, b=50))
            
                fig_singular_values.update_layout(title=f"""
                                DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}: final train acc = {final_train_accuracy:.2%}, final test_acc = {final_test_accuracy:.2%}
                                <br>Singular Values of {weight_type} Weights by Layer ({self.dataset_name}, seed = {self.model.random_seed})<br>""",
                                margin=dict(t=120, l=50, r=50, b=50))

                fig_heatmap = equalize_axes_layout(fig_heatmap)
                # fig_singular_values = equalize_axes_layout(fig_singular_values)

                
                self.all_figures[weight_type_key] = {
                    'histogram': fig_histogram,
                    'heatmap': fig_heatmap,
                    'singular_values': fig_singular_values
                }
            
            fig_condition_numbers = make_subplots(
                rows=int(np.ceil(self.model._N_LAYERS/2)), cols=2,
                subplot_titles=[f"Layer {i}" for i in range(1,self.model._N_LAYERS+1)]
            )
            
            fig_weight_stds = make_subplots(
                rows=int(np.ceil(self.model._N_LAYERS/2)), cols=2,
                subplot_titles=[f"Layer {i}" for i in range(1,self.model._N_LAYERS+1)]
            )

            for i in range(1, self.model._N_LAYERS+1):
                row = int(np.ceil(i/2))
                if i % 2 == 1: ## the count starts at 1
                    col = 1
                else:
                    col = 2

                final_condition_number = self.model_layer_info[f'layer_{i}']['cond_number_by_epoch'][-1]
                
                ## condition numbers follow the same color scheme
                fig_condition_numbers.append_trace(
                    go.Scatter(
                        x=list(range(self._N_EPOCHS)), 
                        y=self.model_layer_info[f'layer_{i}']['cond_number_by_epoch'],
                        name=f'layer_{i}',
                        marker=dict(color=self._layer_color_map[f"layer_{i}"]),
                    ), row=row, col=col
                )
                fig_condition_numbers.update_yaxes(range=[0, final_condition_number*1.2], row=row, col=col)

                fig_weight_stds.append_trace(
                    go.Scatter(
                        x=list(range(self._N_EPOCHS)), 
                        y=self.model_layer_info[f'layer_{i}']['weight_std_by_epoch'],
                        marker=dict(color=self._layer_color_map[f"layer_{i}"]),
                        name=f"layer_{i}",
                    ), row=row, col=col
                )

                
            fig_condition_numbers.update_layout(title=f"""
                            DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}: final train acc = {final_train_accuracy:.2%}, final test_acc = {final_test_accuracy:.2%}
                            <br>Condition Number of Layer Weights by Epoch ({self.dataset_name}, seed = {self.model.random_seed})<br>""")
        
            fig_weight_stds.update_layout(title=f"""
                            DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}: final train acc = {final_train_accuracy:.2%}, final test_acc = {final_test_accuracy:.2%}
                            <br>Standard Deviation of Layer Weights by Epoch ({self.dataset_name}, seed = {self.model.random_seed})<br>""")
            
            fig_accuracies = go.Figure()
            fig_accuracies.add_trace(go.Scatter(x=list(range(self._N_EPOCHS)), y=self.train_accuracies, name='train accuracy'))
            fig_accuracies.add_trace(go.Scatter(x=list(range(self._N_EPOCHS)), y=self.test_accuracies, name='test accuracy'))
            fig_accuracies.update_layout(title=f"""
                            DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}: train acc = {final_train_accuracy:.2%}, final test_acc = {final_test_accuracy:.2%}
                            <br>Train and Test Accuracies by Epoch ({self.dataset_name}, seed = {self.model.random_seed})<br>""")
            
            self.all_figures['condition_numbers_by_epoch'] = fig_condition_numbers
            self.all_figures['weight_stds_by_epoch'] = fig_weight_stds
            self.all_figures['accuracies_by_epoch'] = fig_accuracies


        ################################################ 
        # FOR PRELOADED MODEL, ONLY PLOT FINAL WEIGHTS 
        ################################################

        # print('layer names:')
        # print([layer for layer in self.model.state_dict()])
        
        if self.preloaded:
            all_feature_layers = [layer for layer in self.model.state_dict() if "feature" in layer]
            count_dict = {int(layer.split('.')[1]): 0 for layer in all_feature_layers}
            for layer in all_feature_layers:
                # Extract the number from each layer name
                match = re.search(r'features\.(\d+)', layer)
                if match:
                    number = int(match.group(1))
                    count_dict[number] += 1
            print(count_dict)
            n_stack_layers = [k for k,v in count_dict.items() if v == 2]
            print(n_stack_layers)

            print(f"nrows = {int(np.ceil(len(n_stack_layers)/2))}")
            fig_histogram = make_subplots(
                rows=int(np.ceil(len(n_stack_layers)/2)), cols=2,
                subplot_titles=[f"Layer {i+1}" for i in n_stack_layers]
            )

            ## (1,1),(1,2),(2,1),(2,2),...
            ## (1,2),(2,1), ...
            for i, n_stack in enumerate(n_stack_layers):
                row = (i // 2) + 1
                col = (i % 2) + 1

                weight_weight_layer_string = f"features.{n_stack}.weight"
                layer_weights = self.model.state_dict()[weight_weight_layer_string].data.detach().cpu().numpy()

                # batchnorm_weight_layer_string = f"features.{n_stack}.BatchNorm2d"
                # batch_norm_weights = self.model.state_dict()[weight_weight_layer_string].data.detach().cpu().numpy()
                # print(f"batch_norm_weights have shape: {batch_norm_weights.shape}")
                
                ## setting color scheme for each layer
                self._layer_color_map[f"layer_{n_stack+1}"] = self._color_palette[i-1]

                weights_flat = layer_weights.reshape(-1)
                weights_mean = np.mean(weights_flat)
                weights_std = np.std(weights_flat)

                unscaled_noise_var = weights_std**2 * len(weights_flat)
                print(f"layer {weight_weight_layer_string} has total variance = {unscaled_noise_var}")

                ## print(f"layer {weight_layer_string} has shape {layer_weights.shape} with {len(weights_flat)} scalar weights")   

                ## track the maximum standard deviation of weights
                if self.max_weights_std is None:
                    self.max_weights_std = weights_std
                elif self.max_weights_std < weights_std:
                    self.max_weights_std = weights_std
                else:
                    pass

                print(f"plotting layer_{i} weights on row={row}, col={col}")

                n_bins = 100

                x_min, x_max = np.min(weights_flat), np.max(weights_flat)
                bin_values = np.linspace(x_min, x_max, n_bins)
                bins_for_barchart = 0.5 * (bin_values[:-1] + bin_values[1:])

                counts, bins = np.histogram(weights_flat, bins=bin_values)

                ## histogram color scheme
                fig_histogram.append_trace(go.Bar(
                    x=bins_for_barchart, y=counts, offset=0, showlegend=False, 
                    marker=dict(color=self._layer_color_map[f"layer_{n_stack+1}"]),
                ), row=row, col=col)

                fig_histogram.add_annotation(
                    xref="x domain",yref="y domain",
                    x=0.1, y=0.9, showarrow=False,
                    text=f"mean={weights_mean:.5f} <br>stdev={weights_std:.5f}",
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    row=row, col=col,
                )
            
            fig_histogram.update_layout(title=f"""Distribution of Weights by Layer for Pre-trained CNN""",
                                margin=dict(t=120, l=50, r=50, b=50))
            self.all_figures['fig_histogram'] = fig_histogram
            ## fig_histogram.show()

    ############################################################
    #####  EXPERIMENT 2: freeze layers and then add noise  #####
    ############################################################

    def create_models_with_noise(self, N_NOISE_SAMPLES=10):
        
        np.random.seed(self.noise_random_seed)
        device = next(self.model.parameters()).device

        ## STEPS:
        # (1) iterate over layer i and noise std dev
        # (2) generate N samples of this gaussian noise at fixed stddev
        # (3) for each 1...N noise sample, add it to an individual layer
        # (4) calculate the test accuracy with perturbed layer
        # (5) average the test accuracies over each layer for N samples of noise
        # (6) example: 
        #     layer 1 has average test accuracy of 0.80 with N = 100 samples of noise at stdev = 0.1
        #     layer 1 has average test accuracy of 0.70 with N = 100 samples of noise at stdev = 0.2,
        #     ...
        
        if not self.preloaded:
            noise_vars = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            self.all_layer_noise_test_acc["noise_vars"] = noise_vars
            n_stack_layers = [int(''.join(filter(str.isdigit, layer))) for layer in self.model.state_dict() if (("weight" in layer) and ("classifier" not in layer))]
            print(n_stack_layers)
        
        ## a preloaded CNN model has a different architecture
        else:
            # get standard deviation corresponding to the largest layer
            
            # noise_stds = np.linspace(0, self.max_weights_std, 6)

            noise_stds = np.linspace(0, 0.08, 6)
            noise_vars = noise_stds**2
            # noise_vars = [0.1, 0.4, 0.7, 1.0, 1.4, 1.7, 2.0]

            ## store the noise vars to be plotted (true noise vars may be scaled)
            self.all_layer_noise_test_acc["noise_vars"] = noise_vars
            
            all_feature_layers = [layer for layer in self.model.state_dict() if "feature" in layer]
            count_dict = {int(layer.split('.')[1]): 0 for layer in all_feature_layers}
            for layer in all_feature_layers:
                # Extract the number from each layer name
                match = re.search(r'features\.(\d+)', layer)
                if match:
                    number = int(match.group(1))
                    count_dict[number] += 1
            print(count_dict)
            n_stack_layers = [k for k,v in count_dict.items() if v == 2]
            print(n_stack_layers)

            for i, n_stack in enumerate(n_stack_layers):
                weight_layer_string = f"features.{n_stack}.weight"
                layer_weights = self.model.state_dict()[weight_layer_string].data.detach().cpu().numpy()
                
                ## setting color scheme for each layer
                self._layer_color_map[f"layer_{n_stack+1}"] = self._color_palette[i-1]

                print(f"layer {weight_layer_string} as shape {layer_weights.shape}")

        for n_stack in n_stack_layers:
            ## for a particular layer (n_stack), keep track of the test_accuracy vs noise
            layer_avg_test_accs = []
            for noise_var in noise_vars: 

                test_acc_sum = 0

                ## average test accuracy over N_NOISE_SAMPLES
                for _ in range(N_NOISE_SAMPLES):
                    
                    ## create new instance of base model to add noise to
                    new_experiment = copy.deepcopy(self)

                    ## find the layer, add noise
                    with torch.no_grad():
                        if not self.preloaded:
                            weight_layer_string = f"linear_relu_stack.{n_stack}.weight"
                            bias_layer_string = f"linear_relu_stack.{n_stack}.bias"
                        else:
                            weight_layer_string = f"features.{n_stack}.weight"
                            bias_layer_string = f"features.{n_stack}.bias"
                        
                        print(f"adding noise to {weight_layer_string} weights")

                        weight_layer_size = new_experiment.model.state_dict()[weight_layer_string].data.shape
                        bias_layer_size = new_experiment.model.state_dict()[bias_layer_string].data.shape
                        # print(f"size = {size}")

                        ## get width of previous layer
                        if not self.preloaded:
                            if n_stack == 0:
                                layer_width = self.model._HIDDEN_LAYER_WIDTHS[n_stack]
                            else:
                                layer_width = self.model._HIDDEN_LAYER_WIDTHS[n_stack-1]

                            true_noise_var =  noise_var / layer_width
                        
                        if self.preloaded:
                            # layer_width = size[0]
                            # print(f"layer_width = {layer_width}")
                            # true_noise_var = noise_var / layer_width
                            true_noise_var = noise_var

                        weight_noise = torch.normal(mean=0, std=np.sqrt(true_noise_var), size=weight_layer_size).to(device)
                        bias_noise = torch.normal(mean=0, std=np.sqrt(true_noise_var), size=bias_layer_size).to(device)

                        ## add noise to both (bias) and weight layers
                        ## bias_layer_string = f"features.{n_stack}.bias"

                        new_experiment.model.state_dict()[weight_layer_string].data += weight_noise
                        new_experiment.model.state_dict()[bias_layer_string].data += bias_noise
                        

                    ## after noise has been added to a layer, get the accuracy
                    test_acc = new_experiment.get_test_accuracy()
                    print(f"test accuracy for {weight_layer_string} = {test_acc:.2%}")

                    test_acc_sum += test_acc

                avg_test_acc = test_acc_sum/N_NOISE_SAMPLES
                layer_avg_test_accs.append(avg_test_acc)
                print(f"Average test accuracy for {weight_layer_string} with N = {N_NOISE_SAMPLES} samples of noise at vars = {noise_var}: {avg_test_acc:.2%}")
            
            # layer_number = (2 + int((n_stack-1)/2)) if n_stack != 0 else 1
            # print(f'setting layer_{layer_number}')

            self.all_layer_noise_test_acc[f'layer_{n_stack+1}'] = np.array(layer_avg_test_accs)

    def create_accuracy_vs_noise_plots(self):
        df_accuracy_vs_noise = pd.DataFrame(self.all_layer_noise_test_acc)
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
                    marker=dict(color=self._layer_color_map[layer_col]),
                    name=layer_col,
                )
            )
        
        fig_noise_vs_accuracy.update_traces(mode='lines+markers', opacity=0.8) 
        
        if self.preloaded:
            
            self.test_accuracy = self.get_test_accuracy()

            title = f"""Preloaded CNN with test accuracy = {self.test_accuracy:.2%}
                            <br>Noise vs Test Accuracy by layer ({self.dataset_name})"""
        else:
            title = f"""DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS} with training accuracy = {self.train_accuracy:.2%}
                            <br>Noise vs Test Accuracy by layer ({self.dataset_name}, seed = {self.noise_random_seed})"""
        
        fig_noise_vs_accuracy.update_layout(title=title, xaxis_title='noise vars (unscaled)', height=1200)
        self.all_figures['noise_test_accuracies'] = fig_noise_vs_accuracy

def run_dnn_experiments(dnn_experiments, N_EPOCHS, directory, noise_random_seed, MAX_TRAIN_ATTEMPTS=5, regularizer=None, debug=True):
    """Convenience method to call DNN experiments with different initializations"""

    dnn_experiment_results = {}
    for experiment_name, dnn in dnn_experiments.items():
        print(f"Running {experiment_name}")
        dnn_experiment = DnnLayerWeightExperiment(model=dnn, dataset_name='fashion_mnist', noise_random_seed=42, preloaded=False)
        dnn_experiment.load_dataset()

        train_attempt = 0
        while(True):
            if train_attempt < MAX_TRAIN_ATTEMPTS:
                try:
                    dnn_experiment.train_base_model(N_EPOCHS=N_EPOCHS, regularizer=regularizer) 
                    break
                except Exception as e:
                    if str(e) == "Poor initialization!":
                        train_attempt += 1
                        new_random_seed = np.random.randint(1,100)
                        print(f"Reinitializing weights and re-attempting training with new random seed = {new_random_seed}")
                        dnn.initialize_weights(new_random_seed=new_random_seed)
                        continue
                    else:
                        raise
            else:
                raise Exception("max number of train attempts exceeded!")
        
        dnn_experiment.get_test_accuracy()
        dnn_experiment.create_layer_weight_plots()

        base_folder = f"{directory}/{experiment_name}/"
        os.makedirs(base_folder, exist_ok=True)

        for weight_type in ['initial_weights','final_weights']:
            base_file_path = f"{directory}/{experiment_name}/{dnn.init_type}_init_{weight_type}_"

            ## display figures regardless
            dnn_experiment.all_figures[weight_type]['histogram'].write_html(base_file_path + "histogram.html")
            dnn_experiment.all_figures[weight_type]['singular_values'].write_html(base_file_path + "singular_values.html")
            dnn_experiment.all_figures[weight_type]['heatmap'].write_html(base_file_path + "heatmap.html")

        dnn_experiment.all_figures['condition_numbers_by_epoch'].write_html(base_file_path + f"condition_numbers_by_epoch.html")
        dnn_experiment.all_figures['weight_stds_by_epoch'].write_html(base_file_path + f"weight_stds_by_epoch.html")
        dnn_experiment.all_figures['accuracies_by_epoch'].write_html(base_file_path + f"accuracies_by_epoch.html")

        ## store experimental data
        dnn_experiment_results[experiment_name] = dnn_experiment
    
        ## dry_run = shorten the experiment just to see that it works
        if debug:
            dnn_experiment.create_models_with_noise(N_NOISE_SAMPLES=5)
        else:
            dnn_experiment.create_models_with_noise(N_NOISE_SAMPLES=500)
            # torch.set_num_threads(8)
        
        dnn_experiment.create_accuracy_vs_noise_plots()

        file_path = f"{directory}/{experiment_name}/{dnn.init_type}_init_noise_experiments.html"

        dnn_experiment.all_figures['noise_test_accuracies'].write_html(file_path)

    return dnn_experiment_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run DNN and weight noise experiments")
    parser.add_argument("--experiment-version", type=str, help="Experiment version number for creating folders to hold results, should be a string of the form v1, v2...", required=True)
    parser.add_argument("--experiment-type", type=str, help="Train a DNN or use a pre-loaded CNN", choices=["DNN", "dnn", "CNN", "cnn"], required=True)
    parser.add_argument("--cloud-environment", type=str, help="Local or Tufts HPC", choices=["local","hpc"], required=True)
    parser.add_argument("--debug-mode", type=str, help="Debug Mode or actual experiment", choices=["debug","experiment"], required=True)
    args = parser.parse_args()

    version = args.experiment_version
    directory = f"experiment_plots/{version}"

    cloud_environment = args.cloud_environment
    debug = True if args.debug_mode == "debug" else False
    if debug:
        N_EPOCHS = 20
    else:
        N_EPOCHS = 400

    ## if using the cluster, automatically create a new directory, no user input
    if cloud_environment == 'hpc':
        os.makedirs(directory, exist_ok=True)
    if cloud_environment == 'local':
        if os.path.exists(directory):
            response = input(f"Warning: The directory '{directory}' already exists. Do you want to proceed? (y/n): ").strip().lower()
            if response == 'y':
                print(f"Proceeding with existing directory: {directory}")
            else:
                raise Exception("Operation cancelled by the user.")
        else:
            os.makedirs(directory, exist_ok=True)
    
    experiment_type = args.experiment_type
    if experiment_type.upper() == 'DNN':
    
        # print("----- Running all DNN experiments -----")

        # start = time.time()
        # random_seed = 43

        # ## adjust the DNN hidden layer dimensions as needed
        # dnn1 = DNN(N_CLASSES=10, HIDDEN_LAYER_WIDTHS=[64, 64, 64, 64], random_seed=random_seed, init_type="normal")
        # dnn2 = DNN(N_CLASSES=10, HIDDEN_LAYER_WIDTHS=[128, 128, 128, 128], random_seed=random_seed, init_type="normal")
        # dnn3 = DNN(N_CLASSES=10, HIDDEN_LAYER_WIDTHS=[128, 128, 128, 128, 128, 128, 128], random_seed=random_seed, init_type="normal")

        # dnn_experiments = {
        #     'dnn1': dnn1,
        #     'dnn2': dnn2,
        #     'dnn3': dnn3
        # }

        ## for debugging, set N_EPOCHS = 10
        ## when running the job, set N_EPOCHS = 200 or more
        
        print("----- Running all DNN experiments -----")

        start = time.time()

        random_seed = 42
        noise_random_seed = 42
        dnn1 = DNN(N_CLASSES=10, HIDDEN_LAYER_WIDTHS=[512, 256, 128, 64], random_seed=random_seed, init_type="normal")
        dnn2 = DNN(N_CLASSES=10, HIDDEN_LAYER_WIDTHS=[256, 256, 256, 256, 256, 256, 256], random_seed=random_seed, init_type="normal")
        dnn3 = DNN(N_CLASSES=10, HIDDEN_LAYER_WIDTHS=[64, 128, 256, 512], random_seed=random_seed, init_type="normal")
        dnn_experiments = {
            'dnn1': dnn1,
            'dnn2': dnn2,
            'dnn3': dnn3,
        }
        dnn_experiments_results = run_dnn_experiments(dnn_experiments, N_EPOCHS=N_EPOCHS, noise_random_seed=42, directory=directory, MAX_TRAIN_ATTEMPTS=5, regularizer="l2", debug=debug)
        
        ## save dnn_experiments_results using torch
        torch.save(dnn_experiments_results, f'{directory}/dnn_experiments_results.pth')

        end = time.time()
        runtime = (end - start) / 60
        print(f"Total runtime = {runtime:.2f} minutes")

    ## pretrained CNN + CIFAR-10
    if experiment_type.upper() == 'CNN':
        start = time.time()
        print("----- Running CNN experiments with pretrained VGG models -----")

        if debug:
            cnn_model = torch.hub.load("chenyaofo/pytorch-cifar-models",f"cifar10_vgg11_bn", pretrained=True).to(device)
            print(cnn_model)
            random_seed = 42
            cnn_experiment = DnnLayerWeightExperiment(model=cnn_model, dataset_name='cifar10', preloaded=True, noise_random_seed=random_seed)
            cnn_experiment.load_dataset()
            cnn_experiment.create_layer_weight_plots()
            cnn_experiment.all_figures['fig_histogram'].write_html(f"{directory}/vgg11_final_layer_weights.html")

            cnn_experiment.create_models_with_noise(N_NOISE_SAMPLES=3)
            cnn_experiment.create_accuracy_vs_noise_plots()

            file_path = f"{directory}/vgg11_init_noise_experiments.html"
            cnn_experiment.all_figures['noise_test_accuracies'].write_html(file_path)
        else:
            for vgg_model in ["vgg11","vgg13","vgg16"]:
                cnn_model = torch.hub.load("chenyaofo/pytorch-cifar-models",f"cifar10_{vgg_model}_bn", pretrained=True).to(device)
                print(cnn_model)
                random_seed = 42
                cnn_experiment = DnnLayerWeightExperiment(model=cnn_model, dataset_name='cifar10', preloaded=True, noise_random_seed=random_seed)
                cnn_experiment.load_dataset()
                cnn_experiment.create_layer_weight_plots()
                cnn_experiment.all_figures['fig_histogram'].write_html(f"{directory}/{vgg_model}_final_layer_weights.html")
                
                cnn_experiment.create_models_with_noise(N_NOISE_SAMPLES=500)
                cnn_experiment.create_accuracy_vs_noise_plots()
                file_path = f"{directory}/{vgg_model}_init_noise_experiments.html"
                cnn_experiment.all_figures['noise_test_accuracies'].write_html(file_path)

        end = time.time()
        runtime = (end - start) / 60
        print(f"Total runtime = {runtime:.2f} minutes")

    ## LEGACY SCRATCH WORK

    # Load ResNet18
    # resnet18_model = resnet18()

    # # Modify conv1 for CIFAR-10 (adjust for 32x32 images)
    # resnet18_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # resnet18_model.maxpool = nn.Identity()  # Remove maxpool for smaller images

    # # Modify the fully connected layer for 10 classes (CIFAR-10)
    # resnet18_model.fc = nn.Linear(512, 10)
    # resnet18_model.load_state_dict(torch.load('./cifar10_models/state_dicts/resnet18.pt'))