import argparse
import copy
import os
import time
import numpy as np
import pandas as pd
import re

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.nn.utils.parametrize import remove_parametrizations

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from utils import fit_gaussian_curve, init_weights
from scheduler import WarmupCosineLR

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class DNN(nn.Module):
    def __init__(self, N_CLASSES, DATASET_NAME, HIDDEN_LAYER_WIDTHS, random_seed, init_type, normalization_type):
        super().__init__()
        self._N_CLASSES = N_CLASSES
        self._DATASET_NAME = DATASET_NAME

        self._HIDDEN_LAYER_WIDTHS = HIDDEN_LAYER_WIDTHS
        self._N_HIDDEN_LAYERS = len(self._HIDDEN_LAYER_WIDTHS) 
        self._N_LAYERS = self._N_HIDDEN_LAYERS+1 # hidden layers + 1 output layer

        self.random_seed = random_seed
        self.init_type = init_type
        self.normalization_type = str(normalization_type)

        if device=="mps":
            torch.mps.manual_seed(self.random_seed)
        if device=="cuda":
            torch.cuda.manual_seed(self.random_seed)

        ## NOTE: ideally we should parametrize 28 x 28 or 32 x 32
        ## and construct the layers a more dynamic fashion
        hidden_layers = []
        for i, layer_width in enumerate(self._HIDDEN_LAYER_WIDTHS):
            if i == 0:
                if self._DATASET_NAME.upper() == 'MNIST':
                    first_hidden_layer = nn.Linear(28 * 28 * 1, layer_width)
                elif self._DATASET_NAME.upper() == 'FASHION_MNIST':
                    first_hidden_layer = nn.Linear(28 * 28 * 1, layer_width)
                elif self._DATASET_NAME.upper() == 'CIFAR10':
                    first_hidden_layer = nn.Linear(32 * 32 * 3, layer_width)
                else:
                    raise Exception(f"Unsupported dataset: {self._DATASET_NAME}")
                hidden_layers.append(first_hidden_layer)

            else:
                previous_layer_width = self._HIDDEN_LAYER_WIDTHS[i-1]

                ## add linear layer, batch/layer norm, and Relu
                ## for weight norm, directly apply to linear layer
                if self.normalization_type.upper() == 'BATCH':
                    hidden_layers.append(nn.Linear(previous_layer_width, layer_width))
                    hidden_layers.append(nn.BatchNorm1d(num_features=layer_width))
                    hidden_layers.append(nn.ReLU())
                elif self.normalization_type.upper() == 'LAYER':
                    hidden_layers.append(nn.Linear(previous_layer_width, layer_width))
                    hidden_layers.append(nn.LayerNorm(normalized_shape=layer_width))
                    hidden_layers.append(nn.ReLU())
                elif self.normalization_type.upper() == 'WEIGHT':
                    hidden_layers.append(nn.utils.parametrizations.weight_norm(nn.Linear(previous_layer_width, layer_width)))
                    hidden_layers.append(nn.ReLU())
                elif self.normalization_type.upper() == 'NONE':
                    hidden_layers.append(nn.Linear(previous_layer_width, layer_width))
                    hidden_layers.append(nn.ReLU())
                else:
                    raise Exception(f"Normalization of type {normalization_type} not yet implemented or supported!")
            
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
        if self._DATASET_NAME == 'mnist':
            x = x.view(-1, 28 * 28 * 1)
        elif self._DATASET_NAME == 'fashion_mnist':
            x = x.view(-1, 28 * 28 * 1)
        elif self._DATASET_NAME == 'cifar10':
            x = x.view(-1, 32 * 32 * 3) # this should be parametrized ideally
        else:
            raise Exception(f"Unsupported dataset: {self._DATASET_NAME}")
        
        # x = x.view(-1, 32 * 32 * 3) # this should be parametrized ideally
        x = self.linear_relu_stack(x)
        return x

class DiagDNN(nn.Module):
    def __init__(self, N_CLASSES, DATASET_NAME, HIDDEN_LAYER_WIDTHS, random_seed, init_type, normalization_type=None):
        super().__init__()
        self._N_CLASSES = N_CLASSES
        self._DATASET_NAME = DATASET_NAME

        self._HIDDEN_LAYER_WIDTHS = HIDDEN_LAYER_WIDTHS
        self._N_HIDDEN_LAYERS = len(self._HIDDEN_LAYER_WIDTHS) 
        self._N_LAYERS = self._N_HIDDEN_LAYERS+1 # hidden layers + 1 output layer

        self.random_seed = random_seed
        self.init_type = init_type
        self.normalization_type = normalization_type

        ## the individual diagonal values are the parameters, not the diagonal matrix itself
        ## we want to fix all of them constant?
        np.random.seed(42)
        self.diag_params = nn.ParameterList([
            nn.Parameter(torch.tensor(10**-5)) for _ in self._HIDDEN_LAYER_WIDTHS + [self._N_CLASSES]
        ])

        if device=="mps":
            torch.mps.manual_seed(self.random_seed)
        if device=="cuda":
            torch.cuda.manual_seed(self.random_seed)

        ## NOTE: ideally we should parametrize 28 x 28 or 32 x 32
        ## and construct the layers a more dynamic fashion
        hidden_layers = []
        for i, layer_width in enumerate(self._HIDDEN_LAYER_WIDTHS):
            if i == 0:
                first_hidden_layer = nn.Linear(32 * 32 * 3, layer_width)
                hidden_layers.append(first_hidden_layer)
                hidden_layers.append(nn.ReLU())
            else:
                previous_layer_width = self._HIDDEN_LAYER_WIDTHS[i-1]
                hidden_layers.append(nn.Linear(previous_layer_width, layer_width))
                hidden_layers.append(nn.ReLU())

        output_layer = [nn.Linear(self._HIDDEN_LAYER_WIDTHS[-1], self._N_CLASSES)]
        all_layers = hidden_layers + output_layer

        self.all_layers = nn.ModuleList(all_layers)
        self.initialize_weights()
        
    def initialize_weights(self, new_random_seed=None):
        """Convenience method to initialize or re-initialize weights"""
        if new_random_seed is not None:
            self.random_seed = new_random_seed

        for layer in self.all_layers:
            if isinstance(layer, nn.Linear):
                init_weights(layer, self.init_type, self.random_seed)

    def forward(self, x):
        if self._DATASET_NAME == 'mnist':
            x = x.view(-1, 28 * 28 * 1)
        elif self._DATASET_NAME == 'fashion_mnist':
            x = x.view(-1, 28 * 28 * 1)
        elif self._DATASET_NAME == 'cifar10':
            x = x.view(-1, 32 * 32 * 3) # this should be parametrized ideally
        else:
            raise Exception(f"Unsupported dataset: {self._DATASET_NAME}")
        
        layer_idx = 0
        for layer in self.all_layers:
            if isinstance(layer, nn.Linear):
                weight = layer.weight
                
                # Construct the diagonal matrix from learnable parameters
                diag_matrix = self.diag_params[layer_idx] * torch.ones(layer.out_features)

                # Expand to match weight dimensions
                full_diag = torch.zeros_like(weight)
                min_dim = min(weight.shape)
                full_diag[:min_dim, :min_dim] = diag_matrix

                # Modify the weight
                modified_weight = weight + full_diag

                # Apply linear transformation with bias
                x = nn.functional.linear(x, modified_weight, layer.bias)
                layer_idx += 1
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
        return x

class CNN(nn.Module):
    """Creates a CNN based on the VGG11 architecture, but with different possible normalizations"""
    def __init__(self, N_CLASSES, DATASET_NAME, HIDDEN_LAYER_WIDTHS, random_seed, init_type, normalization_type):
        super().__init__()
        self._N_CLASSES = N_CLASSES
        self._DATASET_NAME = DATASET_NAME

        self._HIDDEN_LAYER_WIDTHS = HIDDEN_LAYER_WIDTHS
        self._N_HIDDEN_LAYERS = len(self._HIDDEN_LAYER_WIDTHS) 
        self._N_LAYERS = self._N_HIDDEN_LAYERS+1 # hidden layers + 1 output layer

        self.random_seed = random_seed
        self.init_type = init_type
        self.normalization_type = str(normalization_type)

        if device=="mps":
            torch.mps.manual_seed(self.random_seed)
        if device=="cuda":
            torch.cuda.manual_seed(self.random_seed)

        ## NOTE: ideally we should parametrize 28 x 28 or 32 x 32
        ## and construct the layers a more dynamic fashion
        hidden_layers = []
        for i, layer_width in enumerate(self._HIDDEN_LAYER_WIDTHS):
            if i == 0:
                if self._DATASET_NAME.upper() == 'MNIST':
                    first_hidden_layer = nn.Linear(28 * 28 * 1, layer_width)
                elif self._DATASET_NAME.upper() == 'FASHION_MNIST':
                    first_hidden_layer = nn.Linear(28 * 28 * 1, layer_width)
                elif self._DATASET_NAME.upper() == 'CIFAR10':
                    first_hidden_layer = nn.Linear(32 * 32 * 3, layer_width)
                else:
                    raise Exception(f"Unsupported dataset: {self._DATASET_NAME}")
                hidden_layers.append(first_hidden_layer)

            else:
                previous_layer_width = self._HIDDEN_LAYER_WIDTHS[i-1]

                ## add linear layer, batch/layer norm, and Relu
                ## for weight norm, directly apply to linear layer
                if self.normalization_type.upper() == 'BATCH':
                    hidden_layers.append(nn.Linear(previous_layer_width, layer_width))
                    hidden_layers.append(nn.BatchNorm1d(num_features=layer_width))
                    hidden_layers.append(nn.ReLU())
                elif self.normalization_type.upper() == 'LAYER':
                    hidden_layers.append(nn.Linear(previous_layer_width, layer_width))
                    hidden_layers.append(nn.LayerNorm(normalized_shape=layer_width))
                    hidden_layers.append(nn.ReLU())
                elif self.normalization_type.upper() == 'WEIGHT':
                    hidden_layers.append(nn.utils.parametrizations.weight_norm(nn.Linear(previous_layer_width, layer_width)))
                    hidden_layers.append(nn.ReLU())
                elif self.normalization_type.upper() == 'NONE':
                    hidden_layers.append(nn.Linear(previous_layer_width, layer_width))
                    hidden_layers.append(nn.ReLU())
                else:
                    raise Exception(f"Normalization of type {normalization_type} not yet implemented or supported!")
            
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
        if self._DATASET_NAME == 'mnist':
            x = x.view(-1, 28 * 28 * 1)
        elif self._DATASET_NAME == 'fashion_mnist':
            x = x.view(-1, 28 * 28 * 1)
        elif self._DATASET_NAME == 'cifar10':
            x = x.view(-1, 32 * 32 * 3) # this should be parametrized ideally
        else:
            raise Exception(f"Unsupported dataset: {self._DATASET_NAME}")
        
        # x = x.view(-1, 32 * 32 * 3) # this should be parametrized ideally
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

        self.normalization_type = model.normalization_type
        self.best_lr = None
        self.model = model
        self.TRAIN_RATIO = 0.8
        self._N_EPOCHS = None
        
        self.max_weights_std = None

        self.preloaded = preloaded
        self.model_layer_info = {}
        self.noise_experiment_data = None

        ## guarantee the same color scheme across plots for readability
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
            
        n = 1
        for name, module in self.model.named_modules():
            ## weights come before biases

            # print(f"layer name: {name} of type {type(module)}")

            if isinstance(module, nn.modules.linear.Linear):
                print(f"found linear layer for n = {n}")
                if self.model_layer_info.get(f'layer_{n}') is None:
                    self.model_layer_info[f'layer_{n}'] = {
                        'initial_weights': None,
                        'initial_biases': None,
                        'final_weights': None,
                        'final_biases': None,
                        'weight_std_by_epoch': [],
                        'weight_abs_max_by_epoch': [],
                        'weight_abs_min_by_epoch': [],
                        'cond_number_by_epoch': [],
                        'min_singular_value_by_epoch': [],
                        'max_singular_value_by_epoch': [],
                    }
                    self.model_layer_info[f'layer_{n}']['initial_weights'] = module.weight.data.clone().numpy()
                    self.model_layer_info[f'layer_{n}']['initial_biases'] = module.bias.data.clone().numpy()
                    
                    # print(self.model_layer_info[f'layer_{n}']['initial_weights'].shape)
                    # print(self.model_layer_info[f'layer_{n}']['initial_biases'].shape)

                    n += 1
                else:
                    pass
        n = 1

        # model output instance attributes
        self.train_accuracies = []
        self.val_accuracies = []
        self.test_accuracies = []
        self.learning_rates = []

        self.layer_summary_stats = {}
        self.all_layer_noise_test_acc = {
            'input_dim': {},
            'output_dim': {},
            'layer_variance': {},
            'noise_vars': None,
        }
        self.layer_condition_numbers = {}

        self.all_figures = {
            'initial_weights': None,
            'final_weights': None,
            'condition_numbers_by_epoch': None,
            'min_singular_values_by_epoch': None,
            'max_singular_values_by_epoch': None,
            'singular_values_by_layer': None, # done
            'final_epoch_by_layer': None, # done
            'weight_means_by_epoch': None,
            'weight_stds_by_epoch': None,
            'accuracies_by_epoch': None,
            # 'noise_test_accuracies': None,
            'noise_acc_vs_layer': None, #
            'learning_rate': None,
            # 'layer_noise_test_accuracies': None,
        }
    
    # Define the transformations for the data
    # Create the dataloaders for train, validation, and test

    def load_dataset(self):
        # Download and load the training, validation, and test datasets

        if self.dataset_name == 'fashion_mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            ## download training data and split into train and validation
            dataset = datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
            dataset_size = len(dataset)
            train_size = int(self.TRAIN_RATIO * dataset_size)
            val_size = dataset_size - train_size

            self.train_data, self.val_data = random_split(dataset, [train_size, val_size])
            self.test_data = datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

        elif self.dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
            dataset_size = len(dataset)
            train_size = int(self.TRAIN_RATIO * dataset_size)
            val_size = dataset_size - train_size

            self.train_data, self.val_data = random_split(dataset, [train_size, val_size])
            self.test_data = datasets.MNIST('./data', train=False, transform=transform, download=True)
        
        elif self.dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize( 
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            dataset = datasets.CIFAR10('./data', train=True, transform=transform, download=True)
            dataset_size = len(dataset)
            train_size = int(self.TRAIN_RATIO * dataset_size)
            val_size = dataset_size - train_size

            self.train_data, self.val_data = random_split(dataset, [train_size, val_size])
            self.test_data = datasets.CIFAR10('./data', train=False, transform=transform, download=True)

        else:
            raise Exception(f"Dataset {self.dataset_name} not supported")
        
        ## create DataLoaders
        trainloader = DataLoader(self.train_data, batch_size=128, shuffle=True, num_workers=2)
        valloader = DataLoader(self.val_data, batch_size=128, shuffle=False, pin_memory=True)
        testloader = DataLoader(self.test_data, batch_size=128, shuffle=True, num_workers=2)
        self.dataloaders = {
            'train': trainloader,
            'validation': valloader,
            'test': testloader
        }

    def train_base_model(self, N_EPOCHS=30, dnn_learning_rates=[], regularizer=None):

        self._N_EPOCHS = N_EPOCHS
        
        device = next(self.base_model.parameters()).device

        ## data is expected to be in batches
        ## we can add regularizer term to CrossEntropyLoss using parameter weight_decay
        ## OR we can implement it out-of-the-box (this is useful if we intend to modify it)
        ## construct l2_norm one layer at a time for future use
        
        # l2_lambda = 10**-5

        l2_loss = torch.tensor(0.)
        loss_function = nn.CrossEntropyLoss()

        ## layer l2
        if regularizer == 'layer-l2':
            layer_multiplier = 1
            for name, module in self.model.named_modules():
                if isinstance(module, nn.modules.linear.Linear):
                    l2_loss = l2_loss + layer_multiplier * torch.linalg.norm(module.weight, 2).pow(2)
                    layer_multiplier *= 0.5
        elif regularizer == 'l2':
            # layer_multiplier = 1
            # for name, module in self.model.named_modules():
            #     if isinstance(module, nn.modules.linear.Linear):
            #         l2_loss = l2_loss + torch.linalg.norm(module.weight, 2).pow(2)
            weight_decay = 1e-3
        elif regularizer is None:
            weight_decay = 0
        else:
            raise Exception(f"Regularizer {regularizer} not implemented!")
        
        ## try different learning rates but keep track of their success
        ## this ensures we choose the BEST learning rate, not the first that succeeds

        # best_train_accuracy = 0.0
        best_test_accuracy = 0.0 
        best_model_state = None

        train_accuracies = []
        test_accuracies = []
        learning_rates = []

        ## we need to calculate these for each epoch AND each learning rate
        ## otherwise these will not be stored 
        if isinstance(self.model, DNN):
            model_layer_info = {
                f'layer_{i}': {
                    'initial_weights': self.model_layer_info[f'layer_{i}']['initial_weights'].copy(),
                    'initial_biases': self.model_layer_info[f'layer_{i}']['initial_biases'].copy(),
                    'final_weights': None,
                    'final_biases': None,
                    'weight_mean_by_epoch': [],
                    'weight_std_by_epoch': [],
                    'weight_abs_max_by_epoch': [],
                    'weight_abs_min_by_epoch': [],
                    'cond_number_by_epoch': [],
                    'min_singular_value_by_epoch': [],
                    'max_singular_value_by_epoch': [],
                } for i in range(1,self.model._N_LAYERS+1)
            }
        elif isinstance(self.model, DiagDNN):
            model_layer_info = {
                f'layer_{i}': {
                    'initial_weights': self.model_layer_info[f'layer_{i}']['initial_weights'].copy(),
                    'initial_biases': self.model_layer_info[f'layer_{i}']['initial_biases'].copy(),
                    'final_weights': None,
                    'final_biases': None,
                    'weight_mean_by_epoch': [],
                    'weight_std_by_epoch': [],
                    'weight_abs_max_by_epoch': [],
                    'weight_abs_min_by_epoch': [],
                    'cond_number_by_epoch': [],
                    'min_singular_value_by_epoch': [],
                    'max_singular_value_by_epoch': [],
                    'diag_params_by_epoch': []
                } for i in range(1,self.model._N_LAYERS+1)
            }
        else:
            raise Exception(f"model of type {type(self.model)} not supported!")

        lrs = dnn_learning_rates

        for lr in lrs:

            print(f"Trying learning rate = {lr}")

            if isinstance(self.model, DNN):
                # optimizer = optim.Adam(self.model.parameters(), lr=lr)

                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=0.9,
                    nesterov=True,
                )
                
            elif isinstance(self.model, DiagDNN):
                
                optimizer = optim.Adam([
                    {'params': self.model.all_layers.parameters(), 'lr': lr}
                ] + [{'params': [diag], 'lr': 1e-2 * lr} for diag in self.model.diag_params])
            
            # Set up optimizer with parameter groups
            
            if lr_scheduler.upper() == 'LINEAR':
                scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.2)
            elif lr_scheduler.upper() == 'WARMUP':
                scheduler = WarmupCosineLR(optimizer, warmup_epochs=N_EPOCHS * 0.3, max_epochs=N_EPOCHS)
            elif lr_scheduler.upper() == 'WARMRESTART':
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
            else:
                raise Exception(f"learning rate scheduler {lr_scheduler} not implemented!")

            for epoch in range(N_EPOCHS):

                train_loss = 0.0
                
                train_correct = 0
                train_total = 0

                val_loss = 0.0
                val_total = 0

                # Training phase
                for images, labels in self.dataloaders['train']:

                    ## move the data to device (CPU/GPU)
                    images=images.to(device)
                    labels=labels.to(device)
                    self.model.train()

                    optimizer.zero_grad()
                    outputs=self.model(images)

                    _, predicted = torch.max(outputs, 1)

                    # print(f"predicted: {predicted.size()}")
                    # print(f"labels: {labels.size()}")

                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    loss = loss_function(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    ## after the optimizer step happens, we can make small changes to weights
                        ## such as adding a small diagonal matrix
                    # if add_diagonal_matrix:

                    #     # Add small diagonal matrix to weights
                    #     beta = 0.9**epoch
                    #     delta = 1e-2
                    #     for name, module in self.model.named_modules():
                    #         if isinstance(module, nn.modules.linear.Linear):
                    #             with torch.no_grad():
                    #                 weight_shape = module.weight.data.shape
                    #                 if weight_shape[0] == weight_shape[1]:
                    #                     diag_matrix = beta * delta * torch.eye(weight_shape[0], device=device)
                    #                     module.weight.data += diag_matrix

                    train_loss += loss.item()

                train_accuracy = train_correct / train_total
                train_accuracies.append(train_accuracy)
                
                train_loss /= len(self.dataloaders['train'])
                
                # Validation phase, we only calculate loss to determine plateauing
                self.model.eval()
                with torch.no_grad():
                    for images, labels in self.dataloaders['validation']:
                        images=images.to(device)
                        labels=labels.to(device)
                        outputs = self.model(images)

                        loss = loss_function(outputs, labels)
                        val_loss += loss.item()
                
                val_loss /= len(self.dataloaders['validation'])
                print(f"val loss: {val_loss}")

                scheduler.step()

                if np.isnan(train_loss):
                    print(f"Training did not converge for learning rate = {lr}")
                    break
                
                # current_lr = optimizer.param_groups[0]['lr']
                current_lr = scheduler.get_last_lr()[0]
                learning_rates.append(current_lr)
                print(f"Epoch {epoch}, lr = {current_lr}: training_loss = {train_loss}, train accuracy = {train_accuracy:.2%}")
                
                ## for each epoch, we need to get layer weight std and condition number

                n = 1
                layer_idx = 0
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.modules.linear.Linear):
                        # print(f"saving layer n = {n}")
                        layer_weights = module.weight.data.clone().numpy()
                        
                        cond_number = np.linalg.cond(layer_weights)
                        if np.isnan(cond_number):
                            print("Warning: condition number is null!")

                        weight_mean = np.mean(layer_weights.flatten())
                        weight_std = np.std(layer_weights.flatten())
                        weight_abs_max = np.max(np.abs(layer_weights.flatten()))
                        weight_abs_min = np.min(np.abs(layer_weights.flatten()))

                        if isinstance(self.model, DiagDNN):
                            layer_diag_param = self.model.diag_params[layer_idx].data.item()
                            model_layer_info[f'layer_{n}']['diag_params_by_epoch'].append(layer_diag_param)
                            layer_idx += 1

                        ## we store the condition number and weight std by epoch
                        ## and set this to the actual model's info if this epoch does the best
                        model_layer_info[f'layer_{n}']['cond_number_by_epoch'].append(cond_number)
                        model_layer_info[f'layer_{n}']['weight_mean_by_epoch'].append(weight_mean)
                        model_layer_info[f'layer_{n}']['weight_std_by_epoch'].append(weight_std)
                        model_layer_info[f'layer_{n}']['weight_abs_max_by_epoch'].append(weight_abs_max)
                        model_layer_info[f'layer_{n}']['weight_abs_min_by_epoch'].append(weight_abs_min)
                        

                        U,S,V = np.linalg.svd(layer_weights) 
                        min_singular_value = S.min()
                        max_singular_value = S.max()

                        model_layer_info[f'layer_{n}']['min_singular_value_by_epoch'].append(min_singular_value)
                        model_layer_info[f'layer_{n}']['max_singular_value_by_epoch'].append(max_singular_value)

                        n += 1

                        
                    else:
                        continue

                # print(f"epoch {epoch}: standard deviation for {name} = {weight_std}")

                ## calculate test accuracy for the epoch
                test_accuracy = self.get_test_accuracy()
                test_accuracies.append(test_accuracy)

            ## model training is completed for some learning rate, lr
            ## check if the accuracy is above a certain threshold AND better than previous accuracy
            if (train_accuracy <= 0.10) | (np.isnan(train_loss)):
                print(f"Training did not converge for (starting) learning rate = {lr}")

            else:
                print(f"Training successfully converged for (starting) learning rate = {lr}")

                if (test_accuracy > best_test_accuracy):
                    print(f"Better test accuracy = {test_accuracy} found for learning rate {lr}, overwriting model")
                    self.best_lr = lr
                    best_test_accuracy = test_accuracy # primitives are immutable! if test_accuracy changes, won't impact best_test_accuracy
                    best_model_state = copy.deepcopy(self.model.state_dict())

                    self.model_layer_info = copy.deepcopy(model_layer_info)
                    self.train_accuracy = train_accuracy # saves train accuracy corresponding to best test accuracy
                    self.train_accuracies = train_accuracies
                    self.test_accuracy = test_accuracy
                    self.test_accuracies = test_accuracies
                    self.learning_rates = learning_rates

            print("--- Resetting model experiment parameters! ---")
            train_accuracies = []
            test_accuracies = []
            learning_rates = []
            
            if isinstance(self.model, DNN):
                model_layer_info = {
                    f'layer_{i}': {
                        'initial_weights': self.model_layer_info[f'layer_{i}']['initial_weights'].copy(),
                        'initial_biases': self.model_layer_info[f'layer_{i}']['initial_biases'].copy(),
                        'final_weights': None,
                        'final_biases': None,
                        'weight_mean_by_epoch': [],
                        'weight_std_by_epoch': [],
                        'weight_abs_max_by_epoch': [],
                        'weight_abs_min_by_epoch': [],
                        'cond_number_by_epoch': [],
                        'min_singular_value_by_epoch': [],
                        'max_singular_value_by_epoch': [],
                    } for i in range(1,self.model._N_LAYERS+1)
                }
            if isinstance(self.model, DiagDNN):
                model_layer_info = {
                    f'layer_{i}': {
                        'initial_weights': self.model_layer_info[f'layer_{i}']['initial_weights'].copy(),
                        'initial_biases': self.model_layer_info[f'layer_{i}']['initial_biases'].copy(),
                        'final_weights': None,
                        'final_biases': None,
                        'weight_mean_by_epoch': [],
                        'weight_std_by_epoch': [],
                        'weight_abs_max_by_epoch': [],
                        'weight_abs_min_by_epoch': [],
                        'cond_number_by_epoch': [],
                        'min_singular_value_by_epoch': [],
                        'max_singular_value_by_epoch': [],
                        'diag_params_by_epoch': []
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
        
        n = 1
        for name, module in self.model.named_modules():
            ## weights come before biases

            # print(f"layer name: {name} of type {type(module)}")

            ## we need to do this to save the models...
            if str(self.normalization_type).upper() == 'WEIGHT':
                try:
                    remove_parametrizations(module, "weight", leave_parametrized=True)
                    print('successfully removed parametrized weight norm')
                except Exception as e:
                    pass

            if isinstance(module, nn.modules.linear.Linear):
                layer_weights = module.weight.data.clone().numpy()
                print(f"setting layer {n} final weights to weights from {name}")
                self.model_layer_info[f'layer_{n}']['final_weights'] = layer_weights
        
                layer_biases = module.bias.data.clone().numpy()
                print(f"setting bias {n} final weights to biases from {name}")
                self.model_layer_info[f'layer_{n}']['final_biases'] = layer_biases

                ## since there will always be a bias corresponding to weight, only increment after bias
                n += 1
        
        print("\nAfter removing parametrized modules...")
        for name, module in self.model.named_modules():
            print(f"layer name: {name} of type {type(module)}")
        
        print("training base_model completed:")

    def get_test_accuracy(self) -> float:
        
        # Ensure model is in evaluation mode
        self.model.eval()

        # Get the device and DataLoader
        device = next(self.model.parameters()).device
        
        # Initialize counters
        correct = 0
        total = 0

        # Disable gradient computation for efficiency
        with torch.no_grad():
            for images, labels in self.dataloaders['test']:
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
        Create all plots related to training of the model:
        - training and test accuracy curves
        - initial and final weight distributions by layer
        - initial and final singular value distributions by layer
        """
        if not self.preloaded:
            for weight_type in ['Initial','Final']:
                weight_type_key = f"{weight_type.lower()}_weights"

                fig_histogram = make_subplots(
                    rows=int(np.ceil(self.model._N_LAYERS/2)), cols=2,
                    subplot_titles=[f"{weight_type} Layer {i} weights" for i in range(1,self.model._N_LAYERS+1)]
                )

                ## HEATMAP OF WEIGHTS MATRIX

                # fig_heatmap = make_subplots(
                #     rows=int(np.ceil(self.model._N_LAYERS/2)), cols=2,
                #     subplot_titles=[f"Layer {i} weights" for i in range(1,self.model._N_LAYERS+1)]
                # )

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
                    # fig_heatmap.append_trace(go.Heatmap(z=weights_matrix, zmin=zmin, zmax=zmax), row=row, col=col)
                    
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
                
                fig_histogram.update_layout(title=f"""
                                DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}: final train acc = {final_train_accuracy:.2%}, final test_acc = {final_test_accuracy:.2%}
                                <br>Distribution of {weight_type} Weights by Layer ({self.dataset_name}, seed = {self.model.random_seed})<br>""",
                                margin=dict(t=120, l=50, r=50, b=50))
            
                fig_singular_values.update_layout(title=f"""
                                DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}: final train acc = {final_train_accuracy:.2%}, final test_acc = {final_test_accuracy:.2%}
                                <br>Singular Values of {weight_type} Weights by Layer ({self.dataset_name}, seed = {self.model.random_seed})<br>""",
                                margin=dict(t=120, l=50, r=50, b=50))
                
                # fig_heatmap.update_layout(title=f"""
                #                 DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}: final train acc = {final_train_accuracy:.2%}, final test_acc = {final_test_accuracy:.2%}
                #                 <br>Heatmap of {weight_type} Weights by Layer ({self.dataset_name}, seed = {self.model.random_seed})<br>""",
                #                 margin=dict(t=120, l=50, r=50, b=50))
                # fig_heatmap = equalize_axes_layout(fig_heatmap)
                # fig_singular_values = equalize_axes_layout(fig_singular_values)
                
                self.all_figures[weight_type_key] = {
                    'histogram': fig_histogram,
                    'singular_values': fig_singular_values
                }
            
            ## layer based plots
            fig_condition_numbers = make_subplots(
                rows=int(np.ceil(self.model._N_LAYERS/2)), cols=2,
                subplot_titles=[f"Layer {i}" for i in range(1,self.model._N_LAYERS+1)]
            )

            fig_weight_means = make_subplots(
                rows=int(np.ceil(self.model._N_LAYERS/2)), cols=2,
                subplot_titles=[f"Layer {i}" for i in range(1,self.model._N_LAYERS+1)]
            )
            
            fig_weight_stds = make_subplots(
                rows=int(np.ceil(self.model._N_LAYERS/2)), cols=2,
                subplot_titles=[f"Layer {i}" for i in range(1,self.model._N_LAYERS+1)]
            )
            
            fig_min_singular_values = make_subplots(
                rows=int(np.ceil(self.model._N_LAYERS/2)), cols=2,
                subplot_titles=[f"Min Singular Values for Layer {i} weights" for i in range(1,self.model._N_LAYERS+1)],
                vertical_spacing=0.05
            )

            fig_max_singular_values = make_subplots(
                rows=int(np.ceil(self.model._N_LAYERS/2)), cols=2,
                subplot_titles=[f"Max Singular Values for Layer {i} weights" for i in range(1,self.model._N_LAYERS+1)],
                vertical_spacing=0.05
            )

            if isinstance(self.model, DiagDNN):
                fig_diag_params = make_subplots(
                    rows=int(np.ceil(self.model._N_LAYERS/2)), cols=2,
                    subplot_titles=[f"Diag Param Values for Layer {i} weights" for i in range(1,self.model._N_LAYERS+1)],
                    vertical_spacing=0.05
                )

            ## singular values by layer requires matplotlib

            number_of_layers = len(self.model_layer_info.keys())
            subplot_titles = []
            for layer_name in self.model_layer_info.keys():
                for plot_type in ["Combined","Initial","Final"]:
                    subplot_titles.append(f"{plot_type} {layer_name.replace('_',' ').capitalize()}")
            
            # fig_training_weights = make_subplots(rows=number_of_layers, cols=3, subplot_titles=subplot_titles)
            fig_singular_values_by_layer, axs = plt.subplots(nrows=number_of_layers, ncols=3, figsize=(10, 16))

            row = 0
            ## each layer goes on its own row, cols=0,1,2
            for layer_name, info in self.model_layer_info.items():
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

            fig_singular_values_by_layer.suptitle('Initial and Final Singular Values')

            ## final epoch vs layer plot, metrics for final layer only
            final_max_singular = []
            final_min_singular = []
            final_condition_numbers = []
            final_weight_means = []
            final_weight_stds = []
            final_max_abs_weight_mean_diffs = []
            layer_numbers = []

            # print("model_layer_info:\n")
            # print(self.model_layer_info)

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

                fig_min_singular_values.append_trace(
                    go.Scatter(
                        x=list(range(self._N_EPOCHS)), 
                        y=self.model_layer_info[f'layer_{i}']['min_singular_value_by_epoch'],
                        name=f'layer_{i}',
                        marker=dict(color=self._layer_color_map[f"layer_{i}"]),
                    ), row=row, col=col
                )
                
                fig_max_singular_values.append_trace(
                    go.Scatter(
                        x=list(range(self._N_EPOCHS)), 
                        y=self.model_layer_info[f'layer_{i}']['max_singular_value_by_epoch'],
                        name=f'layer_{i}',
                        marker=dict(color=self._layer_color_map[f"layer_{i}"]),
                    ), row=row, col=col
                )

                fig_weight_means.append_trace(
                    go.Scatter(
                        x=list(range(self._N_EPOCHS)), 
                        y=self.model_layer_info[f'layer_{i}']['weight_mean_by_epoch'],
                        marker=dict(color=self._layer_color_map[f"layer_{i}"]),
                        name=f"layer_{i}",
                    ), row=row, col=col
                )

                fig_weight_stds.append_trace(
                    go.Scatter(
                        x=list(range(self._N_EPOCHS)), 
                        y=self.model_layer_info[f'layer_{i}']['weight_std_by_epoch'],
                        marker=dict(color=self._layer_color_map[f"layer_{i}"]),
                        name=f"layer_{i}",
                    ), row=row, col=col
                )

                if isinstance(self.model, DiagDNN):
                    fig_diag_params.append_trace(
                        go.Scatter(
                            x=list(range(self._N_EPOCHS)), 
                            y=self.model_layer_info[f'layer_{i}']['diag_params_by_epoch'],
                            marker=dict(color=self._layer_color_map[f"layer_{i}"]),
                            name=f"layer_{i}",
                        ), row=row, col=col
                    )

                ## append to final layer metrics
                final_max_singular.append(self.model_layer_info[f'layer_{i}']['max_singular_value_by_epoch'][-1])
                final_min_singular.append(self.model_layer_info[f'layer_{i}']['min_singular_value_by_epoch'][-1])
                final_condition_numbers.append(self.model_layer_info[f'layer_{i}']['cond_number_by_epoch'][-1])

                final_weight_means.append(self.model_layer_info[f'layer_{i}']['weight_mean_by_epoch'][-1])
                final_weight_stds.append(self.model_layer_info[f'layer_{i}']['weight_std_by_epoch'][-1])

                final_max_abs_weight_mean_diffs.append(
                    self.model_layer_info[f'layer_{i}']['weight_abs_max_by_epoch'][-1] - 
                    self.model_layer_info[f'layer_{i}']['weight_abs_min_by_epoch'][-1]
                )
                layer_numbers.append(f'layer_{i}')
            
            df_final_epoch_stats = pd.DataFrame({
                'max_singular': final_max_singular, 
                'min_singular': final_min_singular,
                'condition_numbers': final_condition_numbers,
                'weight_means': final_weight_means,
                'weight_stds': final_weight_stds,
                'max_abs_weight_mean_diffs': final_max_abs_weight_mean_diffs,
                'layer_numbers': layer_numbers,
            })

            first_layer_name = 'layer_1'
            last_layer_name = f"layer_{self.model._N_LAYERS}"
            df_final_epoch_stats = df_final_epoch_stats[
                (df_final_epoch_stats['layer_numbers'] != first_layer_name) & 
                (df_final_epoch_stats['layer_numbers'] != last_layer_name)
            ]

            df_final_epoch_stats = df_final_epoch_stats.set_index('layer_numbers')
            df_final_epoch_stats_normalized = df_final_epoch_stats.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 0)
            feature_cols = df_final_epoch_stats_normalized.columns
            df_final_epoch_stats_normalized = df_final_epoch_stats_normalized.reset_index()
            
            fig_final_epoch_by_layer = px.line(df_final_epoch_stats_normalized, x='layer_numbers', y=feature_cols)
            fig_final_epoch_by_layer.update_layout(title=f"""DNN with {self.model._N_LAYERS} layers + normalization_type = {self.normalization_type} + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}""",
                              xaxis_title='Layer Number', yaxis_title='Normalized Value')

            fig_condition_numbers.update_layout(title=f"""
                            DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}: final train acc = {final_train_accuracy:.2%}, final test_acc = {final_test_accuracy:.2%}
                            <br>Condition Number of Layer Weights by Epoch ({self.dataset_name}, normalization_type = {self.normalization_type}, seed = {self.model.random_seed})<br>""")
        
            fig_weight_stds.update_layout(title=f"""
                            DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}: final train acc = {final_train_accuracy:.2%}, final test_acc = {final_test_accuracy:.2%}
                            <br>Standard Deviation of Layer Weights by Epoch ({self.dataset_name}, normalization_type = {self.normalization_type}, seed = {self.model.random_seed})<br>""")
            
            fig_accuracies = go.Figure()
            fig_accuracies.add_trace(go.Scatter(x=list(range(self._N_EPOCHS)), y=self.train_accuracies, name='train accuracy'))
            fig_accuracies.add_trace(go.Scatter(x=list(range(self._N_EPOCHS)), y=self.test_accuracies, name='test accuracy'))
            fig_accuracies.update_layout(title=f"""
                            DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}: train acc = {final_train_accuracy:.2%}, final test_acc = {final_test_accuracy:.2%}
                            <br>Train and Test Accuracies by Epoch ({self.dataset_name}, normalization_type = {self.normalization_type}, lr = {self.best_lr}, seed = {self.model.random_seed})<br>""")
            
            fig_learning_rates = go.Figure()
            fig_learning_rates.add_trace(go.Scatter(x=list(range(self._N_EPOCHS)), y=self.learning_rates, name='learning rate'))
            fig_learning_rates.update_layout(title=f"""
                            DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS}: train acc = {final_train_accuracy:.2%}, final test_acc = {final_test_accuracy:.2%}
                            <br>Learning Rates by Epoch ({self.dataset_name}, normalization_type = {self.normalization_type}, lr = {self.best_lr}, seed = {self.model.random_seed})<br>""")
            
            self.all_figures['learning_rates_by_epoch'] = fig_learning_rates
            self.all_figures['condition_numbers_by_epoch'] = fig_condition_numbers
            self.all_figures['min_singular_values_by_epoch'] = fig_min_singular_values
            self.all_figures['max_singular_values_by_epoch'] = fig_max_singular_values
            self.all_figures['singular_values_by_layer'] = fig_singular_values_by_layer
            self.all_figures['final_epoch_by_layer'] = fig_final_epoch_by_layer
            self.all_figures['weight_means_by_epoch'] = fig_weight_means
            self.all_figures['weight_stds_by_epoch'] = fig_weight_stds
            self.all_figures['accuracies_by_epoch'] = fig_accuracies

            if isinstance(self.model, DiagDNN):
                self.all_figures['diag_params_by_epoch'] = fig_diag_params
            
        ################################################ 
        # FOR PRELOADED MODEL, ONLY PLOT FINAL WEIGHTS #
        ################################################
        
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

    ############################################################
    #####  EXPERIMENT 2: freeze layers and then add noise  #####
    ############################################################

    def create_models_with_noise(self, N_NOISE_SAMPLES=10, noise_vars=[0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], noise_types=["input_dim","output_dim","layer_variance"]):
        
        np.random.seed(self.noise_random_seed)
        try:
            device = next(self.model.parameters()).device
        except:
            device = None
        
        self._noise_color_map = {
            f"noise_var_{x}": self._color_palette[i-1]
            for i,x in enumerate(noise_vars)
        }

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
            for noise_type in noise_types:
                self.all_layer_noise_test_acc[noise_type]["noise_vars"] = noise_vars
            
            n_stack_layers = []
            for name, module in self.model.named_modules():
                print(f"layer name: {name} of type {type(module)}")
                # layer name: linear_relu_stack.0
                # layer name: linear_relu_stack.1
                # ...

                if isinstance(module, nn.modules.linear.Linear):
                    layer_num = name.split('.')[-1]
                    n_stack_layers.append(layer_num)
            
            # n_stack_layers = [int(''.join(filter(str.isdigit, layer))) for layer in self.model.state_dict() if (("weight" in layer) and ("classifier" not in layer))]
            print(n_stack_layers)
        
        ## a preloaded CNN model has a different architecture
        else:
            # get standard deviation corresponding to the largest layer

            ## store the noise vars to be plotted (true noise vars may be scaled)
            for noise_type in noise_types:
                self.all_layer_noise_test_acc[noise_type]["noise_vars"] = noise_vars
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

                print(f"layer {weight_layer_string} has shape {layer_weights.shape}")

        for n, n_stack in enumerate(n_stack_layers):
            ## for a particular layer (n_stack), keep track of the test_accuracy vs noise
            ## for figures and layers, we use n to refer to layer 1, ... layer N of the DNN
        
            for noise_type in noise_types:
                layer_avg_test_accs = []
                
                for noise_var in noise_vars: 
                    test_acc_sum = 0

                    ## layer weights have dimension (output_dim, input_dim)
                    if not self.preloaded:
                        if noise_type == "input_dim":
                            layer_width = self.model_layer_info[f"layer_{n+1}"]["final_weights"].shape[1]
                            true_noise_var =  noise_var / layer_width
                        elif noise_type == "output_dim":
                            layer_width = self.model_layer_info[f"layer_{n+1}"]["final_weights"].shape[0]
                            true_noise_var =  noise_var / layer_width
                        elif noise_type == "layer_variance":
                            weight_layer_stdev = self.model_layer_info[f"layer_{n+1}"]["weight_std_by_epoch"][-1]
                            true_noise_var =  weight_layer_stdev**2 * noise_var
                    
                    if self.preloaded:
                        true_noise_var = noise_var

                    ## average test accuracy over N_NOISE_SAMPLES
                    for _ in range(N_NOISE_SAMPLES):
                        
                        ## create new instance of base model to add noise to
                        # new_experiment = copy.deepcopy(self)

                        ## find the layer, add noise
                        with torch.no_grad():
                            if isinstance(self.model, DNN):
                                weight_layer_string = f"linear_relu_stack.{n_stack}.weight"
                                bias_layer_string = f"linear_relu_stack.{n_stack}.bias"
                            elif isinstance(self.model, DiagDNN):
                                weight_layer_string = f"all_layers.{n_stack}.weight"
                                bias_layer_string = f"all_layers.{n_stack}.bias"
                            else:
                                raise Exception(f"model of type {type(self.model)} not supported")

                            
                            print(f"adding noise to {weight_layer_string} weights")

                            weight_layer_size = self.model.state_dict()[weight_layer_string].data.shape
                            bias_layer_size = self.model.state_dict()[bias_layer_string].data.shape
                            # print(f"size = {size}")

                            weight_noise = torch.normal(mean=0, std=np.sqrt(true_noise_var), size=weight_layer_size).to(device)
                            bias_noise = torch.normal(mean=0, std=np.sqrt(true_noise_var), size=bias_layer_size).to(device)

                            ## add noise to both (bias) and weight layers
                            ## bias_layer_string = f"features.{n_stack}.bias"

                            self.model.state_dict()[weight_layer_string].data += weight_noise
                            self.model.state_dict()[bias_layer_string].data += bias_noise
                            
                            ## after noise has been added to a layer, get the accuracy
                            test_acc = self.get_test_accuracy()
                            print(f"test accuracy for {weight_layer_string} = {test_acc:.2%}")
                            
                            ## reset model weights and biases
                            self.model.state_dict()[weight_layer_string].data -= weight_noise
                            self.model.state_dict()[bias_layer_string].data -= bias_noise

                        test_acc_sum += test_acc

                    avg_test_acc = test_acc_sum/N_NOISE_SAMPLES
                    layer_avg_test_accs.append(avg_test_acc)
                    print(f"Average test accuracy for {weight_layer_string} with N = {N_NOISE_SAMPLES} samples of noise at vars = {noise_var} using noise type {noise_type}: {avg_test_acc:.2%}")

                self.all_layer_noise_test_acc[noise_type][f'layer_{n+1}'] = np.array(layer_avg_test_accs)

    def create_accuracy_vs_noise_plots(self, noise_types):

        row = 1
        for noise_type in noise_types:
            df_accuracy_vs_noise = pd.DataFrame(self.all_layer_noise_test_acc[noise_type])
            df_accuracy_vs_noise = df_accuracy_vs_noise.sort_values(by='noise_vars', ascending=True)
            print(df_accuracy_vs_noise)
            self.noise_experiment_data = df_accuracy_vs_noise
            
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

            ## look into why this figure is only saving for the weight normalization experiment!
            df_layer_accuracy_vs_noise = df_accuracy_vs_noise.set_index('noise_vars').T
            print(df_layer_accuracy_vs_noise)
            fig_noise_acc_vs_layer = go.Figure()

            for noise_var in df_layer_accuracy_vs_noise.columns:
                fig_noise_acc_vs_layer.add_trace(go.Scatter(
                    x=df_layer_accuracy_vs_noise.index.tolist(),
                    y=df_layer_accuracy_vs_noise[noise_var],
                    name=f"noise = {noise_var}",
                    marker=dict(color=self._noise_color_map[f"noise_var_{noise_var}"]),
                ))
            fig_noise_acc_vs_layer.update_layout(title=f"""DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS} with training accuracy = {self.train_accuracy:.2%}
                                <br>Test Accuracy vs Layer by {noise_type} Noise ({self.dataset_name}, seed = {self.noise_random_seed})""")
            
            if self.preloaded:
                
                self.test_accuracy = self.get_test_accuracy()

                title = f"""Preloaded CNN with test accuracy = {self.test_accuracy:.2%}
                                <br>Test Accuracy vs {noise_type} Noise by layer ({self.dataset_name})"""
            else:
                title = f"""DNN with {self.model._N_LAYERS} layers + layer widths of {self.model._HIDDEN_LAYER_WIDTHS} with training accuracy = {self.train_accuracy:.2%}
                                <br>Test Accuracy vs {noise_type} Noise by layer ({self.dataset_name}, seed = {self.noise_random_seed})"""
            
            fig_noise_vs_accuracy.update_layout(title=title, xaxis_title='noise vars (unscaled)', height=1200)
            print(f"saving {noise_type}_noise_test_accuracies to all_figures")
            self.all_figures[f'{noise_type}_noise_test_accuracies'] = fig_noise_vs_accuracy
        
        self.all_figures['noise_acc_vs_layer'] = fig_noise_acc_vs_layer

def run_dnn_experiments(
    dnn_experiments, 
    directory, 
    dataset_name,
    dnn_learning_rates_dict=None, 
    N_EPOCHS=None, 
    MAX_TRAIN_ATTEMPTS=5, 
    regularizer=None, 
    debug=True, 
    noise_experiments=False, 
    noise_vars=[], 
    noise_types=["input_dim","output_dim","layer_variance"],
):
    """
    Convenience method to call DNN experiments with different initializations. 

    This function does one of the following: 
    (1) calls the dnn_experiment.train_base_model method with different learning rates
    (2) calls the dnn_experiment.create_models_with_noise method with all 3 types of noise (input layer, output layer, or variance of layer)

    Both of the above methods handle creation of all figures as well
    """

    ##############
    ## TRAINING ##
    ##############
    if not noise_experiments:
        dnn_experiment_results = {}
        for experiment_name, dnn in dnn_experiments.items():
            print(f"Starting training for {experiment_name}...")
            dnn_experiment = DnnLayerWeightExperiment(model=dnn, dataset_name=dataset_name, noise_random_seed=42, preloaded=False)
            dnn_experiment.load_dataset()

            train_attempt = 0
            while(True):
                if train_attempt < MAX_TRAIN_ATTEMPTS:
                    try:
                        print("Calling train_base_model...")
                        dnn_experiment.train_base_model(
                            N_EPOCHS=N_EPOCHS, 
                            dnn_learning_rates=dnn_learning_rates_dict[experiment_name], 
                            regularizer=regularizer,
                        ) 
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
                base_file_path = f"{directory}/{experiment_name}/{weight_type}_"

                ## display figures regardless
                dnn_experiment.all_figures[weight_type]['histogram'].write_html(base_file_path + "histogram.html")
                dnn_experiment.all_figures[weight_type]['singular_values'].write_html(base_file_path + "singular_values.html")
                # dnn_experiment.all_figures[weight_type]['heatmap'].write_html(base_file_path + "heatmap.html")
            
            ## file path name for epoch based figures
            base_file_path_other_figs = f"{directory}/{experiment_name}/"
            dnn_experiment.all_figures['learning_rates_by_epoch'].write_html(base_file_path_other_figs + "learning_rates_by_epoch.html")
            dnn_experiment.all_figures['condition_numbers_by_epoch'].write_html(base_file_path_other_figs + "condition_numbers_by_epoch.html")
            dnn_experiment.all_figures['min_singular_values_by_epoch'].write_html(base_file_path_other_figs + "min_singular_values_by_epoch.html")
            dnn_experiment.all_figures['max_singular_values_by_epoch'].write_html(base_file_path_other_figs + "max_singular_values_by_epoch.html")

            plt.savefig(base_file_path_other_figs + "singular_values_by_layer.pdf")
            plt.clf()

            dnn_experiment.all_figures['final_epoch_by_layer'].write_html(base_file_path_other_figs + "final_epoch_by_layer.html")
            dnn_experiment.all_figures['weight_means_by_epoch'].write_html(base_file_path_other_figs + "weight_means_by_epoch.html")
            dnn_experiment.all_figures['weight_stds_by_epoch'].write_html(base_file_path_other_figs + "weight_stds_by_epoch.html")
            dnn_experiment.all_figures['accuracies_by_epoch'].write_html(base_file_path_other_figs + "accuracies_by_epoch.html")
            
            if isinstance(dnn_experiment.model, DiagDNN):
                dnn_experiment.all_figures['diag_params_by_epoch'].write_html(base_file_path_other_figs + "diag_params_by_epoch.html")

            ## store experimental data from training
            # if dnn.normalization_type.upper() == 'WEIGHT':
            #     print("for weight norm, convert models to their state dictionaries")
            #     dnn_experiment.base_model = dnn_experiment.base_model.state_dict()
            #     dnn_experiment.model = dnn_experiment.model.state_dict()

            dnn_experiment_results[experiment_name] = dnn_experiment
            
        return dnn_experiment_results
    
    #######################
    ## NOISE EXPERIMENTS ##
    #######################

    if noise_experiments:
        dnn_experiment_results = {}
        for experiment_name, dnn_experiment in dnn_experiments.items():
            print(f"Running layer noise experiments for {experiment_name}...")
            if debug:
                dnn_experiment.create_models_with_noise(N_NOISE_SAMPLES=2, noise_vars=noise_vars, noise_types=noise_types)
            else:
                dnn_experiment.create_models_with_noise(N_NOISE_SAMPLES=100, noise_vars=noise_vars, noise_types=noise_types)
            
            dnn_experiment.create_accuracy_vs_noise_plots(noise_types=noise_types)

            ## save the figures
            for k,v in dnn_experiment.all_figures.items():
                # print(f"{k} of type {type(v)}")
                if '_noise_test_accuracies' in k:
                    file_path = f"{directory}/{experiment_name}/{k}.html"
                    dnn_experiment.all_figures[k].write_html(file_path)
            dnn_experiment_results[experiment_name] = dnn_experiment

            ## this was leading to the figure being saved only once
            dnn_experiment.all_figures['noise_acc_vs_layer'].write_html(f"{directory}/{experiment_name}/noise_acc_vs_layer.html")
        return dnn_experiment_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run DNN and weight noise experiments")
    parser.add_argument("--experiment-version", type=str, help="Experiment version number for creating folders to hold results, should be a string of the form v1, v2...", required=True)
    parser.add_argument("--experiment-type", type=str, help="Train a DNN, load a saved DNN, or load a pretrained CNN", choices=["dnn-train", "dnn-load-model", "cnn"], required=True)
    parser.add_argument("--dataset-name", type=str, help="Select from mnist, fashion_mnist, cifar10", choices=["mnist", "fashion_mnist", "cifar10"], required=True)
    parser.add_argument("--learning-rates-dict", help="Pass a dictionary of learning rates for each dnn experiment", required=False)
    parser.add_argument("--regularizer", help="Add a regularizer to training", choices=["l2","layer-l2","none"], required=False)
    parser.add_argument("--lr-scheduler", help="Decide which learning rate scheduler to use", choices=["Linear","Warmup","WarmRestart"], required=True)
    parser.add_argument("--pretrained-model-name", type=str, help="File name of pretrained DNN/CNN (stored as .pth)", required=False)
    parser.add_argument("--noise-vars", help="Pass a list of non-normalized noise variances that are added to a model", required=False)
    parser.add_argument("--noise-type", help="Pass a noise type or generate figures for all possible types of noise", choices=["input_dim","output_dim","layer_variance","all"], default="all", required=False)
    parser.add_argument("--cloud-environment", type=str, help="Local or Tufts HPC", choices=["local","hpc"], required=True)
    parser.add_argument("--debug-mode", type=str, help="Debug Mode or actual experiment", choices=["debug","experiment"], required=True)
    parser.add_argument("--add-diagonal-parameter", type=str, help="Add small diagonal matrix learnable parameter to weights during training", default="false")
    parser.add_argument("--normalization-type", type=str, help="Add normalization before RELU during training", choices=["batch","layer","weight","none"], required=True)
    args = parser.parse_args()

    version = args.experiment_version
    directory = f"experiment_plots/{version}"
    dataset_name = args.dataset_name

    cloud_environment = args.cloud_environment
    regularizer = args.regularizer
    debug = True if args.debug_mode == "debug" else False
    if debug:
        N_EPOCHS = 20
    else:
        # print(torch.__version__)
        # print(torch.cuda.is_available())
        # print(torch.cuda.device_count())

        N_EPOCHS = 100

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
    add_diagonal_parameter = True if str(args.add_diagonal_parameter).upper() == 'TRUE' else False
    normalization_type = args.normalization_type
    lr_scheduler = args.lr_scheduler
    if experiment_type.upper() == 'DNN-TRAIN':

        ## for debugging, set N_EPOCHS = 10
        ## when running the job, set N_EPOCHS = 200 or more
        
        print("----- Running all DNN experiments -----")

        start = time.time()

        random_seed = 42
        noise_random_seed = 42

        if debug:
            if not add_diagonal_parameter:
                dnn2a = DNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[1024, 512, 256, 128], random_seed=random_seed, init_type="uniform", normalization_type=normalization_type)
                dnn2b = DNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[1024, 512, 256, 128], random_seed=random_seed, init_type="normal", normalization_type=normalization_type)
                dnn_experiments = {
                    'dnn2a': dnn2a,
                    'dnn2b': dnn2b,
                }
            else:
                diagDnn = DiagDNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[1024, 1024, 1024, 1024], random_seed=random_seed, init_type="uniform", normalization_type=None)
                dnn_experiments = {
                    'diagDnn': diagDnn
                }
        else:
            # vBATCH, vLAYER, vWEIGHT all share these!
            if dataset_name.upper() == 'MNIST':
                if not add_diagonal_parameter:
                    dnn2a = DNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[128, 128, 128, 128, 128, 128, 128], random_seed=random_seed, init_type="uniform", normalization_type=normalization_type)
                    dnn2b = DNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[128, 128, 128, 128, 128, 128, 128], random_seed=random_seed, init_type="normal", normalization_type=normalization_type)
                    dnn2c = DNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[256, 224, 196, 172, 150, 128, 100], random_seed=random_seed, init_type="uniform", normalization_type=normalization_type)
                    dnn2d = DNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[256, 224, 196, 172, 150, 128, 100], random_seed=random_seed, init_type="normal", normalization_type=normalization_type)
                else:
                    dnn2a = DiagDNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[128, 128, 128, 128, 128, 128, 128], random_seed=random_seed, init_type="uniform", normalization_type=None)
                    dnn2b = DiagDNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[128, 128, 128, 128, 128, 128, 128], random_seed=random_seed, init_type="normal", normalization_type=None)
                    dnn2c = DiagDNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[256, 224, 196, 172, 150, 128, 100], random_seed=random_seed, init_type="uniform", normalization_type=None)
                    dnn2d = DiagDNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[256, 224, 196, 172, 150, 128, 100], random_seed=random_seed, init_type="normal", normalization_type=None)
                dnn_experiments = {
                    'dnn2a': dnn2a,
                    'dnn2b': dnn2b,
                    'dnn2c': dnn2c,
                    'dnn2d': dnn2d,
                }
            elif dataset_name.upper() == 'CIFAR10':
                if not add_diagonal_parameter:
                    dnn2a = DNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[1024, 1024, 1024, 1024, 1024, 1024, 1024], random_seed=random_seed, init_type="uniform", normalization_type=normalization_type)
                    dnn2b = DNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[1024, 1024, 1024, 1024, 1024, 1024, 1024], random_seed=random_seed, init_type="normal", normalization_type=normalization_type)
                    dnn2c = DNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[1024, 768, 512, 384, 256, 192, 128], random_seed=random_seed, init_type="uniform", normalization_type=normalization_type)
                    dnn2d = DNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[1024, 768, 512, 384, 256, 192, 128], random_seed=random_seed, init_type="normal", normalization_type=normalization_type)
                else:
                    dnn2a = DiagDNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[1024, 1024, 1024, 1024, 1024, 1024, 1024], random_seed=random_seed, init_type="uniform", normalization_type=None)
                    dnn2b = DiagDNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[1024, 1024, 1024, 1024, 1024, 1024, 1024], random_seed=random_seed, init_type="normal", normalization_type=None)
                    dnn2c = DiagDNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[1024, 768, 512, 384, 256, 192, 128], random_seed=random_seed, init_type="uniform", normalization_type=None)
                    dnn2d = DiagDNN(N_CLASSES=10, DATASET_NAME=dataset_name, HIDDEN_LAYER_WIDTHS=[1024, 768, 512, 384, 256, 192, 128], random_seed=random_seed, init_type="normal", normalization_type=None)
                dnn_experiments = {
                    'dnn2a': dnn2a,
                    'dnn2b': dnn2b,
                    'dnn2c': dnn2c,
                    'dnn2d': dnn2d,
                }
            else:
                raise Exception(f"dataset {dataset_name} not supported!")

        dnn_learning_rates_dict = eval(args.learning_rates_dict)
        dnn_experiments_results = run_dnn_experiments(
            dnn_experiments, 
            directory=directory, 
            dataset_name=dataset_name,
            dnn_learning_rates_dict=dnn_learning_rates_dict, 
            N_EPOCHS=N_EPOCHS, 
            MAX_TRAIN_ATTEMPTS=5, 
            regularizer=regularizer, 
            debug=debug,
        )
        
        ## save dnn_experiments_results using torch
        if normalization_type.upper() == 'WEIGHT':
            ## don't save the results... just proceed to noise experiments?

            noise_vars = eval(args.noise_vars)
            noise_type = args.noise_type
            if noise_type == "all":
                noise_types = ["input_dim","output_dim","layer_variance"]
            else:
                noise_types = [noise_type]
            
            dnn_experiments_results = run_dnn_experiments(
                dnn_experiments=dnn_experiments_results, 
                directory=directory, 
                dataset_name=dataset_name,
                dnn_learning_rates_dict={}, 
                N_EPOCHS=None, 
                MAX_TRAIN_ATTEMPTS=None, 
                regularizer=None, 
                debug=debug,
                noise_experiments=True,
                noise_vars=noise_vars,
                noise_types=noise_types,
            )

            for name in dnn_experiments_results.keys():
                dnn_experiments_results[name].base_model = dnn_experiments_results[name].base_model.state_dict()
                dnn_experiments_results[name].model = dnn_experiments_results[name].model.state_dict()
            
            ## overwrite the old version!
            torch.save(dnn_experiments_results, f'{directory}/dnn_experiments_results.pth')
            
        else:
            torch.save(dnn_experiments_results, f'{directory}/dnn_experiments_results.pth')

        end = time.time()
        runtime = (end - start) / 60
        print(f"Total runtime = {runtime:.2f} minutes")
    
    elif experiment_type.upper() == 'DNN-LOAD-MODEL':
        model_name = args.pretrained_model_name
        full_model_path = directory + '/' + model_name
        print(f"loading model from {full_model_path}")
        dnn_experiments_results = torch.load(full_model_path, weights_only=False)
        # {
        #     'dnn1': trained_dnn1,
        #     'dnn2': trained_dnn2
        # }

        noise_vars = eval(args.noise_vars)
        noise_type = args.noise_type
        if noise_type == "all":
            noise_types = ["input_dim","output_dim","layer_variance"]
        else:
            noise_types = [noise_type]

        ## call the noise layer weight experiments on all models
        dnn_experiments_results = run_dnn_experiments(
            dnn_experiments=dnn_experiments_results, 
            directory=directory, 
            dataset_name=dataset_name,
            dnn_learning_rates_dict={}, 
            N_EPOCHS=None, 
            MAX_TRAIN_ATTEMPTS=None, 
            regularizer=None, 
            debug=debug,
            noise_experiments=True,
            noise_vars=noise_vars,
            noise_types=noise_types,
        )

        ## we can change this to overwrite once we're confident this works
        torch.save(dnn_experiments_results, f'{directory}/dnn_experiments_results.pth')

    else:
        raise Exception(f"experiment type of {experiment_type} not supported")
    
    ## VGG architecture on CIFAR10
    if experiment_type.upper() == 'CNN':
        start = time.time()
        print("----- Running CNN experiments with VGG models -----")

        if debug:
            pass
        else:
            pass

        end = time.time()
        runtime = (end - start) / 60
        print(f"Total runtime = {runtime:.2f} minutes")