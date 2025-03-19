The goal of these experiments are to throughly investigate the results in the paper "On the Expressive Power of Deep Neural Networks" (2017) where one of the key findings is that for their trained ConvNet, the lower layers are more sensitive to noise than deeper layers. So far, the experiments have been inconclusive.

To debug or run experiments locally, set the following flags. There are two types of experiments that can be run: (1) model training with learning rates passed in the command line arguments, and (2) noise layer experiments with the base amount of noise passed in the command line arguments

For example, to train and save the figures for 4 different DNNs (whose architectures are specified in dictionaries dnn2a, dnn2b, dnn2c, dnn2d in `dnn_weight_noise.py`), with batch normalization on the CIFAR10 dataset, you would run the following command:

export PYTORCH_ENABLE_MPS_FALLBACK='1'
python dnn_weight_noise.py --experiment-version vBATCH-CIFAR10 --experiment-type dnn-train --dataset-name cifar10 --learning-rates-dict "{'dnn2a':[5e-3, 1e-4], 'dnn2b':[5e-3, 1e-4], 'dnn2c':[5e-3, 1e-4], 'dnn2d':[5e-3, 1e-4]}" --cloud-environment hpc --debug-mode experiment --normalization-type batch

This will save the model (and figures) in the location `experiment_plots/vBATCH-CIFAR10/dnn_experiments_results.pth`. To run the noise experiments (with noise scaled to the variance of each layer), you would run the following command:
python dnn_weight_noise.py --experiment-version vBATCH-CIFAR10 --experiment-type dnn-load-model --dataset-name cifar10 --pretrained-model-name "dnn_experiments_results.pth" --noise-vars "[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]" --noise-type "layer_variance" --cloud-environment hpc --debug-mode experiment --normalization-type batch

The default behavior is to generate the following figures for DNNs being trained from scratch.

Figures created during the training process (50+ epochs):
- train and test accuracy curves
- histogram of initial and final layer weight distributions
- heatmap of initial and final layer weights (legacy, difficult to interpret)
- condition number of layers by epoch
- standard deviation of layers by epoch
- mean of layers by epoch
- min and max singular values of layers

Figures associated with the layer weight noise experiments:
- accuracy curves by standard deviation of noise (averaged over 500 samples)

For pretrained CNNs (VGG11, VGG13, VGG16), we generate the following figures
- histogram of final layer weight distributions
- accuracy curves by standard deviation of noise (averaged over 500 samples)