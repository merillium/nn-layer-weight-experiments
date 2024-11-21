The goal of these experiments are to reproduce the results in the paper "On the Expressive Power of Deep Neural Networks" (2017) where one of the key findings is that for their trained ConvNet, the lower layers are more sensitive to noise than deeper layers. So far, the experiments have been inconclusive.

To debug or run experiments locally, set the following flags. 

export PYTORCH_ENABLE_MPS_FALLBACK='1'
python -i dnn_weight_noise.py --experiment-version v1 --experiment-type cnn --cloud-environment local --debug-mode debug

The default behavior is to generate the following figures for DNNs being trained from scratch.

Figures created during the training process (400 epochs):
- train and test accuracy curves
- histogram of initial and final layer weight distributions
- heatmap of initial and final layer weights (legacy, difficult to interpret)
- condition number of layers by epoch
- standard deviation of layers by epoch

Figures associated with the layer weight noise experiments:
- accuracy curves by standard deviation of noise (averaged over 500 samples)

For pretrained CNNs (VGG11, VGG13, VGG16), we generate the following figures
- histogram of final layer weight distributions
- accuracy curves by standard deviation of noise (averaged over 500 samples)