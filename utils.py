import numpy as np
from scipy.optimize import leastsq
from torch import nn
import torch

def fit_gaussian_curve(x_data, y_data, mean, stddev):
    """Fit a Gaussian curve to (x,y) data points taken from a histogram, and return the fitted Gaussian function"""
    fitfunc = lambda p, x: p[0]*(1/(stddev*np.sqrt(2*np.pi))) * np.exp(-(1/2) * ((x_data-mean)/stddev)**2)
    errfunc = lambda p, x, y: (y - fitfunc(p, x))
    init = [1.0]
    out = leastsq(errfunc, init, args=(x_data, y_data))
    coef = out[0][0]
    return lambda x: coef * (1/(stddev*np.sqrt(2*np.pi))) * np.exp(-(1/2) * ((x-mean)/stddev)**2)

def init_weights(m, init_type, random_seed):
    if init_type == 'normal':
        if isinstance(m, nn.Linear):
            torch.cuda.manual_seed(random_seed)
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # m.bias.data.fill_(0)
    elif init_type == 'uniform':
        if isinstance(m, nn.Linear):
            torch.cuda.manual_seed(random_seed)
            torch.nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
            m.bias.data.fill_(0.01)
    else:
        raise Exception(f"init_type = {init_type} not implemented!")

def equalize_axes_layout(fig):
    for name in fig.layout:
        if 'xaxis' in name:
            fig.layout[name]['constrain'] = 'domain'
        if 'yaxis' in name:
            x_anchor = fig.layout[name]['anchor']
            fig.layout[name]['scaleanchor'] = x_anchor
    fig.update_layout(height=2000)
    return fig