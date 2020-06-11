import os
import sys
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from collections import OrderedDict
import pandas as pd

sys.path.append('../..')
from helpers.settings import models_folder, sheets_folder


if __name__ == '__main__':

    samples_per_batch = 1000
    n_classes = 1000
    # we use 5000 validation images to optimize ts
    n_selected = 5

    models = ['densenet121', 'densenet169', 'inceptionv3', 'resnet50', 'vgg16', 'vgg19']
    # models = ['densenet121', 'densenet169', 'inceptionv3', 'resnet50']

    optimal_temps = list()
    for model in models:
        y_pred_base = np.zeros((n_selected*samples_per_batch, n_classes))

        sigmas = np.logspace(-4, -2, 10)[:-1]
        image_net_model_folder = os.path.join(models_folder, 'imagenet_perturbed', model, '{0:.4f}'.format(sigmas[0]))

        for i in range(n_selected):
            base_path = os.path.join(image_net_model_folder, 'y_pred_base_' + str(i) + '_.npy')
            ensemble_path = os.path.join(image_net_model_folder, 'y_pred_ensemble_' + str(i) + '_.npy')
            y_pred_base[i * samples_per_batch: (i + 1) * samples_per_batch] = np.load(base_path)

        true_path = os.path.join(image_net_model_folder, 'y_true_' + str(0) + '_.npy')
        y_true = np.load(true_path)[:n_selected*samples_per_batch, ...]
        print(y_true.shape)

        def softmax_t(y_logit, t):
            return np.exp(y_logit/t)/np.sum(np.exp(y_logit/t), axis=-1)[:, np.newaxis]

        def ll_t(t):
            y_pred_base_logit = np.log(y_pred_base)
            y_pred_temp = softmax_t(y_pred_base_logit, t)
            ll = log_loss(y_true, y_pred_temp)
            return ll

        xopt = minimize(ll_t, 1, method='bfgs', options={'disp': 1})
        optimal_temp = xopt['x'][0]
        optimal_temps.append(OrderedDict({'model': model, 'temp': optimal_temp}))
    pd.DataFrame(optimal_temps).to_csv(os.path.join(sheets_folder, 'imagenet_optimal_temps.csv'))
