import os
import sys

import datetime
import numpy as np
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input

from keras.utils import to_categorical
from sklearn.metrics import log_loss
from collections import OrderedDict
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append('..')
from helpers.settings import arrays_folder, models_folder


model = DenseNet121()
model_uid = 'densenet121'

PHI = (np.sqrt(5) - 1) / 2

if __name__ == '__main__':
    input_folder = os.path.join(arrays_folder, 'imagenet_224')
    output_folder = os.path.join(models_folder, model_uid)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    y_val = np.load(os.path.join(input_folder, 'y_val.npy'))
    y_val = to_categorical(y_val, 1000)
    ws = model.get_weights()
    n_batch = 50
    samples_per_batch = 50000 // n_batch

    n_optimization_iterations = 20
    n_selected_batch = 5
    n_ensemble = 5

    y_val = y_val[:n_selected_batch * samples_per_batch]
    X = np.zeros((n_selected_batch * samples_per_batch, 224, 224, 3))
    sigma_lower, sigma_upper = 5e-4, 5e-3

    sigmas_log = list()
    nlls_log = list()
    nlls = dict()
    for iteration in range(n_optimization_iterations):
        now = datetime.datetime.now().strftime("%m/%d %H:%M:%S")
        print('=' * 100)
        print('iteration {0}... time: {1}'.format(iteration, now))
        print('=' * 100)
        sigma_1 = sigma_lower + (sigma_upper - sigma_lower) * PHI
        sigma_2 = sigma_lower + (sigma_1 - sigma_lower) * PHI
        sigmas_log.append(OrderedDict({"iteration": iteration, "sigma_l": sigma_lower,
                                   "sigma_2": sigma_2, "sigma_1": sigma_1,
                                   "sigma_h": sigma_upper}))
        print('sigma: {0:.6f}, {1:.6f}, {2:.6f}, {3:.6f}'.format(sigma_lower, sigma_2, sigma_1, sigma_upper))
        for sigma in [sigma_1, sigma_2]:
            if sigma not in nlls.keys():
                y_val_pred_ensemble = np.zeros((n_ensemble, *y_val.shape))
                for seed_index, seed in enumerate(range(17, 17 + n_ensemble)):
                    np.random.seed(seed)
                    model.set_weights(ws)
                    wp = np.copy(ws)
                    for index2, w in enumerate(ws):
                        shape = w.shape
                        if len(shape) == 4:
                            noise = np.random.normal(0, sigma, (w.shape[0], w.shape[1], w.shape[2], w.shape[3]))
                            wp[index2] = ws[index2] + noise
                    model.set_weights(wp)
                    for i in range(0, n_selected_batch):
                        images_path = os.path.join(input_folder, 'x_val_' + str(i).zfill(3) + '.npy')
                        array = np.load(images_path)
                        X[i * samples_per_batch:(i + 1) * samples_per_batch] = preprocess_input(array)
                    y_val_pred_p = model.predict(X, batch_size=64, verbose=1)
                    y_val_pred_ensemble[seed_index] = y_val_pred_p
                    nll_val = log_loss(y_val, y_val_pred_p)
                    print('sigma: {0:.6f}, seed: {1}, nll: val {2:.4f}'.format(sigma, seed, nll_val))
                y_val_pred_ensemble_mean = np.mean(y_val_pred_ensemble, axis=0)
                nll = log_loss(y_val, y_val_pred_ensemble_mean)
                nlls[sigma] = nll
                nlls_log.append(OrderedDict({"iteration": iteration, "sigma": sigma, "nll": nll}))
            else:
                print('will skip for {0:6f}'.format(sigma))
            print(nlls_log)
        if nlls[sigma_1] < nlls[sigma_2]:
            sigma_lower = sigma_2
            print('updated lower side...')
        else:
            sigma_upper = sigma_1
            print('updated upper side...')
        print(sigma_lower, sigma_upper)
        pd.DataFrame(nlls_log).to_csv(os.path.join(output_folder, 'golden_optimization_nlls.csv'), index=False)
        pd.DataFrame(sigmas_log).to_csv(os.path.join(output_folder, 'golden_optimization_simgas.csv'), index=False)
