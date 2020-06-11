import os
import sys

import datetime
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from keras.utils import to_categorical
from sklearn.metrics import log_loss
from collections import OrderedDict
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.append('..')
from helpers.settings import arrays_folder, models_folder


def top_k_accuracy(y_true, y_pred, k=1):
    """From: https://github.com/chainer/chainer/issues/606
    Expects both y_true and y_pred to be one-hot encoded.
    """
    argsorted_y = np.argsort(y_pred)[:, -k:]
    return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()


model = InceptionV3()
model_uid = 'inceptionv3'

if __name__ == '__main__':
    input_folder = os.path.join(arrays_folder, 'imagenet_299')
    output_folder = os.path.join(models_folder, model_uid)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    y_test = np.load(os.path.join(input_folder, 'y_val.npy'))
    y_test = to_categorical(y_test, 1000)
    ws = model.get_weights()

    n_batch = 50
    samples_per_batch = 50000 // n_batch

    # todo: update the parameters
    # todo: change the way we are saving nlls

    n_start_batch = 5
    n_selected_batch = 45
    n_ensemble = 10

    df = pd.read_csv(os.path.join(output_folder, 'golden_optimization_simgas.csv'))
    sigma_1 = df[df.iteration == 6].sigma_1.values[0]
    sigma_2 = df[df.iteration == 6].sigma_2.values[0]
    sigma_optimum = sigma_2 + (sigma_1 - sigma_2) / 2
    for i in range(n_start_batch, n_start_batch + n_selected_batch):
        images_path = os.path.join(input_folder, 'x_val_' + str(i).zfill(3) + '.npy')
        array = np.load(images_path)
        X_test_batch = preprocess_input(array)
        y_test_batch = y_test[i * samples_per_batch:(i + 1) * samples_per_batch]
        y_test_pred_ensemble = np.zeros((n_ensemble, *y_test_batch.shape), dtype=np.float32)
        y_pred_base = model.predict(X_test_batch, verbose=1)
        nll_base = log_loss(y_test_batch, y_pred_base)
        print('-'*100)
        print('nll base {0:.4f}'.format(nll_base))
        print('-'*100)
        for seed_index, seed in enumerate(range(17, 17 + n_ensemble)):
            np.random.seed(seed)
            model.set_weights(ws)
            wp = np.copy(ws)
            for index2, w in enumerate(ws):
                shape = w.shape
                if len(shape) == 4:
                    noise = np.random.normal(0, sigma_optimum, (w.shape[0], w.shape[1], w.shape[2], w.shape[3]))
                    wp[index2] = ws[index2] + noise
            model.set_weights(wp)
            y_test_pred_p = model.predict(X_test_batch, batch_size=64, verbose=1)
            y_test_pred_ensemble[seed_index] = y_test_pred_p
            nll_test = log_loss(y_test_batch, y_test_pred_p)
            print('sigma: {0:.6f}, seed: {1}, nll: val {2:.4f}'.format(sigma_optimum, seed, nll_test))
        y_test_pred_ensemble_mean = np.mean(y_test_pred_ensemble, axis=0)
        nll_ensemble = log_loss(y_test_batch, y_test_pred_ensemble_mean)
        print('-'*100)
        print('nll ensemble {0:.4f}, improvement: {1:.4f}'.format(nll_ensemble, nll_base-nll_ensemble))
        print('-'*100)
        np.save(os.path.join(output_folder, 'y_test_' + str(i) + '.npy'), y_test_batch)
        np.save(os.path.join(output_folder, 'y_pred_base_' + str(i) + '.npy'), y_pred_base)
        np.save(os.path.join(output_folder, 'y_pred_ensemble_' + str(i) + '.npy'), y_test_pred_ensemble)
