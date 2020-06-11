import os
import sys

import datetime
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from keras.utils import to_categorical
from sklearn.metrics import log_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append('../..')
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
    output_folder = os.path.join(models_folder, model_uid + '_sigma_chart')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    y_val = np.load(os.path.join(input_folder, 'y_val.npy'))
    y_val_one_hot = to_categorical(y_val, 1000)
    ws = model.get_weights()
    steps = 50
    n = 50000 // steps
    sigmas = np.linspace(1e-5, 5e-3, 20)
    n_ensemble = 5
    for i in range(0, 5):
        now = datetime.datetime.now().strftime("%m/%d %H:%M:%S")
        print('=' * 100)
        print('started on {0} of {1}... time: {2}'.format(i, 5, now))
        print('=' * 100)
        images_path = os.path.join(input_folder, 'x_val_' + str(i).zfill(3) + '.npy')
        array = np.load(images_path)
        x = preprocess_input(array)
        gt = y_val_one_hot[i * n:(i + 1) * n]
        val_pred_sum = np.zeros_like(gt)
        model.set_weights(ws)
        y_pred_base = model.predict(x, verbose=1)
        top_1 = top_k_accuracy(gt, y_pred_base, k=1)
        top_5 = top_k_accuracy(gt, y_pred_base, k=5)
        ll = log_loss(gt, y_pred_base)
        print('nll: {0:.4f}, Top-1: {1:.4f}, Top-5: {2:4f}'.format(ll, top_1, top_5))
        for sigma in sigmas:
            output_sigma_folder = os.path.join(output_folder, str(sigma))
            if not os.path.isdir(output_sigma_folder):
                os.makedirs(output_sigma_folder)
            y_pred_ensemble = np.zeros((n_ensemble, * y_pred_base.shape))
            for index, seed in enumerate(range(17, 17 + n_ensemble)):
                np.random.seed(seed)
                model.set_weights(ws)
                wp = np.copy(ws)
                for index2, w in enumerate(ws):
                    shape = w.shape
                    if len(shape) == 4:
                        noise = np.random.normal(0, sigma, (w.shape[0], w.shape[1], w.shape[2], w.shape[3]))
                        wp[index2] = ws[index2]+noise
                model.set_weights(wp)
                y_pred = model.predict(x, verbose=1)
                top_1 = top_k_accuracy(gt, y_pred, k=1)
                top_5 = top_k_accuracy(gt, y_pred, k=5)
                ll = log_loss(gt, y_pred)
                y_pred_ensemble[index] = y_pred
                print('nll: {0:.4f}, Top-1: {1:.4f}, Top-5: {2:4f}'.format(ll, top_1, top_5))
                val_pred_sum += y_pred
            y_val_pred_ensemble = val_pred_sum / n_ensemble
            top_1 = top_k_accuracy(gt, y_val_pred_ensemble, k=1)
            top_5 = top_k_accuracy(gt, y_val_pred_ensemble, k=5)
            ll = log_loss(gt, y_val_pred_ensemble)
            print('nll: {0:.4f}, Top-1: {1:.4f}, Top-5: {2:4f}'.format(ll, top_1, top_5))
            np.save(os.path.join(output_sigma_folder, 'y_true_' + str(i) + '.npy'), y_val_one_hot)
            np.save(os.path.join(output_sigma_folder, 'y_pred_base_' + str(i) + '.npy'), y_pred_base)
            np.save(os.path.join(output_sigma_folder, 'y_pred_ensemble_' + str(i) + '.npy'), y_pred_ensemble)
