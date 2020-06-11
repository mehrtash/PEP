import glob
import pandas as pd
import os
import sys
import numpy as np
from collections import OrderedDict
from sklearn.metrics import log_loss
from keras.datasets import mnist
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append('../..')
from helpers.settings import models_folder

output_folder = os.path.join(models_folder, 'mnist_mlp')
output_baseline_folder = os.path.join(output_folder, 'baselines')


def load_mnist():
    num_classes = 10
    (x_train_val, y_train_val), (x_test, y_test) = mnist.load_data()
    x_train_val = x_train_val.reshape(x_train_val.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)

    x_train_val = x_train_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train_val /= 255
    x_test /= 255

    y_train_val = to_categorical(y_train_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    x_train = x_train_val[:50000]
    y_train = y_train_val[:50000]
    x_val = x_train_val[50000:]
    y_val = y_train_val[50000:]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()


class PPE:
    def __init__(self, model, n_ensemble):
        self.model = model
        self.n_ensemble = n_ensemble
        self.original_weights = model.get_weights()

    def perturb(self, sigma):
        self.model.set_weights(self.original_weights)
        weights = self.model.get_weights()
        for index2, w in enumerate(weights):
            shape = w.shape
            if len(shape) > 1 and shape[1] == 200:
                noise = np.random.normal(0, sigma, (w.shape[0], w.shape[1]))
                # noise = np.random.normal(0, sigma, (w.shape[0], w.shape[1]//2))
                # zero sum perturbations
                # noise = np.concatenate([noise, -noise], axis=-1)
                weights[index2] = weights[index2] + noise
        self.model.set_weights(weights)

    def ensemble_pred(self, x, sigma):
        y_pred = self.model.predict(x)
        y_pred_ensemble = np.zeros((self.n_ensemble, *y_pred.shape))
        for seed_index, seed in enumerate(range(17, 17 + self.n_ensemble)):
            np.random.seed(seed)
            self.perturb(sigma)
            y_pred = self.model.predict(x)
            y_pred_ensemble[seed_index] = y_pred
        self.model.set_weights(self.original_weights)
        return np.mean(y_pred_ensemble, axis=0)


def golden_search(model, sigma_lower=0, sigma_upper=0.1, n_ppe_ensemble=10, n_iterations=10):
    sigmas_log = list()
    nlls_log = list()
    nlls = dict()
    PHI = (np.sqrt(5) - 1) / 2
    for iteration in range(n_iterations):
        print('iteration {}'.format(iteration) + '-' * 100)
        sigma_1 = sigma_lower + (sigma_upper - sigma_lower) * PHI
        sigma_2 = sigma_lower + (sigma_1 - sigma_lower) * PHI
        sigmas_log.append(OrderedDict({"iteration": iteration, "sigma_l": sigma_lower,
                                       "sigma_2": sigma_2, "sigma_1": sigma_1,
                                       "sigma_h": sigma_upper}))
        for sigma in [sigma_1, sigma_2]:
            if sigma not in nlls.keys():
                ppe = PPE(model, n_ensemble=n_ppe_ensemble)
                y_val_ppe = ppe.ensemble_pred(x_val, sigma)
                nll = log_loss(y_val, y_val_ppe)
                nlls[sigma] = nll
                print(sigma, nll)
                nlls_log.append(OrderedDict({"iteration": iteration, "sigma": sigma, "nll": nll}))
            else:
                print('will skip for {0:6f}'.format(sigma))
        if nlls[sigma_1] < nlls[sigma_2]:
            sigma_lower = sigma_2
            print('updated lower side...')
        else:
            sigma_upper = sigma_1
            print('updated upper side...')
    sigma_optimal = (sigmas_log[-1]['sigma_l'] + sigmas_log[-1]['sigma_h']) / 2
    return sigma_optimal


def mlp(num_classes=10):
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(784,)))
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    return model


if __name__ == '__main__':
    optimal_sigmas_single = dict()
    d = list()
    n_models = 5
    for index, model_id in enumerate(range(17, 17 + n_models)):
        print('finding optimal sigma for model {}...'.format(model_id))
        model_folder = os.path.join(output_baseline_folder, str(model_id))
        checkpoints = sorted(glob.glob(model_folder + '/weights*.hdf5'))
        print(checkpoints)
        for checkpoint_path in checkpoints:
            model = mlp()
            print('loading weights...')
            model.load_weights(checkpoint_path)
            y_pred = model.predict(x_test)
            nll = log_loss(y_test, y_pred)
            print('running optimization...')
            optimal_sigma = golden_search(model)
            #
            model.load_weights(checkpoint_path)
            ppe = PPE(model, n_ensemble=25)
            y_pred = ppe.ensemble_pred(x_test, optimal_sigma)
            nll_ppe = log_loss(y_test, y_pred)
            #
            row = OrderedDict({
                "model": model_id,
                "checkpoint": os.path.basename(checkpoint_path),
                "optimal sigma": optimal_sigma,
                "nll": nll,
                "nll ppe": nll_ppe
            })
            print('=' * 100)
            print('=' * 100)
            print(row)
            print('=' * 100)
            print('=' * 100)
            d.append(row)
    pd.DataFrame(d).to_csv(os.path.join(output_baseline_folder, 'ppe_boost_vs_overfitting.csv'))
