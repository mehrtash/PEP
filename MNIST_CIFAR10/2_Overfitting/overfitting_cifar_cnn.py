import glob
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import log_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append('../..')
from helpers.settings import models_folder

output_folder = os.path.join(models_folder, 'cifar10_cnn')
output_baseline_folder = os.path.join(output_folder, 'baselines')


def load_cifar():
    num_classes = 10
    (x_train_val, y_train_val), (x_test, y_test) = cifar10.load_data()

    x_train_val = x_train_val.astype('float32')
    x_test = x_test.astype('float32')

    x_train_val /= 255
    x_test /= 255

    y_train_val = to_categorical(y_train_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    x_train = x_train_val[:45000]
    y_train = y_train_val[:45000]
    x_val = x_train_val[45000:]
    y_val = y_train_val[45000:]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar()


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
            if len(shape) >= 2 and shape[-1] > 32:
                if len(shape) == 4:
                    noise = np.random.normal(0, sigma, (w.shape[0], w.shape[1], w.shape[2], w.shape[3]))
                else:
                    noise = np.random.normal(0, sigma, (w.shape[0], w.shape[1]))
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
        print('iteration {} {} {} '.format(iteration, sigma_lower, sigma_upper) + '-'* 10)
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
    sigma_optimal = (sigmas_log[-1]['sigma_l'] + sigmas_log[-1]['sigma_h'])/2
    return sigma_optimal


def cnn():
    img_rows, img_cols = 32, 32
    input_shape = (img_rows, img_cols, 3)

    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization(momentum=.9))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(BatchNormalization(momentum=.9))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=.9))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization(momentum=.9))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization(momentum=.9))

    model.add(Dense(10))
    model.add(Activation('softmax'))

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
            model = cnn()
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
