import numpy as np
import pandas as pd
from scipy.io import loadmat

import os
import sys
sys.path.append('..')

from helpers.settings import raw_folder, arrays_folder


def convert_original_idx_to_keras_idx(idx):
    return synset_to_keras_idx[original_idx_to_synset[idx]]


if __name__ == '__main__':
    output_folder = os.path.join(arrays_folder, 'imagenet_224')

    meta = loadmat(os.path.join(raw_folder, "meta.mat"))
    original_idx_to_synset = {}
    synset_to_name = {}

    for i in range(1000):
        ilsvrc2012_id = int(meta["synsets"][i, 0][0][0][0])
        synset = meta["synsets"][i, 0][1][0]
        name = meta["synsets"][i, 0][2][0]
        original_idx_to_synset[ilsvrc2012_id] = synset
        synset_to_name[synset] = name

    synset_to_keras_idx = {}
    keras_idx_to_name = {}
    f = open(os.path.join(raw_folder, "synset_words.txt"), "r")
    idx = 0
    for line in f:
        parts = line.split(" ")
        synset_to_keras_idx[parts[0]] = idx
        keras_idx_to_name[idx] = " ".join(parts[1:])
        idx += 1
    f.close()

    f = open(os.path.join(raw_folder, "ILSVRC2012_validation_ground_truth.txt"), "r")
    y_val = f.read().strip().split("\n")
    y_val = list(map(int, y_val))
    y_val = np.array([convert_original_idx_to_keras_idx(idx) for idx in y_val])
    f.close()

    df = pd.DataFrame(keras_idx_to_name.items())
    df.columns = ['idx', 'name']
    df.to_csv(os.path.join(output_folder, 'keras_ids_to_name.csv'), index=False)
    np.save(os.path.join(output_folder, "y_val.npy"), y_val)
