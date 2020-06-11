import os
import glob

import numpy as np
from tqdm import tqdm
import cv2

import sys

sys.path.append('..')
from helpers.settings import raw_folder, arrays_folder

if __name__ == '__main__':
    all_images = sorted(glob.glob(os.path.join(raw_folder, 'ILSVRC2012_img_val') + '/*.JPEG'))
    steps = 50
    n = 50000 // steps
    output_shapes = [224, 299]
    for output_shape in output_shapes:
        output_folder = os.path.join(arrays_folder, 'imagenet_' + str(output_shape))
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        for i in range(steps):
            images = all_images[i * n: (i + 1) * n]
            array = np.zeros((len(images), output_shape, output_shape, 3), dtype='float32')
            for index, image_path in enumerate(tqdm(images)):
                # note: code from here: http://calebrob.com/ml/imagenet/ilsvrc2012/2018/10/22/imagenet-benchmarking.html
                # img = imread(image_path)
                img = cv2.imread(image_path)
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=-1)
                height, width, _ = img.shape
                crop_size = 256 if output_shape == 224 else 299
                new_height = height * crop_size // min(img.shape[:2])
                new_width = width * crop_size // min(img.shape[:2])
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                height, width, _ = img.shape
                startx = width // 2 - (output_shape // 2)
                starty = height // 2 - (output_shape // 2)
                img = img[starty:starty + output_shape, startx:startx + output_shape]
                assert img.shape[0] == output_shape and img.shape[1] == output_shape, (img.shape, height, width)
                img = np.expand_dims(img[:, :, ::-1], axis=0)
                array[index] = img
            np.save(os.path.join(output_folder, 'x_val_' + str(i).zfill(3) + '.npy'), array)
