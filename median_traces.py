#!/usr/bin/env python
from __future__ import division

from glob import glob
from itertools import izip
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sklearn import svm, cross_validation
from sklearn.metrics import accuracy_score as accuracy

__author__ = "Michele Orru`"
__email__ = "michele.orru@studenti.unitn.it"
__license__ = """"THE BEER-WARE LICENSE" (Revision 42):
 maker wrote this file. As long as you retain this notice you
 can do whatever you want with this stuff. If we meet some day, and you think
 this stuff is worth it, you can buy me a beer in return.
"""

def g(a, b):
    p = np.sign(a*b)
    if p == -1: return 1
    if p == 1: return  0
    if p == 0: return -1

def neighbours(x, y):
    return [
        (x-1, y-1), (x, y-1), (x+1, y-1), (x+1, y),
        (x+1, y+1), (x, y+1), (x-1, y+1), (x-1, y)
    ]

def extract_dataset(dataset_type):
    base_path = join('dataset', dataset_type, '')

    try:
        return np.load(base_path + 'features.npy')
    except:
        features = []
        for image_file in glob(base_path + '*.tif'):
            print('processing: {}'.format(image_file))
            features.append(extract_feature(image_file))

        features = np.asarray(features)
        np.save(base_path + 'features', features)
        return features


directions = ('h', 'v', 'dl', 'dr')
make_histogram = lambda: np.zeros(256, dtype='int')

def extract_feature(image_file):
    # monochrome image
    img = np.asarray(Image.open(image_file).convert('L'), dtype='int32')

    # compute second_order ternary patterns
    cimg = img[1:-1, 1:-1]
    # first-order derivate
    i = {
        'h': cimg - img[1:-1, 2:],
        'v': cimg - img[:-2, 1:-1],
        'dr': cimg - img[:-2, 2:],
        'dl': cimg - img[:-2, :-2]
    }

    width, height = cimg.shape
    indices = ((x, y) for x in xrange(1, width-1) for y in xrange(1, height-1))

    histograms = dict(
        ltpp = {'h':  make_histogram(),
                'v':  make_histogram(),
                'dr': make_histogram(),
                'dl': make_histogram(),
        },
        ltpn = {'h':  make_histogram(),
                'v':  make_histogram(),
                'dr': make_histogram(),
                'dl': make_histogram(),
        }
    )

    for (px, py) in indices:
        for direction in directions:
            # centred, first order derivate of image
            p = i[direction][px, py]

            ltpp = 0
            ltpn = 0
            for dc, (x, y) in enumerate(neighbours(px, py)):
                ltp = g(p, i[direction][x, y])
                if ltp == 1: ltpp += 1 << dc
                elif ltp == -1: ltpn += 1 << dc
            else:
                histograms['ltpp'][direction][ltpp] += 1
                histograms['ltpn'][direction][ltpn] += 1

    return np.concatenate([histograms[sign][direction]
                           for direction in directions
                           for sign in ('ltpp', 'ltpn')])


def learn(a, b):
    svc = svm.SVC(kernel='rbf', C=1., gamma=1/32)
    features_a = extract_dataset(a)
    features_b = extract_dataset(b)
    xs = np.concatenate((features_a, features_b))
    ys = np.concatenate((np.repeat(a, len(features_a)),
                         np.repeat(b, len(features_b))))
    # shuffle dataset
    indexes = np.random.permutation(len(xs))
    xs = xs[indexes]
    ys = ys[indexes]

    folds = cross_validation.KFold(len(xs), n_folds=5)
    for training, testing in folds:
        x_training = xs[training]
        y_training = ys[training]
        x_testing = xs[testing]
        y_testing = ys[testing]
        svc.fit(x_training, y_training)

        predicted = svc.predict(x_testing)
        a = accuracy(predicted, y_testing)

        print('achieving {}'.format(a))

    return svc


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 3 and  sys.argv[1] == 'extract':
        extract_dataset(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == 'learn':
        learn()
    else:
        sys.exit(1)
