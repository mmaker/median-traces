#!/usr/bin/env python
from __future__ import division, print_function

from glob import glob
from itertools import izip
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sklearn import svm, cross_validation, decomposition
from sklearn.metrics import accuracy_score as accuracy

__author__ = "Michele Orru`"
__email__ = "michele.orru@studenti.unitn.it"
__license__ = """"THE BEER-WARE LICENSE" (Revision 42):
 maker wrote this file. As long as you retain this notice you
 can do whatever you want with this stuff. If we meet some day, and you think
 this stuff is worth it, you can buy me a beer in return.
"""

__all__ = ['g', 'neighbours', 'extract_dataset', 'extract_feature', 'learn']

def g(a, b):
    p = np.sign(a*b)
    if p == -1: return 1
    if p == 1: return  0
    if p == 0: return -1

def neighbours(x, y):
    return [
        (x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1),
        (x+1, y+1), (x+1, y), (x+1, y-1), (x, y-1)
    ]

def extract_dataset(dataset_type):
    base_path = join('dataset', dataset_type, '')

    try:
        return np.load(base_path + 'features.npy')
    except:
        features = []
        for image_file in sorted(glob(base_path + '*.tif')):
            print('processing: {}'.format(image_file))
            features.append(extract_feature(image_file))

        features = np.asarray(features)
        np.save(base_path + 'features', features)
        return features


directions = ('h', 'dr', 'v', 'dl')
make_histogram = lambda: np.zeros(256, dtype='int')

def extract_feature(image_file):
    # monochrome image
    img = np.asarray(Image.open(image_file), dtype='int')
    # img = np.asarray(Image.open(image_file).convert('L'), dtype='int')

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
        },
    )

    for (px, py) in indices:
        for direction in directions:
            # centred, first order derivate of image
            p = i[direction][px, py]

            ltpp = ['0', ] * 8
            ltpn = ['0', ] * 8
            for dc, (x, y) in enumerate(neighbours(px, py)):
                ltp = g(i[direction][x, y], p)
                if ltp == 1: ltpp[dc] = '1'
                if ltp == -1: ltpn[dc] = '1'

            ltpp = int(''.join(ltpp), 2)
            ltpn = int(''.join(ltpn), 2)
            histograms['ltpp'][direction][ltpp] += 1
            histograms['ltpn'][direction][ltpn] += 1

    return np.concatenate([histograms[sign][direction]
                           for direction in directions
                           for sign in ('ltpp', 'ltpn')])


def plot(a, b):
    features_a = extract_dataset(a)[:50]
    features_b = extract_dataset(b)[:50]

    xs = np.arange(0, features_a.shape[1])
    for fa, fb in izip(features_a, features_b):
        plt.scatter(xs, fa,  c='red', alpha=.5)
        plt.scatter(xs, fb, c='blue', alpha=.5)

def learn(a, b):
    ori = np.load('dataset/ori/features.npy')
    mf5 = np.load('dataset/mf5/features.npy')

    ori = np.array([np.reshape(x, (256, -1))for x in ori])
    mf5 = np.array([np.reshape(x, (256, -1)) for x in mf5])

    features_x = np.concatenate((ori, mf5))
    features_y = np.concatenate((
        np.repeat('ori', len(ori)),
        np.repeat('mf5', len(mf5))))
    i = np.random.permutation(len(ori) + len(mf5))
    features_x = features_x[i]
    features_y = features_y[i]

    clf = svm.SVC()

    xs = features_x[:800]
    ys = features_y[:800]

    xs = xs.reshape(800, -1)
    clf.fit(xs, ys)

    xs = features_x[400:]
    samples = len(xs)
    ys = features_y[400:]
    xs = xs.reshape(samples, -1)
    clf.fit(xs, ys)
    pred = clf.predict(xs)
    print(accuracy(pred, ys))

    return clf

def test(a, b, image_file):
    clf = learn(a, b)
    f = extract_feature(image_file)

    print(clf.predict(f))


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 3 and  sys.argv[1] == 'extract':
        extract_dataset(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == 'learn':
        learn()
    elif len(sys.argv) == 4 and sys.argv[1] == 'plot':
        plot(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == 'test':
        test(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        sys.exit(1)
