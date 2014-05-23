#!/usr/bin/env python
"""Median Traces; an implementation of
Revealing the Traces of Median Filtering Using High-Order Local Ternary Patterns.

Usage:
  median_traces.py extract <dataset> <dataset>
  median_traces.py learn <dataset> <dataset>
  median_traces.py -h | --help

Options:
  -h --help     Show this screen.

"""
from __future__ import division, print_function

from glob import glob
from itertools import izip
from multiprocessing import Pool
from os.path import join

from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sklearn import svm, cross_validation, decomposition
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score as accuracy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

__version__ = float('nan')
__author__ = "Michele Orru`"
__email__ = "michele.orru@studenti.unitn.it"
__license__ = """"THE BEER-WARE LICENSE" (Revision 42):
 maker wrote this file. As long as you retain this notice you
 can do whatever you want with this stuff. If we meet some day, and you think
 this stuff is worth it, you can buy me a beer in return.
"""

__all__ = ['g', 'neighbours', 'extract_dataset', 'extract_feature', 'learn']

def g(a, b):
    # hey, this shit makes g() 10s/50imgs faster
    if a == 0 or b == 0: return -1

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
        features = np.load(base_path + 'features.npy')
    except:
        files = glob(base_path + '*.jpeg')
#        features = [extract_feature(file) for file in files]
        features = Pool().map(extract_feature, files)
        features = np.array(features, dtype='float64')
        # caching
        np.save(base_path + 'features', features)
    finally:
        return features


directions = ('h', 'dr', 'v', 'dl')
make_histogram = lambda: np.zeros(256, dtype='int')

def extract_feature(image_file):
    print('processing {}'.format(image_file))
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
        plt.scatter(xs, fa, c='red', alpha=.5)
        plt.scatter(xs, fb, c='blue', alpha=.5)

def learn(a, b):
    features_a = np.array(extract_dataset(a), dtype='float64')
    features_b = np.array(extract_dataset(b), dtype='float64')
    features = np.concatenate((features_a, features_b))

    pca = decomposition.KernelPCA(kernel='linear')
    clf = svm.SVC(kernel='rbf')

    xs = pca.fit_transform(normalize(features))
    ys = np.concatenate((np.repeat(a, len(features_a)),
                         np.repeat(b, len(features_b))))
    parameters = {
#        'kernel': ['linear', 'rbf'],
        'C': 2**np.arange(0, 10, 0.5),
        'gamma': 2**np.arange(-5, 3, 0.5),
    }
    grid = GridSearchCV(clf, parameters, n_jobs=3, cv=5)
    clf = grid.fit(xs, ys)
    print('{} vs {}: {:3.3f}'.format(a, b, clf.best_score_))
    return Pipeline([('pca', pca), ('clf', clf)])


if __name__ == '__main__':
    arguments = docopt(__doc__, version=__version__)
    if arguments['learn']:
        learn(*arguments['<dataset>'])
    elif arguments['extract']:
        extract(*arguments['<dataset>'])
