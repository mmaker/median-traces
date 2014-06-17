#!/usr/bin/env python
# coding: UTF-8
"""Median Traces; an implementation of \
«Revealing the Traces of Median Filtering Using High-Order Local Ternary Patterns».

Usage:
  median_traces.py extract [options] <dataset>
  median_traces.py learn   [options] <dataset> <dataset>
  median_traces.py plot    [options] <dataset> <dataset> [--samples INT]
  median_traces.py test    [options] <dataset> <dataset> [--cls CLASS] <targets>...
  median_traces.py measure [-r REGEX] <unit> <targets> <targets>
  median_traces.py -h | --help

Options:
  -h --help           Show this screen.
  -v --version        Print version number.
  -p, --path=DIR      Specifies dataset path [default: dataset/].
  -r, --regex=REGEX   Specifies regex for locating images [default: *.jpg].
  --samples=INT       Specifies the number of samples when plotting [default: 30].
  --cls CLASS         Specifies that all test inputs belong to the same class, and output accuracy.
"""
from __future__ import division, print_function

import os.path
from functools import reduce
from glob import glob
from itertools import izip
from math import log10
from multiprocessing import Pool
from operator import add
import cPickle as pickle

from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sklearn import svm, cross_validation, decomposition
#from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

from skimage.measure import structural_similarity as ssim


__version__ = float('nan')
__author__ = "Michele Orru`"
__email__ = "michele.orru@studenti.unitn.it"
__license__ = """"THE BEER-WARE LICENSE" (Revision 42):
 maker wrote this file. As long as you retain this notice you
 can do whatever you want with this stuff. If we meet some day, and you think
 this stuff is worth it, you can buy me a beer in return.
"""

__all__ = ['g', 'neighbours', 'psnr',
           'extract_dataset', 'extract_feature', 'extract_feature_from_file',
           'learn', 'test', 'measure']

DIRECTIONS = ('h', 'dr', 'v', 'dl')
CLF_FILE_TEMPLATE ='{}_vs_{}.clf'.format

dataset_path = 'dataset'
images_regex = '*.jpg'
make_histogram = lambda: np.zeros(256, dtype='int')


def _load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def _dump(obj, file):
    with open(file, 'wb') as f:
        return pickle.dump(obj, f)

def psnr(img, nimg, bits=8):
    """
    Calculate the Peak Signal Noise Ratio on the given image channel,
    interpreted as `bits`-bit image.
    """
    max_value =  1 << bits - 1

    if not img.shape == nimg.shape:
        raise ValueError("Image shapes differ")
    if (img == nimg).all():
        return float('inf')

    mse = np.sum((img - nimg)**2) / np.prod(img.shape)
    return 10 * (2*log10(max_value) - log10(mse))

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
    base_path = os.path.join(dataset_path, dataset_type, '')

    try:
        features = np.load(base_path + 'features.npy')
    except:
        files = glob(base_path + images_regex)
        # features = [extract_feature(file) for file in files]
        features = Pool().map(extract_feature_from_file, files)
        features = np.array(features, dtype='float64')
        # caching
        np.save(base_path + 'features', features)
    finally:
        return features

def extract_feature_from_file(image_file):
    """
    Given an image path `image_file`, extract the monochrome grayscale 8-bit
    image and produce the second-order LTP histogram from it.

    :rtype: np.array
    """
    # print('processing {}'.format(image_file))
    img = np.asarray(Image.open(image_file), dtype='int')
    return extract_feature(img)

def extract_feature(img):
    """
    Extract the second-order, LTP features from the image matrix `img`.

    :param img: the image matrix. `dtype=int` is assumed.
    :return: the concatenated histograms, one for each direction and sign.
    :rtype: np.array
    """
    # compute second_order ternary patterns
    cimg = img[1:-1, 1:-1]
    # first-order derivate
    i = {'h': cimg - img[1:-1, 2:],
         'v': cimg - img[:-2, 1:-1],
         'dr': cimg - img[:-2, 2:],
         'dl': cimg - img[:-2, :-2],
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
        for direction in DIRECTIONS:
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
                           for direction in DIRECTIONS
                           for sign in ('ltpp', 'ltpn')])


def plot(a, b, samples=30):
    """
    Merge with different colors, the resulting histograms of two datasets.
    """
    features_a = extract_dataset(a)[:samples]
    features_b = extract_dataset(b)[:samples]

    xs = np.arange(0, features_a.shape[1])
    for fa, fb in izip(features_a, features_b):
        plt.scatter(xs, fa, c='red', alpha=.5)
        plt.scatter(xs, fb, c='blue', alpha=.5)
    plt.show()


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
    grid = GridSearchCV(clf, parameters, n_jobs=4, cv=5)
    clf = grid.fit(xs, ys)
    print('{} vs {}: {:3.3f}'.format(a, b, clf.best_score_))
    pipeline = Pipeline([('pca', pca), ('clf', clf)])

    # caching
    _dump(pipeline, os.path.join(dataset_path, CLF_FILE_TEMPLATE(a, b)))

    return pipeline

def test(a, b, targets, targets_class=None):
    """
    Specifically test a number of images after having classified the two
    datasets `a` and `b`.

    :param str a: dataset
    :param str b:
    :param list targets: list of glob patterns, directories, single files.
    """
    # load classificator. If not found, create it.
    try:
        clf = _load(os.path.join(dataset_path, CLF_FILE_TEMPLATE(a, b)))
    except IOError, ValueError:
        clf = learn(a, b)

    # Create test cases
    targets = [target if not os.path.isdir(target) else
               os.path.join(target, images_regex)
               for target in targets
               # commodity, remove any 'feature.npy', if present
               if not target.endswith('features.npy')
    ]
    files = reduce(add, map(glob, targets))
    # features = [extract_feature(file) for file in files]
    tests = Pool().map(extract_feature_from_file, files)
    tests = normalize(np.array(tests, dtype='float64'))

    if targets_class is None:
        predicted = clf.predict(tests)
        for file, prediction in izip(files, predicted):
            print('{}\t{}'.format(file, prediction))

    else:
        score = clf.score(tests, np.repeat(targets_class, len(tests)))
        print('{}/{} had accuracy {:3.3f}'.format(a, b, score))

def measure(a, b, measuref=psnr):
    """
    Output the structural similarity in terms of (mean, var) for all images
    present in directories a, b.
    """
    a_files = sorted(glob(os.path.join(a, images_regex)
                          if os.path.isdir(a) else a))
    b_files = sorted(glob(os.path.join(b, images_regex)
                          if os.path.isdir(b) else b))

    open_image = lambda f: np.asarray(Image.open(f).convert('L'), dtype='float')
    simil = [measuref(open_image(file_a), open_image(file_b))
             for file_a, file_b in zip(a_files, b_files)]

    mean, var = np.mean(simil), np.var(simil)
    print('{:4.3f} {:5.4f}'.format(mean, var))

    return mean, var


if __name__ == '__main__':
    args = docopt(__doc__, version=__version__)

    dataset_path = args['--path']
    images_regex = args['--regex']
    datasets = args['<dataset>']

    if args['extract']:
        dataset, = datasets
        extract_dataset(dataset)

    elif args['learn']:
        a, b = datasets
        learn(a, b)

    elif args['plot']:
        a, b = datasets
        samples = int(args['--samples'])
        plot(a, b, samples=samples)

    elif args['test']:
        a, b = datasets
        targets = args['<targets>']
        targets_class = args['--cls']
        test(a, b, targets, targets_class)

    elif args['measure']:
        a, b = args['<targets>']
        measuref = dict(psnr=psnr, ssim=ssim)[args['<unit>']]
        measure(a, b, measuref=measuref)
