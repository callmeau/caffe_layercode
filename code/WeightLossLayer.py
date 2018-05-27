#encoding=utf-8
# imports
import sys
import loadpath as lp
sys.path.append(lp.caffe_root)
import scipy.misc
import skimage.io
import caffe

import numpy as np
import os.path as osp
from random import shuffle
from PIL import Image

from caffe.io import Transformer
from collections import Counter


class LossLayer(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):
        
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.

        if len(bottom) != 3:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        top[0].reshape(1)
        self.diff = np.zeros_like( bottom[0].data,dtype = np.int8)

    def forward(self, bottom, top):
        """
        Load data.
        """
        ndim = bottom[0].data.shape
        label = bottom[1].data.astype(np.int8)
        score = bottom[0].data
        temp = bottom[2].data
        ss = 0.0
        for itt in range(ndim[0]):
            for i in range( ndim[1] ):
                tmp = label[itt,0] == i
                ss += np.sum( -1.0* np.log(score[itt,i,tmp]+1e-12) * temp[itt,i] ) 
        top[0].data[...] = ss / (ndim[2]*ndim[3]) / ndim[0]

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        for ib in range(3):
            if not propagate_down[ib]:
                continue
            ndim = bottom[0].data.shape
            label = bottom[1].data.astype(np.int8)
            bottom[ib].diff[ ... ] = self.diff
            temp = bottom[2].data
            for itt in range(ndim[0]):
                for ic in range( ndim[1] ):
                    tmp = label[itt,0] == ic
                    bottom[ib].diff[itt,ic,tmp] -= 1
                    bottom[ib].diff[itt,ic,tmp] *= temp
