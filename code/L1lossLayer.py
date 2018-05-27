#encoding=utf-8
# imports
import sys
import loadpath as lp
sys.path.append(lp.caffe_root)
import caffe

import numpy as np
from collections import Counter


class L1lossLayer(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):
        
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.

        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        self.iter = 1.
        self.maxiter = 600000.
        self.loss = 0.

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        top[0].reshape(1)
        #top[2].reshape(1)
        self.diff = np.zeros_like( bottom[0].data,dtype = np.int8)

    def forward(self, bottom, top):
        """
        Load data.
        """
        predict = bottom[0].data
        label =   bottom[1].data
        self.count = np.sum(label > 0 )
        if self.count :
            loss = np.abs(predict - label)
            loss = 1.*np.sum( loss[label>0] ) / self.count
            top[0].data[...]  = loss
            self.loss = loss
        else:
            top[0].data[ ... ] = self.loss
        self.iter += 1.
    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        for ib in range(2):
            if not propagate_down[ib]:
                continue
            ndim = bottom[0].data.shape
            count = self.count
            if not self.count:
                bottom[ib].diff[ ... ] = np.zeros_like( bottom[0].data )
                continue

            bottom[ib].diff[ ... ] = np.ones_like( bottom[ib].data )
            inop = bottom[0].data < bottom[1].data
            bottom[ib].diff[ inop ] *= -1

            # ingore false label and repair
            ignore = bottom[1].data <= 0.
            bottom[ib].diff[ignore] = 0.

            #smoole
            score = np.abs(bottom[0].data - bottom[1].data)
            ignore = score < 1.0
            bottom[ib].diff[ ignore ] *= score [ ignore ] 
            # normlize
            bottom[ib].diff[...]  /= count
            


