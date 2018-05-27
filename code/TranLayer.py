#encoding=utf-8
# imports
import sys
import loadpath as lp
sys.path.append(lp.caffe_root)
import caffe

import numpy as np



class TransLayer(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):
        
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.

        print 'Transpose with %d bottom'% len(bottom)

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        ndim = battom[0].data.shape
        for ib in range( len(top) ):
            top[ib].reshape(ndim[0], ndim[1],ndim[3],ndim[2])

    def forward(self, bottom, top):
        """
        Load data.
        """
        for ib in range( len(top) ):
            top[ib].data = np.transpose( bottom[0].data,top[0].data.shape)

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        for ib in range( len(bottom) ):
            if not propagate_down[ib]:
                continue
            bottom[ib].diff[ ... ] = np.transpose( top[ib].data, bottom[ib].data.shape )
