#encoding=utf-8
# imports
import sys
import loadpath as lp
sys.path.append(lp.caffe_root)
import caffe

import numpy as np


class DummyData(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):
        
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.

        if len(bottom) != 1:
            raise Exception("Need an example inputs to reshape.")

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        ndim = bottom[0].data.shape
        top[0].reshape( ndim[0],ndim[1],ndim[2],ndim[3] )

    def forward(self, bottom, top):
        """
        Load data.
        """
        top[0].data[ ... ] = np.zeros_like( bottom[0].data )
    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass

