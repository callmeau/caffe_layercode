#encoding=utf-8
# imports
import sys
import loadpath as lp
sys.path.append(lp.caffe_root)
import caffe

import numpy as np
from collections import Counter


def static_pixes(label,ndim,alpha = .5,tag='Test'):
    temp = np.zeros((ndim[0],ndim[1]))
    for itt in xrange(ndim[0]):
        c = Counter( label[itt].flatten() )
        for k,v in c.iteritems():
            temp[itt,k] = 1.0*v
    temp /= np.max( temp,axis = 1,keepdims = True)
    temp = np.exp(-1.0 * alpha * temp)
    temp /= np.sum( temp[ temp != 1] ) if np.sum( temp[ temp != 1]) else 1
    return temp
def get_weight(label,ndim,filed):
    weight = np.ones(ndim)
    for ir in range( ndim[1] / filed):
        for ic in range(ndim[1] / filed):
            weight[:,:,ir,ic] = 1.0 + \
               static_pixes(label[:,ir*filed:(ir+1)*filed,ic*filed:(ic+1)*filed],ndim)
    return weight
    

class L2lossLayer(caffe.Layer):

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
        self.iter = 0
        self.maxiter = 600000
        self.loss = 0.

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        top[0].reshape(1)
        #top[2].reshape(1)

    def forward(self, bottom, top):
        """
        Load data.
        """
        ndim = bottom[0].data.shape[0]
        predict = bottom[0].data
        label =   bottom[1].data
        loss =  (predict - label) ** 2
        index = label < np.inf
        self.count = np.sum( index )
        if self.count:
            loss = 1.*np.sum( loss[index] ) / self.count
            top[0].data[...] = loss
            self.loss = loss
        else:
            top[0].data[...] = self.loss

            
    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        for ib in range(2):
            if not propagate_down[ib]:
                continue
            ndim = bottom[0].data.shape
            count = ndim[0] * ndim[2] * ndim[3]
            if not self.count:
                bottom[ib].diff[ ... ] = np.zeros_like( bottom[0].data )
                continue
            if top[0].data < 1.
                bottom[ib].diff[ ... ] = np.abs( bottom[0].data - bottom[1].data )
                bottom[ib].diff[ ... ] *= ( 1 - 1.0*self.iter/self.maxiter )
            else:
                bottom[ib].diff[ ... ] = np.ones_like( bottom[ib].data )
                inop = bottom[0].data < bottom[1].data
                bottom[ib].diff[ inop ] *= -1
            
            # ingore false label and repair
            ignore = bottom[1].data <= 0.
            count -= np.sum(ignore)
            bottom[ib].diff[ignore] = 0.
            #normlist
            bottom[ib].diff[...]  /= count
            
