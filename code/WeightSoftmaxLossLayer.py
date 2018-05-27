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
    

class LossLayer(caffe.Layer):

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
        self.iters = 0
        self.acc = 0.0

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
        diff = bottom[0].data
        diff = np.exp(diff - np.max(diff,axis=1,keepdims=True))
        self.diff =  diff / (np.sum(diff,axis = 1,keepdims=True))
        ndim = bottom[0].data.shape
        label = bottom[1].data.astype(np.int8)
        skip = label.shape[1]/ndim[2]
        label = label[:,::skip,::skip]
        self.count = ndim[2]*ndim[3]*ndim[0]
        #temp = bottom[2].data
        ss = 0.0
        for itt in range(ndim[0]):
            for i in range( ndim[1] ):
                tmp = label[itt] == i
                ss += np.sum( -1.0* np.log( self.diff[itt,i,tmp] + 1e-12) ) 
        top[0].data[...] = ss / self.count
        self.iters += 1
        self.acc += 1.0*np.sum(np.argmax(bottom[0].data,axis =1)==label)/self.count
        if self.iters % 20 == 0:
            print '='*15 
            print 'accaury: ', self.acc/20
            self.iters = 0
            self.acc = 0.0
    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        for ib in range(2):
            if not propagate_down[ib]:
                continue
            ndim = bottom[0].data.shape
            label = bottom[1].data.astype(np.int8)
            skip = label.shape[1]/ndim[2]
            temp = get_weight(label,ndim,skip)
            label = label[:,::skip,::skip]
            bottom[ib].diff[ ... ] = self.diff
            for itt in range(ndim[0]):
                for ic in range( ndim[1] ):
                    tmp = label[itt] == ic
                    bottom[ib].diff[itt,ic,tmp] -= 1
                    bottom[ib].diff[itt,ic,tmp] *= temp[itt,ic,tmp]
                    
            bottom[ib].diff[...] = bottom[ib].diff / ndim[0] / (self.count+1)



