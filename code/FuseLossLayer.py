#encoding=utf-8
# imports
import sys
import loadpath as lp
sys.path.append(lp.caffe_root)
import caffe

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import scipy.misc as scmi
import labels as query

dquery = { q.id:q.name for q in query.labels }
dneet = {
    "bicycle"    :  4672.3249222261 ,
    "caravan"    : 36771.8241758242 ,
    "motorcycle" :  6298.7200839748 ,
    "rider"      :  3930.4788056518 ,
    "bus"        : 35732.1511111111 ,
    "train"      : 67583.7075812274 ,
    "car"        : 12794.0202738185 ,
    "person"     :  3462.4756337644 ,
    "truck"      : 27855.1264367816 ,
    "trailer"    : 16926.9763313609 ,
}
def get_weight(label,ndim):
    temp = np.zeros_like(label)
    instSize = np.count_nonzero(label)
    c = Counter( label.flatten() )
    maxd = max(c.values())
    for k,v in c.items():
        tmp = label == k
        temp[tmp] +=  np.log(1.0*maxd / v ) + 1.
        if dquery[k] in dneet:
            temp[tmp] += dneet[ dquery[k] ] / float(instSize)
    temp +=1
    return temp 

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
        self.acc = 0.
        self.need = [34,27,25,28,31,32,33]

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
        label = bottom[1].data.reshape(ndim[0],ndim[2],ndim[3])
        weight = get_weight(label,label.shape)
        #temp = bottom[2].data
        ss = 0.0
        for itt in range(ndim[0]):
            for i in range( ndim[1] ):
                tmp = label[itt] == i
                ss += np.sum( -1.0* weight[itt,tmp]*np.log(self.diff[itt,i,tmp] + 1e-12) ) 
        top[0].data[...] = ss / np.sum(weight)
        self.weight = weight
        '''
        self.iters += 1
        if self.iters % 10 == 0:
            need = np.zeros_like(label)
            print '='*15 
            for iid in self.need:
                need += label == iid
            need = need > 0 
            if np.sum(need) > 0:
                acc = np.argmax(bottom[0].data,axis = 1)== label
                acc1 = 1.0*np.sum(acc[need])/(np.sum(need)+1)
                self.acc = acc1
            print 'iIOU accaury: ', self.acc
            if self.acc > .8:
                scmi.imsave('test_08.png',np.argmax(bottom[0].data,axis=1)[0])
            self.iters = 0
        '''
    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        for ib in range(2):
            if not propagate_down[ib]:
                continue
            ndim = bottom[0].data.shape
            label = bottom[1].data
            label = bottom[1].data.reshape(ndim[0],ndim[2],ndim[3])
            bottom[ib].diff[ ... ] = self.diff
            for itt in range(ndim[0]):
                for ic in range( ndim[1] ):
                    tmp = label[itt] == ic
                    bottom[ib].diff[itt,ic,tmp] -= 1
            bottom[ib].diff[...] *= self.weight.reshape(ndim[0],1,ndim[2],ndim[3])
            bottom[ib].diff[...] /= np.sum(self.weight)



