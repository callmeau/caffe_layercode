#encoding=utf-8
# imports
import sys
import loadpath as lp
sys.path.append(lp.caffe_root)
import caffe
import os.path as osp

import numpy as np


def save_loss(filename,item):
    with open(filename,'a') as fp:
        fp.write('%.3f\n'%item)


class EvaLayer(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):
        
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.

        chanels = len(bottom)
        self.dloss = np.zeros(chanels)
        self.acc = np.zeros(chanels)
        self.chanels =chanels

        if osp.isfile(lp.result_file+'dloss.txt'):
            raise Exception("=====================result_file exit.=============")
        self._cur = 0

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        top[0].reshape(1)
        top[1].reshape(1)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for ic in range(self.chanels/2):
            self.dloss[ic] = bottom[ic].data
            self.acc[ic]  =   bottom[ic+self.chanels/2].data
        ss = np.sum(self.dloss)/self.chanels
        top[0].data[...] = ss
        top[1].data[...] = self.acc[0]
        self._cur += 1
        if self._cur == 10:
            for ic in range(self.chanels/2):
                save_loss(lp.result_file+'dloss%d.txt'%ic,self.dloss[ic])
                save_loss(lp.result_file+'acc%d.txt'%ic,self.acc[ic])
                self._cur = 0
            save_loss(lp.result_file+'joint.txt',ss)

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass
