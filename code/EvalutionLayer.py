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
        fp.write('%.3f\n'%item[0])


class EvaLayer(caffe.Layer):

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
        self.dloss = []
        self.sloss = []
        self.acc = []
        if osp.isfile(lp.result_file+'dloss.txt'):
            raise Exception("=====================result_file exit.=============")
        self.ids = 0

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
        self.dloss.append(bottom[0].data)
        self.sloss.append(bottom[1].data)
        self.acc.append(bottom[2].data)
        ss = (self.dloss[-1]+self.sloss[-1])/2
        top[0].data[...] = ss
        top[1].data[...] = self.acc[-1]
        if len(self.dloss) == 10:
            save_loss(lp.result_file+'dloss.txt',self.dloss)
            save_loss(lp.result_file+'sloss.txt',self.sloss)
            save_loss(lp.result_file+'acc.txt',self.acc)
            self.dloss=[]
            self.sloss=[]
            self.acc=[]

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass
