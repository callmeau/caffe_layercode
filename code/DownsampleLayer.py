#encoding=utf-8
# imports
import sys
import loadpath as lp
sys.path.append(lp.caffe_root)
import caffe

import numpy as np
import os.path as osp


class DownsampleLayer(caffe.Layer):
    """
    This is a layer to downsample the label through nearest interpolation.
    """

    def setup(self, bottom, top):
        '''
        Examine if number of bottom layers is valid
        '''
        if len(bottom) != 2:
            raise Exception('There must be two bottom layers.')

    def reshape(self, bottom, top):
        """
        Reshape the top layer referring to the shape of bottom[1]
        """
	top[0].reshape(*bottom[1].shape)

    def forward(self, bottom, top):
        """
        Resize bottom[0] to the shape of bottom[1] and assign the value to top[0]
        Shape of bottom[0] must be integral multiples of that of bottom[1]
        """
        step_h = bottom[0].shape[2] / bottom[1].shape[2]
        step_w = bottom[0].shape[3] / bottom[1].shape[3]
	print step_w
	print bottom[0].shape[0], bottom[0].shape[1], bottom[0].shape[2], bottom[0].shape[3],
        top[0].data[:, :,:,:] = bottom[0].data[:, :, ::step_h, ::step_w] 

    def backward(self, top, propagate_down, bottom):
        """
        Not need back propagatation
        """
        pass
