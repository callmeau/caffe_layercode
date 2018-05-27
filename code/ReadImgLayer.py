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


class PascalMultilabelDataLayerSync(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):


        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the parameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']
        

        #要修正文件存放位置,存放格式:/home/zhanwj/data/1.jpg 1000100100
        list_file = params['split'] + '.txt'
        indexlist = [line.rstrip('\n') for line in open(osp.join(lp.label_root,list_file))]
        #进行扰动,不然多GPU训练时每个GPU读取的图片数据是一样的
        shuffle(indexlist)

        # Create a batch loader to load the images.indexlist 直接传给BatchLoader
        transformer = Transformer({'data':(self.batch_size,3,params['im_shape'][0], params['im_shape'][1])})
        transformer.set_transpose('data',(2,0,1))
        transformer.set_mean('data', np.array(params['mean']))
        transformer.set_raw_scale('data',255)
        transformer.set_channel_swap('data',(2,1,0))


        self.batch_loader = BatchLoader(params,indexlist,transformer,None)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        top[1].reshape(
            self.batch_size, params['im_shape'][0], params['im_shape'][1])
        '''
        self.multi_scale = params['multi_scale']
        for ic,scale in enumerate( self.multi_scale ):
            sl = 2**scale
            top[ic+1].reshape(
                self.batch_size, 1, params['im_shape'][0]/sl, params['im_shape'][1]/sl)
        self.to_temp = len(top) > len(self.multi_scale)+1
        if self.to_temp:
            top[ic+2].reshape(
                self.batch_size, params['label_num'])
        '''

        print_info("PascalMultilabelDataLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im,label = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label
            '''
            for ic,scale in enumerate( self.multi_scale ):
                sl = 2**scale
                top[ic+1].data[itt, ...] = label[np.newaxis::sl,::sl]
            if self.to_temp:
                top[ic+2].data[itt, ...] = temp
            '''

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass
def crop_image(ndim,new_dim,tag = 'Train'):
    if tag == 'Test':
        sh = int(( ndim[0] -new_dim[0] ) /2)
        sw = int(( ndim[1] - new_dim[1] ) /2)
        return sh,sw
    sh = int( np.random.rand()*( ndim[0] -new_dim[0] ) )
    sw = int( np.random.rand()*( ndim[1] -new_dim[1] ) )
    return sh,sw
def static_pixes(label,alpha = 0.0,channel=34,tag='Test'):
    c = Counter( label.flatten() )
    temp = np.zeros(channel)
    for k,v in c.iteritems():
        temp[k] = 1.0*v
    temp /= temp.max()
    temp = np.exp(-1.0 * alpha * temp)
    temp /= np.sum( temp[ temp != 1]) if np.sum( temp[ temp != 1]) else 1
    return temp


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, indexlist,transformer,result):
        self.result = result
        self.batch_size = params['batch_size']
        #may be need to repaire
        self.pascal_root = lp.label_root
        #统一压缩为360,480
        self.im_shape = params['im_shape'] 
        #要修正文件存放位置,存放格式:/home/zhanwj/data/1.jpg 1000100100
        self.indexlist = indexlist
        self.total=len(self.indexlist)
        self._cur = 0  # current image
        # build pregross 
        self.transformer = transformer
        self.alpha = params['alpha']
        self.tag = params['tag']
        self.label_num = params['label_num']
        if 'random' in params:
            self.random = True
        else:
            self.random = False

        # this class does some simple data-manipulations
        print "BatchLoader initialized with {} images".format(
            len(self.indexlist))

    def load_next_image(self):
        # Did we finish an epoch?
        if  self._cur == self.total :
            self._cur = 0
            shuffle(self.indexlist)
        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        #split image_name and image_lablel
        left_name,_,label_name,coord,_=index.split()
        #image_file_name 
        #im_L = np.asarray(Image.open(lp.image_root+left_name))
        im_L = np.asarray(Image.open(left_name))
        if self.random:
            y1,x1 = crop_image(im_L.shape[:2],self.im_shape,tag = self.tag)
        else:
            x1,y1 = int(coord.split(',')[0]),int(coord.split(',')[1])
        im = im_L[ y1:y1+self.im_shape[0],x1:x1+self.im_shape[1],:]
        #im_L = scipy.misc.imresize(im, self.im_shape)  # resize

        #load image to label
        #label = np.asarray(Image.open(lp.image_root+label_name)).astype(np.int8)
        self._cur += 1
        label = np.asarray(Image.open(label_name)).astype(np.int8)
        label = label[ y1:y1+self.im_shape[0],x1:x1+self.im_shape[1]]
        #temp = static_pixes(label,alpha = self.alpha,channel=self.label_num)
        return self.transformer.preprocess('data',im),label
            





def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size',  'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Output some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape'])
