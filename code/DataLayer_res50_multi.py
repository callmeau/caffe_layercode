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
import random
from PIL import Image, ImageEnhance
import skimage

from caffe.io import Transformer
from collections import Counter
import math
import labels as query


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
        '''
        error_file = []
        for idx,it in enumerate(indexlist):
            name = it.split()[0]
            if  not osp.isfile(name):
                error_file.append(idx)
        for it in error_file:
            indexlist.pop(it)
        print '='*15
        print 'image error %d'%len(error_file)
        '''
        #进行扰动,不然多GPU训练时每个GPU读取的图片数据是一样的
        shuffle(indexlist)

        # Create a batch loader to load the images.indexlist 直接传给BatchLoader
        self.params = params
        self.tag = params['tag']
        self.size_list=[640,608,576,544,512,480,448,416,384,352,320,288,256,224,192]

        self.id2trainId = {label.id: label.trainId for label in query.labels}  # dictionary mapping from raw IDs to train IDs

        self.batch_loader = BatchLoader(params,indexlist,None)


        #print_info("PascalMultilabelDataLayerSync", params)
    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        while True:
            try:
                imgsList = []
                labelsList = []
                for itt in range(self.batch_size):
                    im, label = self.batch_loader.load_next_image()
                    imgsList.append(im)
                    labelsList.append(label)
                break
            except:
                self.batch_loader._cur += 1
        pro = .5
        ot = 2
        if random.random() < pro:
            if random.random() < pro:
                scale = 0.6 * random.random() + .7
                scale = int(scale * self.params['im_shape'][0])
                scale = scale // 32 * 32
                for itt in range(self.batch_size):
                    im =  imgsList[itt]
                    label = labelsList[itt]
                    im = im.resize((scale*2,scale))
                    label = label.resize( (scale*2,scale),Image.NEAREST)
                    imgsList[itt] = im;
                    labelsList[itt] = label;
            if  random.random() < pro:
                for itt in range(self.batch_size):
                    im = imgsList[itt]
                    label = labelsList[itt]
                    im,label = rotate(im,label, im.size[1] - 32, 2*(im.size[1]-32))
                    imgsList[itt] = im;
                    labelsList[itt] = label;
            for itt in range(self.batch_size):
				if random.random() < pro :
					im = imgsList[itt]
					label = labelsList[itt]
					im = im.transpose(Image.FLIP_LEFT_RIGHT)
					label = label.transpose(Image.FLIP_LEFT_RIGHT)
					imgsList[itt] = im;
					labelsList[itt] = label;                 
        for size in self.size_list:
            if size <= imgsList[0].size[1]:
                im_shape = [size,ot*size]
                break

        self.imgsList = []
        self.labelsList = []
        for itt in range(self.batch_size):
            im = np.asarray(imgsList[itt])
            label = np.asarray(labelsList[itt], dtype = np.int8)
            y1,x1 = crop_image(im.shape,im_shape,tag = self.tag)
            im = im[ y1:y1+im_shape[0],x1:x1+im_shape[1],:]
            label = label[ y1:y1+im_shape[0],x1:x1+im_shape[1]]
            label = assign_trainIds(self.id2trainId,label)
            self.imgsList.append(im)
            self.labelsList.append(label)
        
        top[0].reshape(
            self.batch_size, 3, im_shape[0], im_shape[1])
        self.multi_scale = self.params['multi_scale']
        for ic,scale in enumerate( self.multi_scale ):
            sl = 2**scale
            top[ic+1].reshape(
                self.batch_size, 1, im_shape[0]/sl, im_shape[1]/sl)

        transformer = Transformer({'data':(self.batch_size,3,im_shape[0], im_shape[1])})
        transformer.set_transpose('data',(2,0,1))
        transformer.set_mean('data', np.array(self.params['mean']))
        transformer.set_raw_scale('data',255)
        transformer.set_channel_swap('data',(2,1,0))
        self.transformer = transformer

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im = self.transformer.preprocess('data',self.imgsList[itt])
            label = self.labelsList[itt]

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            for ic,scale in enumerate( self.multi_scale ):
                sl = 2**scale
                top[ic+1].data[itt, ...] = label[np.newaxis,::sl,::sl]


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
    '''
	thresh = 0.2
	if random.random() <= thresh:
		sh = int(( ndim[0] -new_dim[0] ) /2)
        sw = int(( ndim[1] - new_dim[1] ) /2)
        return sh, sw
    '''
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

    def __init__(self, params, indexlist,result):
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
        index = self.indexlist[self._cur].replace('dispflownet-release/models','flownet2')  # Get the image index
        #split image_name and image_lablel
        items = index.split()
        flag =True
        progate = .3
        if index.find('-1,-1') != -1 or len(items) == 2:
            if len(items) == 2:
                left_name,label_name =items
            else:
                left_name,label_name,_=items
            im = Image.open(left_name)
            label = Image.open(label_name)
            y1,x1 = crop_image((im.size[1],im.size[0]),self.im_shape,tag = self.tag)
            im = im.crop((x1,y1,x1+self.im_shape[1],y1+self.im_shape[0]))
            label = label.crop((x1,y1,x1+self.im_shape[1],y1+self.im_shape[0]))

        else:
            left_name,label_name,corrd = items
            start,ends = corrd.split(',')
            s = int(start)
            e = int(ends)
            im = Image.open(left_name).crop((s,e,s+self.im_shape[1],e+self.im_shape[0]))
            label = Image.open(label_name).crop((s,e,s+self.im_shape[1],e+self.im_shape[0]))
        if random.random() < progate :
            # color saturation change
            color_factor = np.random.uniform(0.8,1.2)
            im = ImageEnhance.Color(im).enhance(color_factor)
            # brightness change
            brightness_factor = np.random.uniform(0.8,1.2)
            im = ImageEnhance.Brightness(im).enhance(brightness_factor)
            # contrast change
            contrast_factor = np.random.uniform(0.8,1.2)    # 阈值需要重新确定？
            im = ImageEnhance.Contrast(im).enhance(contrast_factor)

        
        self._cur = self._cur + 1 
        return im,label

def assign_trainIds(id2trainId, label):
    res = np.zeros_like( label )
    for k, v in id2trainId.iteritems():
        if v!= 255:
            res[label == k] = v
        else:
            res[label == k] = 19
    return res.astype( np.int8 )
def rotate(img, label, new_height, new_width):
    height, width = img.size[1], img.size[0]
    sh = math.floor((height - new_height) / 2)
    sw = math.floor((width - new_width) / 2)
    rotate_angle = random.uniform(-3, 3)
    #rotate_angle = random.uniform(-18.9, 18.9)
    img = img.rotate(rotate_angle)
    label = label.rotate(rotate_angle)
    img = img.crop((sw, sh, sw+new_width, sh+new_height))
    label = label.crop((sw, sh, sw+new_width, sh+new_height))
    return img, label




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
