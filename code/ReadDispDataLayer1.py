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
from PIL import Image, ImageEnhance
import random

from caffe.io import Transformer
from collections import Counter
import math
import matplotlib.pyplot as plt

    
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
        #transformer.set_mean('data', np.array(params['mean']))
        #transformer.set_raw_scale('data',255)
        transformer.set_channel_swap('data',(2,1,0))


        self.batch_loader = BatchLoader(params,indexlist,transformer,None)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size*2, 3, params['im_shape'][0], params['im_shape'][1])
        top[1].reshape(
            self.batch_size, 6, params['im_shape'][0], params['im_shape'][1])
        self.multi_scale = params['multi_scale']
        for ic,scale in enumerate( self.multi_scale ):
            sl = 2**scale
            top[ic+2].reshape(
                self.batch_size, 1, params['im_shape'][0]/sl, params['im_shape'][1]/sl)
        '''
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
            im_L,im_R,disp = self.batch_loader.load_next_image()
            # Add directly to the caffe data layer
            top[0].data[itt*2,...] = im_L
            top[0].data[itt*2+1, ...] = im_R 
            top[1].data[itt,:3,...] = im_L 
            top[1].data[itt,3:, ...] = im_R
            for ic,scale in enumerate( self.multi_scale ):
                sl = 2**scale
                top[ic+2].data[itt, ...] = disp[np.newaxis,::sl,::sl]
            '''
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
        left_name,right_name,seg_name,disp_name=index.split()
        # load a image and its label
        img_L = Image.open(osp.join(lp.image_root,left_name).replace('gtFine','leftImg8bit'))
        img_R = Image.open(osp.join(lp.image_root,right_name).replace('gtFine','rightImg8bit'))
        disp = Image.open(osp.join(lp.image_root,disp_name))
        #seg = Image.open(osp.join(lp.image_root,seg_name))
        size = self.im_shape[::-1]
        img_scale1 = img_L.resize(size, Image.BILINEAR)
        img_scale2 = img_R.resize(size, Image.BILINEAR)
        disp_scale = disp.resize( size, Image.BILINEAR)
        #seg_scale = seg.resize(self.im_shape[::-1], Image.NEAREST)
        img_scale1 = np.asarray( img_scale1 ) /255
        img_scale2 = np.asarray( img_scale2 ) /255
        disp_scale = (np.asarray(disp_scale,dtype=np.float32) -1)/256
        #seg_scale = np.asarray(seg_scale).astype(np.int8)
        
        '''
        if random.random() < 0.0:
            if random.random() < 1.1:
                img,label= self.scale_aug(img,label)
            img = self.color_aug(img)
            if img.shape[0] > self.im_shape[0]:
                y1,x1 = crop_image(img.shape[:2],self.im_shape,tag = self.tag)
                img = img[ y1:y1+self.im_shape[0],x1:x1+self.im_shape[1],:]
                label = label[ y1:y1+self.im_shape[0],x1:x1+self.im_shape[1]]
        else:   
            #im_L = np.asarray(Image.open(lp.image_root+left_name))
            im_L = np.asarray(img)
            if self.random:
                y1,x1 = crop_image(im_L.shape[:2],self.im_shape,tag = self.tag)
            else:
                x1,y1 = int(coord.split(',')[0]),int(coord.split(',')[1])
            img = im_L[ y1:y1+self.im_shape[0],x1:x1+self.im_shape[1],:]
            label = np.asarray(label).astype(np.int8)
            label = label[ y1:y1+self.im_shape[0],x1:x1+self.im_shape[1]]
        '''
        
        #temp = static_pixes(label,alpha = self.alpha,channel=self.label_num)
        self._cur += 1
        return self.transformer.preprocess('data',img_scale1),self.transformer.preprocess('data',img_scale2),disp_scale

    def color_aug(self,img):
        # Data augmentation of the loaded image
        # Color distort
        cut = 1
        prob_distort = cut
        if random.random() <= prob_distort:
            img = distort(img)
        img = np.asarray(img).astype(np.uint8)
        return img
        # Add gaussian noise
        #img = gaussian_noise(img)
    def scale_aug(self,img,label):
        # Rotate and crop to expected shape(fixed 640*1280)
        idx = random.choice([1,2,3])
        if idx == 1 :
            img,label=rotate(img, label, self.im_shape[0], self.im_shape[1])
        elif idx == 2 :
            img,label=rescale(img, label, self.im_shape[0], self.im_shape[1], True)
        else :
            img,label = mirror(img,label)
        label = np.asarray(label).astype(np.int8)
        return img,label
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

def distort(img):
    # color saturation change
    color_factor = np.random.uniform(0.8,1.2)
    img = ImageEnhance.Color(img).enhance(color_factor)

    # brightness change
    brightness_factor = np.random.uniform(0.8,1.2)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # contrast change
    contrast_factor = np.random.uniform(0.8,1.2)    # 阈值需要重新确定？
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img

def gaussian_noise(img):
    sigma = random.randint(0, 4) / 100
    var = sigma**2
    img_arr = np.array(img)
    img_arr_noise = skimage.util.random_noise(img_arr, mode='gaussian', var=var)
    img_arr_noise = img_arr_noise * 255
    img_arr_noise = np.array(img_arr_noise, np.uint8)
    img = Image.fromarray(img_arr_noise)
    return img

def rotate(img, label, new_height, new_width):
    height, width = img.size[1], img.size[0]
    sh = math.floor((height - new_height) / 2)
    sw = math.floor((width - new_width) / 2)
    rotate_angle = random.uniform(-18.9, 18.9)
    img = img.rotate(rotate_angle)
    label = label.rotate(rotate_angle)
    img = img.crop((sw, sh, sw+new_width, sh+new_height))
    label = label.crop((sw, sh, sw+new_width, sh+new_height))
    return img,label

def rescale(img, label, new_height, new_width, randoms=False):
    scale = 1.0*random.randint(9, 21) / 10
    height, width = img.size[1], img.size[0]
    scale_height = int(math.floor(height * scale))
    scale_width = int(math.floor(width * scale))
    img_scale = img.resize((scale_width, scale_height), Image.BILINEAR)
    label_scale = label.resize((scale_width, scale_height), Image.NEAREST)
    crop_height = (new_height if new_height < scale_height else scale_height)
    crop_width = (new_width if new_width < scale_width else scale_width)
    if randoms == False:
        sh = math.floor((scale_height - crop_height) / 2)
        sw = math.floor((scale_width - crop_width) / 2)
    else:
        sh = random.randint(0, scale_height-crop_height)
        sw = random.randint(0, scale_width - crop_width)
    img = img_scale.crop((sw, sh, sw+new_width, sh+new_height))
    label = label_scale.crop((sw, sh, sw+new_width, sh+new_height))
    return img,label

def gamma(img, gamma_val):
    img_arr = np.array(img)
    img_gamma_arr = exposure.adjust_gamma(img_arr, gamma_val)
    img_gamma = Image.fromarray(img_gamma_arr)
    return img_gamma


def gaussian_blur(img, radius, mean):
    img_blur_arr = np.array(img)
    img_blur_arr = cv2.GaussianBlur(img_blur_arr, (radius, radius), mean)
    img_blur = Image.fromarray(img_blur_arr)
    return img_blur

def mirror(img, label):
    img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT)
    label_mirror = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img_mirror, label_mirror
