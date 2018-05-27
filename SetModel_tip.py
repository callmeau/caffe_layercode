#encoding=utf-8
import sys
sys.path.append('../python')
sys.path.append('code')
#import optIO
import caffe
#print caffe.__file__
import numpy as np
import time
import os
import scipy.io as scio
import loadpath as lp
import scipy.misc as smi

__doc__=\
        '''
        call __init__ to see how to init
        call set_solver to set the solver,if the solver_file is not exist
        call to_train() to show result
        if the model has already trained,call to_show to show the result
        '''
class SetModel():
    def __init__(self,train='train.prototxt',solver='solver.prototxt',\
            val=None,work_root=None):
        '''
            (self,train,solver,val=None,work_root=None)
            given work_root,begin by / if config not in the current root 
            train_savefile is train.prototxt
            if val is same to train.prototxt,don't need to input
            solver_file is solver.prototxt
        '''
        self.path=work_root+'/' if work_root else os.getcwd()+'/'
        self.train=self.path+train
        self.val=self.path+val if val else None
        self.solver=self.path+solver
        
    def to_train(self,rank=1,loss_layer_name='loss',acc_layer_name=\
            'accuracy',niter=100,display=10,test_iter=10,test_interval=10,model_save='mymodel',\
            data_layer_name='data',fc_layer_name='fc',image_label='label',\
            show_tag=False):
        caffe.set_mode_gpu()
        caffe.set_device(rank)
        solver=caffe.SGDSolver(lp.solver_tip)
        solver.net.forward()
        model_coarse = 'nips_fpn50/Seg_fpn_640_with_coarse_aug_test_iter_600000.caffemodel'
        model_init = 'nips_joint50_city/Joint_city50_4s_continue_iter_50000.caffemodel'
        model_int = 'nips_poster/Use_for_draw_city_demo_warp_iter_210000.caffemodel'
        #solver.net.copy_from(lp.joint_city_model)
        #solver.net.copy_from(lp.fpn101_model)
        #solver.net.copy_from(lp.joint_city_model)
        solver.net.copy_from(model_coarse)
        solver.net.copy_from(model_init)
        solver.solve()
        return
        solver.step(1)
        img1 = solver.net.blobs['img0s'].data[0]
        img2 = solver.net.blobs['img1s'].data[0]
        disp = solver.net.blobs['disp_1s'].data[0]
        label = solver.net.blobs['input_seg'].data[0]
        smi.imsave('forTest/im11.png', img1)
        smi.imsave('forTest/im21.png', img2)
        smi.imsave('forTest/seg1.png', label)
        smi.imsave('forTest/disp1.png', disp)
        np.savetxt('forTest/seg1.txt', label, fmt='%.5f')
        np.savetxt('forTest/disp1.txt', disp, fmt='%.5f')

        solver.step(1)
        img1 = solver.net.blobs['img0s'].data[0]
        img2 = solver.net.blobs['img1s'].data[0]
        disp = solver.net.blobs['disp_1s'].data[0]
        label = solver.net.blobs['input_seg'].data[0]
        smi.imsave('forTest/im12.png', img1)
        smi.imsave('forTest/im22.png', img2)
        smi.imsave('forTest/seg2.png', label)
        smi.imsave('forTest/disp2.png', disp)
        np.savetxt('forTest/seg2.txt', label, fmt='%.5f')
        np.savetxt('forTest/disp2.txt', disp, fmt='%.5f')

        #solver.step(2000)
        return
        dataL=solver.net.blobs['img0_aug'].data
        print 'dataL shape',dataL.shape
        dataL=dataL.reshape(dataL.shape[1:])
        dataL=dataL.transpose(1,2,0)
        dataL=dataL[:,:,::-1]

        dataR=solver.net.blobs['img1_aug'].data
        dataR=dataR.reshape(dataR.shape[1:])
        dataR=dataR.transpose(1,2,0)
        dataR=dataR[:,:,::-1]
        print 'dataR shape',dataR.shape
        plt.figure()
        plt.subplot(131)
        plt.imshow(dataL)
        plt.title('dataL')

        plt.subplot(132)
        plt.imshow(dataR)
        plt.title('dataR')

        disp =solver.net.blobs['disp_gt_aug'].data
        disp=disp.reshape(disp.shape[2:])
        print 'disp', disp.shape
        plt.subplot(133)
        plt.imshow(disp,cmap='gray')
        plt.title('disp')


        plt.show()
        

if __name__=='__main__':
    #give the root begin by /,and save file of the model ,all of config will
    #be save by work_root
    m=SetModel(work_root='models/standard_train',train='256_inception2_train_val.prototxt',\
            solver='256_inception2_solver.prototxt')
    #m.set_solver(test_iter=1,max_iter=200,test_interval=np.inf,base_lr=1e-6,wei=1e-4,display=1)
    m.to_train(rank=int(sys.argv[1]),niter=10,display=1,test_interval=1,test_iter=30)
    #m.to_show(model_save='mymodel149760726717.caffemodeal')
