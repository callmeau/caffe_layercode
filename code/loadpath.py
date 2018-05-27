image_root = '/home/zhanwj/Desktop/scenflow/flownet2/DispNet/citycapes/'
label_root1 = '/home/zhanwj/Desktop/scenflow/flownet2/DispNet/citycapes/code/dataAug/'
label_root_KITTI = '/home/zhanwj/Desktop/scenflow/flownet2/DispNet/KITTI_disp/info'
label_root = 'citycapes/joint_info/'
#label_root = 'KITTI_disp/info/'
#label_root = 'citycapes/label_info_fine/'
#label_root = 'citycapes/code/dataAug/'
#label_root = 'KITTI/info/'
caffe_root = '/home/zhanwj/Desktop/scenflow/flownet2/python/'

info_root = 'citycapes/joint_info/'
info_root = 'KITTI_disp/info/'
kitti_disp = 'KITTI_disp/'
kitti_seg = 'KITTI/'
result_file = 'log/kitti_'
result_file = 'log1/city_fine'
result_file = 'log1/city_fine_fine'

KITTI_disp_root = '/home/zhanwj/Desktop/scenflow/flownet2/DispNet/KITTI_disp'
aug_info_root = '/home/zhanwj/Desktop/scenflow/flownet2/DispNet/KITTI_disp/info'
aug_image_root = '/home/zhanwj/Desktop/scenflow/dispflownet-release/models/DispNet/KITTI_disp'
disp_info_root = '/home/zhanwj/Desktop/scenflow/dispflownet-release/models/DispNet/citycapes/joint_info/'

solver = 'weight/solver_seg.prototxt'
solver = 'weight/solver_flownets.prototxt'
solver = 'weight/solver_dispnet.prototxt'
solver_joint = 'weight/solver_dispnet1.prototxt'
solver_joint_KITTI = 'weight/solver_KITTI_fuse.prototxt'
solver_tip = 'weight/solver_tip.prototxt'
#solver = 'weight/solver_augseg.prototxt'
solver_fpn101 = 'weight/solver_fpn101.prototxt'
solver_fpn50 = 'weight/solver_fpn50.prototxt'
solver1_fpn50 = 'weight/solver1_fpn50.prototxt'
solver2_fpn50 = 'weight/solver2_fpn50.prototxt'
model = 'result_city_34/models/seg_state2_resnet50_4s_640_random_NODROP_iter_980000.caffemodel'
#model = 'result_city_34/segfpn_state2_resnet50_iter_40000.caffemodel'
model = 'weight/ResNet-50-model.caffemodel'
model = 'result_city_34/models/seg_state2_resnet50_4s_640_random_NODROP_iter_980000.caffemodel'
#model = 'result_city_34/segfpn_state2_resnet50_iter_90000.caffemodel'
model = 'result_itsc/model/dispnet_ALL1_iter_600010.caffemodel'
model ='result_itsc/dispSeg_fixseg_768_iter_40000.caffemodel'
fpn50_model = 'result_city_34/model/segfpn_state2_resnet50_iter_310000.caffemodel' ## final fpn
fpn50_model = 'nips_joint_city/Joint_city50_iter_40000.caffemodel'
fpn50_34_model = 'city_fpn50/bat/Seg_fpn_640_with_coarse_aug_nocost_iter_600000.caffemodel'
model1 = 'city_fpn50/init.caffemodel'
model1 = 'city_fpn50/Seg_fpn_640_with_coarse_aug_continue_iter_3000.caffemodel'
model = 'result_itsc/dispSeg_fixseg_384_iter_280000.caffemodel'
model = 'result_both/dispSeg_fixdisp_both_640_iter_160000.caffemodel'
joint_city_model = 'result_both/dispSeg_fixdisp_both_640_iter_120000.caffemodel' #final cityscape
fpn101_model = 'city_fpn101/Seg_fpn_640_with_coarse_aug_iter_200000.caffemodel'
fpn101_34_model = 'city_fpn101/bat/Seg_fpn_640_with_coarse_aug_iter_600000.caffemodel'
model = 'result_kitti/dispSeg_both_kitti_160_iter_100000.caffemodel' ##final kitti
model = 'city_fine/dispSeg_both_city_384_iter_40000.caffemodel'
#model = 'result_kitti/dispSeg_both_kitti_320_iter_20000.caffemodel'
#model = 'weight/SegCorrDisp.caffemodel'
#model = 'model/DispNet_CVPR2016.caffemodel'
model_wei = 'weight/params_wei.mat' #from dispnet
model_bias = 'weight/params_bias.mat'

kitti_image_root = '/home/zhanwj/Desktop/scenflow/flownet2/DispNet/KITTI_disp/'
kitti_semantics_root = '/home/zhanwj/Desktop/scenflow/flownet2/DispNet/KITTI/'
city_image_root = '/home/zhanwj/Desktop/scenflow/flownet2/DispNet/KITTI_disp/'
kitti_image_root1 = '/home/zhanwj/Desktop/scenflow/flownet2/DispNet/KITTI/'
fuse_info_root = '/home/zhanwj/Desktop/scenflow/flownet2/DispNet/City_KITTI_disp/info'
kitti_info_root = '/home/zhanwj/Desktop/scenflow/flownet2/DispNet/KITTI_disp/info'
op_root = '/home/zhanwj/Desktop/scenflow/flownet2/DispNet/cao/Segmentation_Rigid_Training/Training/info'
