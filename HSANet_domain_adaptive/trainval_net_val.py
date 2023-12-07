# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pprint
import pdb
import time
import _init_paths
import random
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.nms.nms_wrapper import nms
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, FocalLoss, sampler, calc_supp, EFocalLoss
from Fourier_based import FDA_source_to_target_np
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.utils.parser_func import parse_args, set_dataset_args

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# def custom_blur_demo(image):
#     img = image.transpose((1, 2, 0))
#     print(img.size)
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
#     dst = cv.filter2D(img, -1, kernel=kernel)
#     return dst
try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

# Get Fourier
def Get_Fourier(img_s, img_t):
    img_s = np.squeeze(img_s.cpu().numpy())  # [3,h,w]
    # img_s_sharpen = custom_blur_demo(img_s)
    # img_s = img_s_sharpen.transpose((2, 0 ,1))
    img_t = np.squeeze(img_t.cpu().numpy())
    number = random.randint(1, 7) * 0.01
    src_in_trg = FDA_source_to_target_np(img_s, img_t, L=number)
    src_in_trg = torch.from_numpy(src_in_trg)
    src_in_trg = src_in_trg.type(torch.cuda.FloatTensor)
    return (torch.unsqueeze(src_in_trg, 0))


if __name__ == '__main__':

    args = parse_args()

    val = 1
    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    # source dataset
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)
    # val target dataset
    imdb_val, roidb_val, ratio_list_val, ratio_index_val = combined_roidb(args.imdbval_name_target)
    train_size_val = len(roidb_val)

    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))
    print('{:d} target roidbval entries'.format(len(roidb_val)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)

    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True)

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=args.num_workers)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    im_data_t = torch.FloatTensor(1)
    im_info_t = torch.FloatTensor(1)
    num_boxes_t = torch.LongTensor(1)
    gt_boxes_t = torch.FloatTensor(1)

    im_data_val = torch.FloatTensor(1)
    im_info_val = torch.FloatTensor(1)
    num_boxes_val = torch.LongTensor(1)
    gt_boxes_val = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

        im_data_t = im_data_t.cuda()
        im_info_t = im_info_t.cuda()
        num_boxes_t = num_boxes_t.cuda()
        gt_boxes_t = gt_boxes_t.cuda()

        im_data_val = im_data_val.cuda()
        im_info_val = im_info_val.cuda()
        num_boxes_val = num_boxes_val.cuda()
        gt_boxes_val = gt_boxes_val.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    im_data_t = Variable(im_data_t)
    im_info_t = Variable(im_info_t)
    num_boxes_t = Variable(num_boxes_t)
    gt_boxes_t = Variable(gt_boxes_t)

    im_data_val = Variable(im_data_val)
    im_info_val = Variable(im_info_val)
    num_boxes_val = Variable(num_boxes_val)
    gt_boxes_val = Variable(gt_boxes_val)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    from model.faster_rcnn.vgg16 import vgg16
    from model.faster_rcnn.resnet import resnet

    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, context=args.context)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, context=args.context)

    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    # params = []
    # for key, value in dict(fasterRCNN.named_parameters()).items():
    #     print(key)
    #     if value.requires_grad:
    #         if 'bias' in key:
    #             params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
    #                         'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
    #         else:
    #             params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    paramtxt = open('update/RCNN_base1.txt', 'r')
    param = paramtxt.readlines()
    RCNN_base1 = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_base1.append(name)

    paramtxt = open('update/RCNN_base2.txt', 'r')
    param = paramtxt.readlines()
    RCNN_base2 = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_base2.append(name)

    paramtxt = open('update/RCNN_top_base.txt', 'r')
    param = paramtxt.readlines()
    RCNN_top_base = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_top_base.append(name)

    paramtxt = open('update/RCNN_cls_score_base.txt', 'r')
    param = paramtxt.readlines()
    RCNN_cls_score_base = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_cls_score_base.append(name)

    paramtxt = open('update/RCNN_bbox_pred_base.txt', 'r')
    param = paramtxt.readlines()
    RCNN_bbox_pred_base = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_bbox_pred_base.append(name)

    paramtxt = open('update/RCNN_rpn.txt', 'r')
    param = paramtxt.readlines()
    RCNN_rpn = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_rpn.append(name)

    paramtxt = open('update/content.txt', 'r')
    param = paramtxt.readlines()
    content = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        content.append(name)

    paramtxt = open('update/RCNN_bbox_pred_di.txt', 'r')
    param = paramtxt.readlines()
    RCNN_bbox_pred_di = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_bbox_pred_di.append(name)

    paramtxt = open('update/RCNN_cls_score_di.txt', 'r')
    param = paramtxt.readlines()
    RCNN_cls_score_di = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_cls_score_di.append(name)

    paramtxt = open('update/RCNN_top_di.txt', 'r')
    param = paramtxt.readlines()
    RCNN_top_di = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        RCNN_top_di.append(name)

    paramtxt = open('update/netD_ds.txt', 'r')
    param = paramtxt.readlines()
    netD_ds = []
    for i in range(len(param)):
        name = str(param[i][:-1])
        netD_ds.append(name)

    content_p = [];
    RCNN_base1_p = [];
    RCNN_base2_p = [];
    netD_ds_p = [];
    RCNN_bbox_pred_base_p = [];
    RCNN_bbox_pred_di_p = [];
    RCNN_cls_score_base_p = [];
    RCNN_cls_score_di_p = [];
    RCNN_rpn_p = [];
    RCNN_top_base_p = [];
    RCNN_top_di_p = []

    for key, value in dict(fasterRCNN.named_parameters()).items():
        if key in RCNN_top_di:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_top_di_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                       'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_top_di_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        if key in content:
            if value.requires_grad:
                if 'bias' in key:
                    content_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                   'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    content_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_base1:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_base1_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                      'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_base1_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_base2:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_base2_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                      'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_base2_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in netD_ds:
            if value.requires_grad:
                if 'bias' in key:
                    netD_ds_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                   'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    netD_ds_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_bbox_pred_base:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_bbox_pred_base_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                               'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_bbox_pred_base_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_bbox_pred_di:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_bbox_pred_di_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                             'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_bbox_pred_di_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_cls_score_base:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_cls_score_base_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                               'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_cls_score_base_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_cls_score_di:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_cls_score_di_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                             'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_cls_score_di_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_rpn:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_rpn_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                    'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_rpn_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        elif key in RCNN_top_base:
            if value.requires_grad:
                if 'bias' in key:
                    RCNN_top_base_p += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                                         'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    RCNN_top_base_p += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    opt_di = torch.optim.SGD(content_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_netD_ds = torch.optim.SGD(netD_ds_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_base1 = torch.optim.SGD(RCNN_base1_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_base2 = torch.optim.SGD(RCNN_base2_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_bbox_pred_base = torch.optim.SGD(RCNN_bbox_pred_base_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_bbox_pred_di = torch.optim.SGD(RCNN_bbox_pred_di_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_cls_score_base = torch.optim.SGD(RCNN_cls_score_base_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_cls_score_di = torch.optim.SGD(RCNN_cls_score_di_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_rpn = torch.optim.SGD(RCNN_rpn_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_top_base = torch.optim.SGD(RCNN_top_base_p, momentum=cfg.TRAIN.MOMENTUM)
    opt_RCNN_top_di = torch.optim.SGD(RCNN_top_di_p, momentum=cfg.TRAIN.MOMENTUM)

    optimizer = [opt_di, opt_netD_ds, opt_RCNN_base1, opt_RCNN_base2, opt_RCNN_bbox_pred_di, opt_RCNN_cls_score_di,
                 opt_RCNN_rpn,
                 opt_RCNN_top_di, opt_RCNN_cls_score_base, opt_RCNN_bbox_pred_base, opt_RCNN_top_base]


    def reset_grad():
        opt_di.zero_grad()
        opt_RCNN_base1.zero_grad()
        opt_RCNN_base2.zero_grad()
        opt_netD_ds.zero_grad()
        opt_RCNN_bbox_pred_base.zero_grad()
        opt_RCNN_bbox_pred_di.zero_grad()
        opt_RCNN_cls_score_di.zero_grad()
        opt_RCNN_cls_score_base.zero_grad()
        opt_RCNN_rpn.zero_grad()
        opt_RCNN_top_base.zero_grad()
        opt_RCNN_top_di.zero_grad()


    def group_step(step_list):
        for i in range(len(step_list)):
            step_list[i].step()
        reset_grad()


    if args.cuda:
        fasterRCNN.cuda()

    if args.resume:
        checkpoint = torch.load(args.load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
    iters_per_epoch = int(train_size / args.batch_size)
    #iters_per_epoch = 3000
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs")
    count_iter = 0
    max_mAP = 0
    Gprototypes_s = np.zeros(2048, dtype=float)
    Gprototypes_t = np.zeros(2048, dtype=float)
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            lr_decay = args.lr_decay_gamma
            for m in range(len(optimizer)):
                adjust_learning_rate(optimizer[m], lr_decay)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)
        for step in range(iters_per_epoch):
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)

                data_s = next(data_iter_s)
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(dataloader_t)
                data_t = next(data_iter_t)
            eta = 1.0  # calc_supp(count_iter)
            count_iter += 1
            im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
            gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
            num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])

            im_data_t.data.resize_(data_t[0].size()).copy_(data_t[0])
            im_info_t.data.resize_(data_t[1].size()).copy_(data_t[1])
            gt_boxes_t.data.resize_(1, 1, 5).zero_()
            num_boxes_t.data.resize_(1).zero_()

            # First Step
            fasterRCNN.zero_grad()
            reset_grad()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, content_feat, \
            out_ds_s, prototypes_s = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, phase=1, eta=eta)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + (RCNN_loss_cls[0].mean() + RCNN_loss_bbox[0].mean() \
                      + RCNN_loss_cls[1].mean() + RCNN_loss_bbox[1].mean()) * 0.5

            domain_ds_s = Variable(torch.zeros(out_ds_s.size(0)).long().cuda())
            dloss_ds_s = 0.5 * FL(out_ds_s, domain_ds_s)

            out_ds_t, prototypes_t = fasterRCNN(im_data_t, im_info_t, gt_boxes_t, num_boxes_t, target=True, phase=1, eta=eta, )
            domain_ds_t = Variable(torch.ones(out_ds_t.size(0)).long().cuda())
            dloss_ds_t = 0.5 * FL(out_ds_t, domain_ds_t)

            #Get Pseudo-SAR images
            Fourier_im_data = Get_Fourier(im_data, im_data_t)

            F_content_feat = fasterRCNN(Fourier_im_data, im_info, gt_boxes, num_boxes, phase=1, eta=eta, Fourier=True)
            criterion = nn.L1Loss()
            Consis_loss = criterion(content_feat, F_content_feat)
            Consis_loss = Consis_loss / content_feat.shape[1] / content_feat.shape[2] / content_feat.shape[3]

            PSA_loss = 0
            if epoch > 3:
                prototypes_s = prototypes_s.cpu().detach().numpy()
                prototypes_t = prototypes_t.cpu().detach().numpy()
                if (prototypes_s.shape == prototypes_t.shape == Gprototypes_s.shape == Gprototypes_t.shape):
                    if np.sum(prototypes_s).item() != 0:
                        if np.sum(Gprototypes_s).item() == 0:
                            Gprototypes_s = prototypes_s
                        else:
                            p_norm = np.linalg.norm(prototypes_s)
                            gp_norm = np.linalg.norm(Gprototypes_s)
                            sim_s = (np.dot(prototypes_s, Gprototypes_s) / (p_norm * gp_norm) + 1) / 2
                            Gprototypes_s = sim_s * prototypes_s + (1 - sim_s) * Gprototypes_s
                    if np.sum(prototypes_t).item() != 0:
                        if np.sum(Gprototypes_t).item() == 0:
                            Gprototypes_t = prototypes_t
                        else:
                            p_norm = np.linalg.norm(prototypes_t)
                            gp_norm = np.linalg.norm(Gprototypes_t)
                            sim_t = (np.dot(prototypes_t, Gprototypes_t) / (p_norm * gp_norm) + 1) / 2
                            Gprototypes_t = sim_t * prototypes_t + (1 - sim_t) * Gprototypes_t
                if np.sum(Gprototypes_s).item() != 0 and np.sum(Gprototypes_t).item() != 0:
                    PSA_loss += np.linalg.norm(Gprototypes_s - Gprototypes_t) ** 2

            loss = loss + dloss_ds_t + dloss_ds_s + Consis_loss * 0.1 + PSA_loss * 0.01
            #loss = loss + dloss_ds_t + dloss_ds_s + Consis_loss + PSA_loss * 0.01
            loss_temp += loss.item()

            loss.backward()
            group_step(
                [opt_di, opt_RCNN_base1, opt_RCNN_base2, opt_netD_ds, opt_RCNN_bbox_pred_base, opt_RCNN_bbox_pred_di,
                 opt_RCNN_cls_score_base, opt_RCNN_cls_score_di, opt_RCNN_rpn, opt_RCNN_top_base, opt_RCNN_top_di])

            # Second Step
            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, MI_s, out_ds_s = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, phase=2, eta=eta)
            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + (RCNN_loss_cls[0].mean() + RCNN_loss_bbox[0].mean())

            loss_temp += loss.item()

            MI_t, out_ds_t = fasterRCNN(im_data_t, im_info_t, gt_boxes_t, num_boxes_t, phase=2, target=True, eta=eta)

            # print(MI_t)
            loss = loss + (MI_t + MI_s) * 0.5

            loss.backward()
            group_step([opt_di, opt_RCNN_bbox_pred_di, opt_RCNN_cls_score_di, opt_RCNN_top_di])

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                loss_rpn_cls = rpn_loss_cls.item()
                loss_rpn_box = rpn_loss_box.item()
                loss_rcnn_cls = RCNN_loss_cls[0].item()
                loss_rcnn_box = RCNN_loss_bbox[0].item()

                MI_s_p = MI_s.item()
                MI_t_p = MI_t.item()
                if PSA_loss!= 0 :
                    PSA_loss = PSA_loss.item()
                Consis_loss = Consis_loss.item()
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f, Consis_loss: %.4f, Mi_s_p: %.4f, Mi_t_p: %.4f, PSA_loss: %.4f " \
                    % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, Consis_loss, MI_s_p, MI_t_p, PSA_loss))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box,
                        'Consis_loss': Consis_loss
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                                       (epoch - 1) * iters_per_epoch + step)

                loss_temp = 0
                start = time.time()

        save_name = os.path.join(output_dir,'{}_{}_{}.pth'.format(args.dataset_t,epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

        if val == 1:
            load_name = '{}_{}_{}.pth'.format(args.dataset_t,epoch, step)
            print("Using " + load_name + " to val")
            # checkpoint = torch.load(args.load_name)
            # fasterRCNN.load_state_dict(checkpoint['model'])
            # if 'pooling_mode' in checkpoint.keys():
            #     cfg.POOLING_MODE = checkpoint['pooling_mode']
            # print('load model successfully!')

            if args.cuda:
                cfg.CUDA = True

            if args.cuda:
                fasterRCNN.cuda()

            start = time.time()
            max_per_image = 100

            thresh = 0.0
            save_name = args.load_name.split('/')[-1]
            num_images = len(imdb_val.image_index)
            all_boxes = [[[] for _ in xrange(num_images)]
                         for _ in xrange(imdb_val.num_classes)]

            output_dir_val = get_output_dir(imdb_val, save_name)
            dataset_val = roibatchLoader(roidb_val, ratio_list_val, ratio_index_val, 1, \
                                         imdb_val.num_classes, training=False, normalize=False)
            dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1,
                                                         shuffle=False, num_workers=0,
                                                         pin_memory=True)

            data_iter_val = iter(dataloader_val)

            _t = {'im_detect': time.time(), 'misc': time.time()}
            det_file = os.path.join(output_dir_val, 'detections.pkl')
            fasterRCNN.eval()
            empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
            for i in range(num_images):

                data_val = next(data_iter_val)
                im_data_val.data.resize_(data_val[0].size()).copy_(data_val[0])
                im_info_val.data.resize_(data_val[1].size()).copy_(data_val[1])
                gt_boxes_val.data.resize_(data_val[2].size()).copy_(data_val[2])
                num_boxes_val.data.resize_(data_val[3].size()).copy_(data_val[3])

                # gt_boxes = torch.unsqueeze(gt_boxes, 1)

                det_tic = time.time()
                rois_val, cls_prob_list_val, bbox_pred_list_val, \
                rpn_loss_cls_val, rpn_loss_box_val, \
                RCNN_loss_cls_val, RCNN_loss_bbox_val, \
                rois_label_val, _, _, _ = fasterRCNN(im_data_val, im_info_val, gt_boxes_val, num_boxes_val, phase=1)

                # scores = cls_prob.data
                # boxes = rois.data[:, :, 1:5]
                cls_prob_val = cls_prob_list_val[0] * 1.0 + cls_prob_list_val[1] * 0.0
                bbox_pred_val = bbox_pred_list_val[0] * 1.0 + bbox_pred_list_val[1] * 0.0
                scores_val = cls_prob_val.data
                boxes_val = rois_val.data[:, :, 1:5]

                if cfg.TEST.BBOX_REG:
                    # Apply bounding-box regression deltas
                    box_deltas = bbox_pred_val.data
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        # Optionally normalize targets by a precomputed mean and stdev
                        if args.class_agnostic:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(1, -1, 4 * len(imdb_val.classes))

                    pred_boxes_val = bbox_transform_inv(boxes_val, box_deltas, 1)
                    pred_boxes_val = clip_boxes(pred_boxes_val, im_info.data, 1)
                else:
                    # Simply repeat the boxes, once for each class
                    pred_boxes_val = np.tile(boxes, (1, scores.shape[1]))

                pred_boxes_val /= data_val[1][0][2].item()

                scores_val = scores_val.squeeze()
                pred_boxes_val = pred_boxes_val.squeeze()
                det_toc = time.time()
                detect_time = det_toc - det_tic
                misc_tic = time.time()

                for j in xrange(1, imdb_val.num_classes):
                    inds = torch.nonzero(scores_val[:, j] > thresh).view(-1)
                    # if there is det
                    if inds.numel() > 0:
                        cls_scores_val = scores_val[:, j][inds]
                        _, order = torch.sort(cls_scores_val, 0, True)
                        if args.class_agnostic:
                            cls_boxes_val = pred_boxes_val[inds, :]
                        else:
                            cls_boxes_val = pred_boxes_val[inds][:, j * 4:(j + 1) * 4]

                        cls_dets_val = torch.cat((cls_boxes_val, cls_scores_val.unsqueeze(1)), 1)
                        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                        cls_dets_val = cls_dets_val[order]
                        keep = nms(cls_dets_val, cfg.TEST.NMS)
                        cls_dets_val = cls_dets_val[keep.view(-1).long()]

                        all_boxes[j][i] = cls_dets_val.cpu().numpy()
                    else:
                        all_boxes[j][i] = empty_array

                # Limit to max_per_image detections *over all classes*
                if max_per_image > 0:
                    image_scores = np.hstack([all_boxes[j][i][:, -1]
                                              for j in xrange(1, imdb_val.num_classes)])
                    if len(image_scores) > max_per_image:
                        image_thresh = np.sort(image_scores)[-max_per_image]
                        for j in xrange(1, imdb_val.num_classes):
                            keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                            all_boxes[j][i] = all_boxes[j][i][keep, :]

                misc_toc = time.time()
                nms_time = misc_toc - misc_tic

                sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                                 .format(i + 1, num_images, detect_time, nms_time))
                sys.stdout.flush()

            with open(det_file, 'wb') as f:
                pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

            print('Evaluating detections')
            mAP = imdb_val.evaluate_detections(all_boxes, output_dir_val)
            if mAP > max_mAP:
                max_mAP = mAP
                save_name = os.path.join(output_dir, 'best.pth')
                save_checkpoint({
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)

    if args.use_tfboard:
        logger.close()

