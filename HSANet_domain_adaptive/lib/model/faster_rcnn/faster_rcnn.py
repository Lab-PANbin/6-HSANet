import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta, grad_reverse


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data, im_info, gt_boxes, num_boxes, phase=1, target=False, eta=1.0, Fourier=False):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        if Fourier == True:
            base_feat1 = self.RCNN_base1(im_data)
            base_feat = self.RCNN_base2(base_feat1)
            base_feat_content = self.Encoder_Content(base_feat)
            # content_feat = F.avg_pool2d(base_feat_content, (base_feat_content.shape[2], base_feat_content.shape[3]))[:, :, 0, 0]
            return base_feat_content

        if phase == 1:

            # feed image data to base model to obtain base feature map
            base_feat1 = self.RCNN_base1(im_data)
            base_feat = self.RCNN_base2(base_feat1)
            base_feat_content = self.Encoder_Content(base_feat)  # torch.Size([1, 1024, 38, 50])

            if target == False:
                domain_ds = self.netD_ds(grad_reverse(base_feat))
                if self.training:
                    self.RCNN_rpn.train()
                # feed base feature map tp RPN to obtain rois
                rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_content, im_info, gt_boxes, num_boxes)

                # if it is training phrase, then use ground trubut bboxes for refining
                if self.training:
                    roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
                    rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

                    rois_label = Variable(rois_label.view(-1).long())
                    rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
                    rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
                    rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
                else:
                    rois_label = None
                    rois_target = None
                    rois_inside_ws = None
                    rois_outside_ws = None
                    rpn_loss_cls = 0
                    rpn_loss_bbox = 0

                rois = Variable(rois)
                # do roi pooling based on predicted rois

                # if self.training:
                #     num = num_boxes.item()
                #     n = gt_boxes.shape[1]
                #     gt_rois = rois[:,:num,:]
                #     gt_rois[:,:num,1:]= gt_boxes[:,:num,:-1]  # gt_boxes[batchsize, box_num, 5]
                #     pooled_feat_gt = self.RCNN_roi_align(base_feat_content, gt_rois.view(-1, 5))
                #     pooled_feat_gt_out = self._head_to_tail_di(pooled_feat_gt)
                #     prototypes = self.get_prototypes_gt(pooled_feat_gt_out)
                # else:
                #     prototypes = 0

                if cfg.POOLING_MODE == 'align':
                    pooled_feat_di = self.RCNN_roi_align(base_feat_content,rois.view(-1, 5))  # torch.Size([128, 1024, 7, 7])
                    pooled_feat = self.RCNN_roi_align(base_feat, rois.detach().view(-1, 5))

                cls_prob_list = []
                bbox_pred_list = []
                RCNN_loss_cls_list = []
                RCNN_loss_bbox_list = []

                pooled_elem = pooled_feat_di
                pooled_feat_di_out = self._head_to_tail_di(pooled_elem)

                # compute bbox offset
                bbox_pred_di = self.RCNN_bbox_pred_di(pooled_feat_di_out)
                if self.training and not self.class_agnostic:
                    # select the corresponding columns according to roi labels
                    bbox_pred_view = bbox_pred_di.view(bbox_pred_di.size(0), int(bbox_pred_di.size(1) / 4),
                                                       4)
                    bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                                    rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0),
                                                                                                     1,
                                                                                                     4))
                    bbox_pred_di = bbox_pred_select.squeeze(1)

                cls_score_di = self.RCNN_cls_score_di(pooled_feat_di_out)  # [128, 2]
                cls_prob_di = F.softmax(cls_score_di, 1)  # [128,2]  lable: [128]

                RCNN_loss_cls_di = 0
                RCNN_loss_bbox_di = 0

                if self.training:
                    # classification loss
                    RCNN_loss_cls_di = F.cross_entropy(cls_score_di, rois_label)
                    # bounding box regression L1 loss
                    RCNN_loss_bbox_di = _smooth_l1_loss(bbox_pred_di, rois_target, rois_inside_ws, rois_outside_ws)

                pooled_elem = pooled_feat
                pooled_feat_base_out = self._head_to_tail_base(pooled_elem)
                bbox_pred_base = self.RCNN_bbox_pred_base(pooled_feat_base_out)

                if self.training and not self.class_agnostic:
                    bbox_pred_view = bbox_pred_base.view(bbox_pred_base.size(0), int(bbox_pred_base.size(1) / 4), 4)
                    bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                                    rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0),
                                                                                                     1,
                                                                                                     4))
                    bbox_pred_base = bbox_pred_select.squeeze(1)
                cls_score_base = self.RCNN_cls_score_base(pooled_feat_base_out)
                cls_prob_base = F.softmax(cls_score_base, 1)

                #Get Re-weighted Prototypes
                cls_pre_label1 = cls_prob_di.argmax(1).detach()
                cls_pre_label2 = cls_prob_base.argmax(1).detach()
                cls_pre_label = cls_pre_label2 + cls_pre_label1
                cls_prob_temp = (cls_prob_di + cls_prob_base)/2
                target_weight = []
                for i in range(len(cls_pre_label1)):
                    label_i = cls_pre_label[i].item()
                    if label_i == 2:
                        target_weight.append(cls_prob_temp[i][1])
                    else:
                        target_weight.append(1.0)

                prototypes = self.get_prototypes(pooled_feat_di_out, cls_pre_label, target_weight)

                RCNN_loss_cls_base = 0
                RCNN_loss_bbox_base = 0

                if self.training:
                    # classification loss
                    RCNN_loss_cls_base = F.cross_entropy(cls_score_base, rois_label)
                    RCNN_loss_bbox_base = _smooth_l1_loss(bbox_pred_base, rois_target, rois_inside_ws, rois_outside_ws)

                cls_prob_di = cls_prob_di.view(batch_size, rois.size(1), -1)
                bbox_pred_di = bbox_pred_di.view(batch_size, rois.size(1), -1)
                cls_prob_base = cls_prob_base.view(batch_size, rois.size(1), -1)
                bbox_pred_base = bbox_pred_base.view(batch_size, rois.size(1), -1)

                cls_prob_list.append(cls_prob_di)
                bbox_pred_list.append(bbox_pred_di)
                RCNN_loss_cls_list.append(RCNN_loss_cls_di)
                RCNN_loss_bbox_list.append(RCNN_loss_bbox_di)

                cls_prob_list.append(cls_prob_base)
                bbox_pred_list.append(bbox_pred_base)
                RCNN_loss_cls_list.append(RCNN_loss_cls_base)
                RCNN_loss_bbox_list.append(RCNN_loss_bbox_base)

                return rois, cls_prob_list, bbox_pred_list, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls_list, \
                       RCNN_loss_bbox_list, rois_label, base_feat_content, domain_ds, prototypes

            elif target == True:
                domain_ds = self.netD_ds(grad_reverse(base_feat))
                self.RCNN_rpn.eval()  #  Essential Step

                # feed base feature map tp RPN to obtain rois
                rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_content, im_info, gt_boxes, num_boxes)

                if self.training:
                    roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
                    rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

                    rois_label = Variable(rois_label.view(-1).long())
                    rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
                    rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
                    rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
                else:
                    rois_label = None
                    rois_target = None
                    rois_inside_ws = None
                    rois_outside_ws = None
                    rpn_loss_cls = 0
                    rpn_loss_bbox = 0

                rois = Variable(rois)
                # do roi pooling based on predicted rois
                if cfg.POOLING_MODE == 'align':
                    pooled_feat_content = self.RCNN_roi_align(base_feat_content, rois.view(-1, 5))     #torch.Size([128, 1024, 7, 7])
                    pooled_feat = self.RCNN_roi_align(base_feat, rois.detach().view(-1, 5))

                pooled_elem = pooled_feat_content
                pooled_feat_di_out = self._head_to_tail_di(pooled_elem)
                cls_score_di = self.RCNN_cls_score_di(pooled_feat_di_out)
                cls_prob_di = F.softmax(cls_score_di, 1)   #[300, 2]

                pooled_elem = pooled_feat
                pooled_feat_base_out = self._head_to_tail_base(pooled_elem)
                cls_score_base = self.RCNN_cls_score_base(pooled_feat_base_out)
                cls_prob_base = F.softmax(cls_score_base, 1)

                cls_pre_label1 = cls_prob_di.argmax(1).detach()
                cls_pre_label2 = cls_prob_base.argmax(1).detach()
                cls_pre_label = cls_pre_label2 + cls_pre_label1
                cls_prob_temp = (cls_prob_di + cls_prob_base)/2
                target_weight = []
                for i in range(len(cls_pre_label)):
                    label_i = cls_pre_label[i].item()
                    if label_i == 2 :
                        #diff_value = torch.exp(cls_prob_temp[i][1]).item()
                        target_weight.append(cls_prob_temp[i][1])
                    else:
                        target_weight.append(1.0)
                prototypes = self.get_prototypes(pooled_feat_di_out, cls_pre_label, target_weight)
                return domain_ds, prototypes

        elif phase == 2:
            # feed image data to base model to obtain base feature map
            base_feat1 = self.RCNN_base1(im_data)
            base_feat = self.RCNN_base2(base_feat1)
            base_feat_content = self.Encoder_Content(base_feat)
            base_feat_style = base_feat - base_feat_content

            if target == False:
                domain_ds = self.netD_ds(grad_reverse(base_feat))
                self.RCNN_rpn.train()
                # feed base feature map tp RPN to obtain rois
                rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_content, im_info, gt_boxes, num_boxes)

                # if it is training phrase, then use ground trubut bboxes for refining
                if self.training:
                    roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
                    rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

                    rois_label = Variable(rois_label.view(-1).long())
                    rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
                    rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
                    rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
                else:
                    rois_label = None
                    rois_target = None
                    rois_inside_ws = None
                    rois_outside_ws = None
                    rpn_loss_cls = 0
                    rpn_loss_bbox = 0

                rois = Variable(rois)
                # do roi pooling based on predicted rois

                if cfg.POOLING_MODE == 'align':
                    pooled_feat_di = self.RCNN_roi_align(base_feat_content, rois.view(-1, 5))
                    pooled_feat_style = self.RCNN_roi_align(base_feat_style, rois.view(-1, 5))

                Mutual_invariant = F.avg_pool2d(pooled_feat_di, (7, 7))[:, :, 0, 0]
                Mutual_specific = F.avg_pool2d(pooled_feat_style, (7, 7))[:, :, 0, 0]
                Mutual_invariant = F.normalize(Mutual_invariant, dim=1)
                Mutual_specific = F.normalize(Mutual_specific, dim=1)
                Mutual_loss = Mutual_invariant * Mutual_specific
                Mutual_loss = torch.abs(torch.sum(Mutual_loss, dim=1))
                Mutual_loss = torch.mean(Mutual_loss)

                cls_prob_list = []
                bbox_pred_list = []
                RCNN_loss_cls_list = []
                RCNN_loss_bbox_list = []

                pooled_elem = pooled_feat_di
                pooled_feat_di_out = self._head_to_tail_di(pooled_elem)

                # compute bbox offset
                bbox_pred_di = self.RCNN_bbox_pred_di(pooled_feat_di_out)

                if self.training and not self.class_agnostic:
                    # select the corresponding columns according to roi labels
                    bbox_pred_view = bbox_pred_di.view(bbox_pred_di.size(0), int(bbox_pred_di.size(1) / 4),
                                                       4)
                    bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                                    rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0),
                                                                                                     1,
                                                                                                     4))
                    bbox_pred_di = bbox_pred_select.squeeze(1)

                # compute object classification probability
                cls_score_di = self.RCNN_cls_score_di(pooled_feat_di_out)
                cls_prob_di = F.softmax(cls_score_di, 1)

                RCNN_loss_cls_di = 0
                RCNN_loss_bbox_di = 0

                if self.training:
                    # classification loss
                    RCNN_loss_cls_di = F.cross_entropy(cls_score_di, rois_label)
                    # bounding box regression L1 loss
                    RCNN_loss_bbox_di = _smooth_l1_loss(bbox_pred_di, rois_target, rois_inside_ws, rois_outside_ws)

                cls_prob_di = cls_prob_di.view(batch_size, rois.size(1), -1)
                bbox_pred_di = bbox_pred_di.view(batch_size, rois.size(1), -1)

                cls_prob_list.append(cls_prob_di)
                bbox_pred_list.append(bbox_pred_di)
                RCNN_loss_cls_list.append(RCNN_loss_cls_di)
                RCNN_loss_bbox_list.append(RCNN_loss_bbox_di)

                return rois, cls_prob_list, bbox_pred_list, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls_list, RCNN_loss_bbox_list, rois_label, Mutual_loss, domain_ds
            else:
                # if it is training phrase, then use ground trubut bboxes for refining
                domain_ds = self.netD_ds(grad_reverse(base_feat))
                self.RCNN_rpn.eval()

                # feed base feature map tp RPN to obtain rois
                rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_content, im_info, gt_boxes, num_boxes)

                if self.training:
                    roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
                    rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

                    rois_label = Variable(rois_label.view(-1).long())
                    rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
                    rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
                    rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
                else:
                    rois_label = None
                    rois_target = None
                    rois_inside_ws = None
                    rois_outside_ws = None
                    rpn_loss_cls = 0
                    rpn_loss_bbox = 0

                rois = Variable(rois)
                # do roi pooling based on predicted rois

                if cfg.POOLING_MODE == 'align':
                    pooled_feat_di = self.RCNN_roi_align(base_feat_content, rois.view(-1, 5))
                    # pooled_feat = self.RCNN_roi_align(base_feat, rois.detach().view(-1, 5))
                    pooled_feat_style = self.RCNN_roi_align(base_feat_style, rois.view(-1, 5))

                Mutual_invariant = F.avg_pool2d(pooled_feat_di, (7, 7))[:, :, 0, 0]
                Mutual_specific = F.avg_pool2d(pooled_feat_style, (7, 7))[:, :, 0, 0]
                Mutual_invariant = F.normalize(Mutual_invariant, dim=1)
                Mutual_specific = F.normalize(Mutual_specific, dim=1)
                Mutual_loss = Mutual_invariant * Mutual_specific
                Mutual_loss = torch.abs(torch.sum(Mutual_loss, dim=1))
                Mutual_loss = torch.mean(Mutual_loss)

                return Mutual_loss, domain_ds

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

# get_prototypes.  SHIP CLASS ONLY//SHIP CLASS ONLY//SHIP CLASS ONLY//SHIP CLASS ONLY//SHIP CLASS ONLY//SHIP CLASS ONLY
    def get_prototypes(self, pooled_feat, label, weight):
        label = label.cpu().numpy().reshape(-1)
        prototypes = torch.zeros(pooled_feat.size(1))
        w = 0
        if cfg.CUDA:
            prototypes = prototypes.cuda()
        for i in range(len(label)):
            label_i = label[i].item()
            if label_i > 1:
                prototypes += pooled_feat[i] * weight[i]
                w += weight[i]
        if w!=0:
            prototypes = torch.div(prototypes, w)
        return prototypes