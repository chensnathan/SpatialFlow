import math

import numpy as np
from mmcv.cnn import constant_init, kaiming_init, ConvModule
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import force_fp32
from ..builder import build_loss, HEADS


def _make_stuff_lateral_layers(num_conv,
                               num_upsample,
                               in_channels,
                               feat_channels,
                               conv_cfg=None,
                               norm_cfg=dict(type='GN', num_groups=32),
                               upsample_method='bilinear'):
    """Tool function to make stuff lateral layers.

    There are two situation:
        1. the feature map does not need to upsample, only a 3x3 conv is needed
        2. the feature map needs to upsample, each time we upsample 2x,
           we use a combine of 3x3 conv + GN + ReLU + Bilinear Upsample 2x.

    Args:
        num_conv (int): number of conv layers needed in the lateral layer.
        num_upsample (int): number of upsample times needed in the lateral
            layer.
        in_channels (int): number of channels of input.
        feat_channels (int): number of channels of feature maps.
        conv_cfg (dict, optional): conv layer config.
        norm_cfg (dict, optional): norm layer config, we use `GN`.
    """
    lateral_conv = []
    if num_conv == 1 and num_upsample == 0:
        lateral_conv.append(
            ConvModule(
                in_channels,
                feat_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))
    else:
        assert num_conv == num_upsample, \
            'each upsampling stage consist of: ' \
            '3x3 conv + GN + ReLU + 2x bilinea upsample.'
        for i in range(num_conv):
            in_chns = in_channels if i == 0 else feat_channels
            lateral_conv.append(
                ConvModule(
                    in_chns,
                    feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
            lateral_conv.append(nn.Upsample(
                scale_factor=2, mode=upsample_method, align_corners=True))
    return nn.Sequential(*lateral_conv)


@HEADS.register_module()
class StuffHead(nn.Module):
    """Head for segmenting the `stuffs` in Panoptic Segmentation.

    Args:
        stuff_num_classes (int): Number of classes in stuff + background.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        feat_indexes (Iterable): The indexes to get the features in FPN.
        feat_strides (Iterable): The strides of feature map.
        out_stride (int): The stride of the result feature map.
        conv_cfg (dict, optional): conv layer config.
        norm_cfg (dict, optional): norm layer config, we use `GN`.
    """  # noqa: W605

    def __init__(self,
                 stuff_num_classes,
                 in_channels,
                 feat_channels=128,
                 feat_indexes=[0, 1, 2, 3],
                 feat_strides=[4, 8, 16, 32],
                 out_stride=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32),
                 loss_stuff=dict(
                     type='CrossEntropyLoss',
                     ignore_index=255,
                     loss_weight=1.0),
                 upsample_method='bilinear'):
        super(StuffHead, self).__init__()
        self.stuff_num_classes = stuff_num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.feat_indexes = feat_indexes
        self.feat_strides = feat_strides
        self.out_stride = out_stride
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        # add ignore index for loss
        assert hasattr(loss_stuff, 'ignore_index')
        self.ignore_index = loss_stuff.pop('ignore_index')
        self.loss_stuff = build_loss(loss_stuff)
        self.upsample_method = upsample_method
        # fp16
        self.fp16_enabled = False

        self.num_ups = [int(math.log2(feat_stride // self.out_stride))
                        for feat_stride in self.feat_strides]
        self.num_lateral_convs = [num_up + 1
                                  if num_up == 0 else num_up
                                  for num_up in self.num_ups]
        self._init_layers()

    def _init_layers(self):
        # build lateral convs for stuff head
        self.lateral_convs = nn.ModuleList()
        for num_conv, num_up in zip(self.num_lateral_convs, self.num_ups):
            lateral_conv = _make_stuff_lateral_layers(
                num_conv, num_up, self.in_channels, self.feat_channels,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                upsample_method=self.upsample_method)
            self.lateral_convs.append(lateral_conv)
        self.conv = nn.Conv2d(self.feat_channels,
                              self.stuff_num_classes,
                              kernel_size=1)
        self.upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)

    def init_weights(self):
        for m in self.lateral_convs:
            for mm in m.modules():
                if isinstance(mm, nn.Conv2d):
                    kaiming_init(mm)
                elif isinstance(mm, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(mm, 1)
        kaiming_init(self.conv)

    def forward_single(self, x, lateral_conv):
        up_feat = lateral_conv(x)
        return up_feat

    def forward(self, feats, return_feats=False):
        up_feats = [self.forward_single(feats[idx], lateral_conv)
                    for idx, lateral_conv in zip(self.feat_indexes,
                                                 self.lateral_convs)]
        feat_outs = sum(up_feats)
        score_maps = self.conv(feat_outs)
        score_maps = self.upsample(score_maps)
        return score_maps

    @force_fp32(apply_to=('score_maps',))
    def loss(self, score_maps, stuff_map_targets, img_metas):
        valid_h, valid_w, _ = img_metas[0]['img_shape']
        stuff_weights = stuff_map_targets.new_zeros(stuff_map_targets.size(),
                                                    dtype=torch.float)
        stuff_weights[..., :valid_h, :valid_w] = 1.
        valid_num_samples = max(
            torch.sum(stuff_weights > 0).float().item(), 1.)
        # loss stuff
        loss_stuff = self.loss_stuff(score_maps,
                                     stuff_map_targets.squeeze(1).long(),
                                     stuff_weights,
                                     avg_factor=valid_num_samples,
                                     ignore_index=self.ignore_index)
        return dict(loss_stuff=loss_stuff)

    @force_fp32(apply_to=('score_maps',))
    def get_stuff_map(self, score_maps, img_metas, rescale=True):
        """Get the predicted stuff semantic segmentation masks.

        Args:
            score_maps (Tensor or ndarray): shape (n, #class, h, w). For
                single-scale testing, `score_maps` is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            img_metas (list[dict]): the information for images
            rescale (bool): rescale to original shape or not.
        Returns:
            list[list]: encoded stuff mask
        """
        if isinstance(score_maps, torch.Tensor):
            stuff_map = score_maps.argmax(1).cpu().numpy()
        else:
            stuff_map = score_maps
        assert isinstance(stuff_map, np.ndarray)

        num_imgs = stuff_map.shape[0]
        stuff_segms = []

        for i in range(num_imgs):
            img_meta = img_metas[i]
            img_h, img_w = img_meta['ori_shape'][:2]
            valid_h, valid_w, _ = img_meta['img_shape']
            stuffs = stuff_map[i, ...].astype(np.uint8)
            stuffs = stuffs[:valid_h, :valid_w]
            stuff_segms.append(
                dict(stuff_map=stuffs, img_shape=(img_w, img_h)))
        return stuff_segms

    @force_fp32(apply_to=('score_maps',))
    def get_stuff_map_aug(self,
                          score_maps,
                          img_metas,
                          rescale=True,
                          weights=None):
        """Get the predicted stuff semantic segmentation masks.

        Args:
            score_maps (list[Tensor or ndarray]): each element with the shape
                (n, #class, h, w). For single-scale testing, `score_maps` is
                the direct output of model, whose type is Tensor, while for
                multi-scale testing, it will be converted to numpy array
                outside of this method.
            img_metas (list[list[dict]]): the information for images
            rescale (bool): rescale to original shape or not.
            weights (list or Tensor): weights to re-weight the stuff map.
        Returns:
            list[list]: encoded stuff mask
        """
        assert isinstance(score_maps, list)
        num_scales = len(score_maps)
        num_imgs = score_maps[0].shape[0]
        assert num_scales == len(img_metas)
        stuff_segms = []

        for i in range(num_imgs):
            score_map_img_size = (num_scales, score_maps[0].shape[1], ) + \
                img_metas[0][0]['ori_shape'][:2]
            score_map_img = score_maps[0].new_zeros(size=score_map_img_size)
            for s in range(num_scales):
                img_meta = img_metas[s][i]
                img_h, img_w = img_meta['ori_shape'][:2]
                valid_h, valid_w, _ = img_meta['img_shape']
                flip = img_meta['flip']
                score_map_img_scale = score_maps[s][i]
                score_map_img_scale = score_map_img_scale[
                    ..., :valid_h, :valid_w]
                if rescale:
                    score_map_img_scale = F.interpolate(
                        score_map_img_scale.unsqueeze(0), size=(img_h, img_w),
                        mode='bilinear', align_corners=True)
                if flip:
                    inv_idx = torch.arange(
                        score_map_img_scale.size(-1) - 1, -1, -1).long()
                    score_map_img_scale = score_map_img_scale[..., inv_idx]
                score_map_img[s, ...] = score_map_img_scale.squeeze(0)
            if weights is None:
                score_map_img_avg = torch.mean(score_map_img, dim=0)
            else:
                weights = score_map_img.new_tensor(weights)
                weights = weights.reshape(num_scales, 1, 1, 1) / weights.sum()
                score_map_img_avg = torch.sum(weights * score_map_img, dim=0)
            stuff_map = score_map_img_avg.argmax(0).cpu().numpy()

            stuffs = stuff_map.astype(np.uint8)
            stuff_segms.append(
                dict(stuff_map=stuffs,
                     img_shape=img_metas[0][0]['ori_shape'][:2]))
        return stuff_segms
