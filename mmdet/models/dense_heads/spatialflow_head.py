import torch
import torch.nn as nn
from mmcv.cnn import normal_init, bias_init_with_prob, ConvModule

from mmdet.core import multi_apply
from ..builder import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module()
class SpatialFlowHead(AnchorHead):
    """SpatialFlow Head.
    There are four parallel sub-networks in the SpatialFlowHead,
    which are cls, reg, mask, and stuff sub-networks.

    We design spatial flows among all sub-networks, and also a stuff-mask path.

    Args:
        stacked_mask_convs (int): number of convs in mask sub-network.
        stacked_stuff_convs (int): number of convs in stuff sub-network.
        dcn_cls_convs_idx (list, optional): The indexes for applying
            deformable convs in cls sub-network.
        dcn_reg_convs_idx (list, optional): The indexes for applying
            deformable convs in reg sub-network.
        dcn_mask_convs_idx (list, optional): The indexes for applying
            deformable convs in mask sub-network.
        dcn_stuff_convs_idx (list, optional): The indexes for applying
            deformable convs in stuff sub-network.
        spatial_flows (list, optional): The spatial flow for sub-networks.
        stuff_mask_flow (bool): Whether to add stuff-mask flow in sub-networks.
        return_cls_feat (bool): Whether to return cls feat for refining.
        return_reg_feat (bool): Whether to return reg feat for refining.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 stacked_mask_convs=1,
                 stacked_stuff_convs=4,
                 dcn_cls_convs_idx=[0, 1, 2, 3],
                 dcn_reg_convs_idx=None,
                 dcn_mask_convs_idx=[0],
                 dcn_stuff_convs_idx=[0, 1, 2, 3],
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 **kwargs):
        self.stacked_convs = stacked_convs
        # mask and stuff sub-networks
        self.stacked_mask_convs = stacked_mask_convs
        self.stacked_stuff_convs = stacked_stuff_convs
        # deformable convs
        self.dcn_cls_convs_idx = \
            dcn_cls_convs_idx if dcn_cls_convs_idx is not None else []
        self.dcn_reg_convs_idx = \
            dcn_reg_convs_idx if dcn_reg_convs_idx is not None else []
        self.dcn_mask_convs_idx = \
            dcn_mask_convs_idx if dcn_mask_convs_idx is not None else []
        self.dcn_stuff_convs_idx = \
            dcn_stuff_convs_idx if dcn_stuff_convs_idx is not None else []
        assert isinstance(self.dcn_cls_convs_idx, list)
        assert isinstance(self.dcn_reg_convs_idx, list)
        assert isinstance(self.dcn_mask_convs_idx, list)
        assert isinstance(self.dcn_stuff_convs_idx, list)
        self.with_cls_dcn = len(self.dcn_cls_convs_idx) > 0
        self.with_reg_dcn = len(self.dcn_reg_convs_idx) > 0
        self.with_mask_dcn = len(self.dcn_mask_convs_idx) > 0
        self.with_stuff_dcn = len(self.dcn_stuff_convs_idx) > 0

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(SpatialFlowHead, self).__init__(
            num_classes, in_channels,
            anchor_generator=anchor_generator, **kwargs)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
        self.stuff_convs = nn.ModuleList()
        self.cls_spatial_flow_convs = nn.ModuleList()
        self.mask_spatial_flow_convs = nn.ModuleList()
        self.stuff_spatial_flow_convs = nn.ModuleList()
        self.stuff_mask_flow_convs = nn.ModuleList()

        # cls and reg sub-networks
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            # cls sub-net
            if self.with_cls_dcn and i in self.dcn_cls_convs_idx:
                conv_cfg = dict(type='ModulatedDeformConv')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))
            # reg sub-net
            if self.with_reg_dcn and i in self.dcn_reg_convs_idx:
                conv_cfg = dict(type='ModulatedDeformConvPack')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))

        # mask sub-net
        for i in range(self.stacked_mask_convs):
            if self.with_mask_dcn and i in self.dcn_mask_convs_idx:
                conv_cfg = dict(type='ModulatedDeformConv')
            else:
                conv_cfg = self.conv_cfg
            self.mask_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))

        # stuff sub-net
        for i in range(self.stacked_stuff_convs):
            if self.with_stuff_dcn and i in self.dcn_stuff_convs_idx:
                conv_cfg = dict(type='ModulatedDeformConv')
            else:
                conv_cfg = self.conv_cfg
            self.stuff_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))

        # information flows
        for i in range(self.stacked_convs):
            if self.with_cls_dcn and i in self.dcn_cls_convs_idx:
                out_channels = 3 * 9
            else:
                out_channels = self.feat_channels
            self.cls_spatial_flow_convs.append(
                ConvModule(
                    self.feat_channels,
                    out_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None))

        for i in range(self.stacked_mask_convs):
            if self.with_mask_dcn and i in self.dcn_mask_convs_idx:
                out_channels = 3 * 9
            else:
                out_channels = self.feat_channels
            self.mask_spatial_flow_convs.append(
                ConvModule(
                    self.feat_channels,
                    out_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None))

        for i in range(self.stacked_stuff_convs):
            if self.with_stuff_dcn and i in self.dcn_stuff_convs_idx:
                out_channels = 3 * 9
            else:
                out_channels = self.feat_channels
            self.stuff_spatial_flow_convs.append(
                ConvModule(
                    self.feat_channels,
                    out_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None))

        stuff_mask_flow_stacked_convs = min(self.stacked_mask_convs,
                                            self.stacked_stuff_convs)
        for i in range(stuff_mask_flow_stacked_convs):
            self.stuff_mask_flow_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None))

        # retina cls and reg
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.mask_convs:
            normal_init(m.conv, std=0.01)
        for m in self.stuff_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cls_spatial_flow_convs:
            normal_init(m.conv, std=0.01)
        for m in self.mask_spatial_flow_convs:
            normal_init(m.conv, std=0.01)
        for m in self.stuff_spatial_flow_convs:
            normal_init(m.conv, std=0.01)
        for m in self.stuff_mask_flow_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single_bbox(self, x):
        cls_feat = x
        reg_feat = x
        for i, cls_conv, reg_conv, cls_spatial_flow_conv in zip(
                range(self.stacked_convs), self.cls_convs, self.reg_convs,
                self.cls_spatial_flow_convs):
            # reg subnet
            reg_feat = reg_conv(reg_feat)

            # cls subnet
            if self.with_cls_dcn and i in self.dcn_cls_convs_idx:
                out = cls_spatial_flow_conv(cls_feat + reg_feat)
                o1, o2, mask = torch.chunk(out, 3, dim=1)
                offset = torch.cat((o1, o2), dim=1)
                mask = torch.sigmoid(mask)
                cls_feat = cls_conv(cls_feat, offset, mask)
            else:
                cls_feat = cls_conv(cls_feat + cls_spatial_flow_conv(reg_feat))
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def forward_single_bbox_stuff(self, x):
        cls_feat = x
        reg_feat = x
        mask_feat = x
        stuff_feat = x
        for i, cls_conv, reg_conv, cls_spatial_flow_conv in zip(
                range(self.stacked_convs), self.cls_convs, self.reg_convs,
                self.cls_spatial_flow_convs):
            # reg subnet
            reg_feat = reg_conv(reg_feat)

            # cls subnet
            if self.with_cls_dcn and i in self.dcn_cls_convs_idx:
                out = cls_spatial_flow_conv(cls_feat + reg_feat)
                o1, o2, mask = torch.chunk(out, 3, dim=1)
                offset = torch.cat((o1, o2), dim=1)
                mask = torch.sigmoid(mask)
                cls_feat = cls_conv(cls_feat, offset, mask)
            else:
                cls_feat = cls_conv(cls_feat + cls_spatial_flow_conv(reg_feat))

            # stuff subnet
            stuff_conv = self.stuff_convs[i]
            stuff_spatial_flow_conv = self.stuff_spatial_flow_convs[i]

            if self.with_stuff_dcn and i in self.dcn_stuff_convs_idx:
                out = stuff_spatial_flow_conv(stuff_feat + reg_feat)
                o1, o2, mask = torch.chunk(out, 3, dim=1)
                offset = torch.cat((o1, o2), dim=1)
                mask = torch.sigmoid(mask)
                stuff_feat = stuff_conv(stuff_feat, offset, mask)
            else:
                stuff_feat = stuff_conv(stuff_feat +
                                        stuff_spatial_flow_conv(reg_feat))

            # mask subnet
            mask_idx = self.stacked_convs - i - 1
            if mask_idx == 0:
                # NOTE: one only conv in mask sub-net
                mask_conv = self.mask_convs[0]
                mask_spatial_flow_conv = self.mask_spatial_flow_convs[0]
                stuff_mask_flow_conv = self.stuff_mask_flow_convs[0]

                if self.with_mask_dcn and mask_idx in \
                        self.dcn_mask_convs_idx:
                    out = mask_spatial_flow_conv(mask_feat + reg_feat)
                    o1, o2, mask = torch.chunk(out, 3, dim=1)
                    offset = torch.cat((o1, o2), dim=1)
                    mask = torch.sigmoid(mask)
                    mask_feat = mask_feat + stuff_mask_flow_conv(
                        stuff_feat)
                    mask_feat = mask_conv(mask_feat, offset, mask)
                else:
                    mask_feat = mask_feat + stuff_mask_flow_conv(
                        stuff_feat)
                    mask_feat = mask_conv(mask_feat +
                                          mask_spatial_flow_conv(reg_feat))
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred, stuff_feat, mask_feat

    def forward(self, feats):
        cls_score_list = []
        bbox_pred_list = []
        (cls_score_list1, bbox_pred_list1, stuff_feat_list,
         mask_feats_list) = multi_apply(
            self.forward_single_bbox_stuff, feats[:3])
        cls_score_list.extend(cls_score_list1)
        bbox_pred_list.extend(bbox_pred_list1)

        cls_score_list2, bbox_pred_list2 = multi_apply(
            self.forward_single_bbox, feats[3:])
        cls_score_list.extend(cls_score_list2)
        bbox_pred_list.extend(bbox_pred_list2)

        return (cls_score_list, bbox_pred_list,
                tuple(stuff_feat_list), tuple(mask_feats_list))
