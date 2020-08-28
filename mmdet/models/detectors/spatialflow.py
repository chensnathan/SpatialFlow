import numpy as np
import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms, bbox_mapping)
from ..builder import (DETECTORS, build_backbone, build_head, build_neck,
                       build_roi_extractor)
from .base import BaseDetector


@DETECTORS.register_module()
class SpatialFlow(BaseDetector):
    """SpatialFlow for Panoptic Segmentation."""

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 stuff_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SpatialFlow, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        if bbox_head is not None:
            if train_cfg is not None:
                bbox_head.update(train_cfg=train_cfg.single_stage)
            bbox_head.update(test_cfg=test_cfg.single_stage)
            self.bbox_head = build_head(bbox_head)
        if mask_head is not None:
            self.mask_roi_extractor = build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = build_head(mask_head)
        if stuff_head is not None:
            self.stuff_head = build_head(stuff_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    @property
    def with_stuff(self):
        return hasattr(self, 'stuff_head') and self.stuff_head is not None

    def init_weights(self, pretrained=None):
        super(SpatialFlow, self).init_weights()
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_bbox:
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_roi_extractor.init_weights()
            self.mask_head.init_weights()
        if self.with_stuff:
            self.stuff_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        x = self.extract_feat(img)

        losses = dict()

        # bbox head forward and loss
        bbox_outs = self.bbox_head(x)
        bbox_loss_inputs = bbox_outs[:2] + (
            gt_bboxes, gt_labels, img_metas)
        bbox_losses = self.bbox_head.loss(
            *bbox_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(bbox_losses)

        # mask
        # get det bboxes for each image after nms
        bbox_inputs = bbox_outs[:2] + (
            img_metas, self.train_cfg.single_stage_nms)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        single_stage_bboxes_list = []
        for bbox, labels in bbox_list:
            single_stage_bboxes_list.append(bbox[:, :4])

        # assign the bboxes and labels for mask
        mask_assigner = build_assigner(
            self.train_cfg.single_stage_mask.assigner)
        mask_sampler = build_sampler(
            self.train_cfg.single_stage_mask.sampler, context=self)
        num_imgs = img.size(0)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = mask_assigner.assign(
                single_stage_bboxes_list[i],
                gt_bboxes[i],
                gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = mask_sampler.sample(
                assign_result,
                single_stage_bboxes_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        # get all rois
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])

        # mask subnet
        mask_roi_feats = bbox_outs[-1]

        # mask head forward and loss
        mask_feats = self.mask_roi_extractor(
            mask_roi_feats[:self.mask_roi_extractor.num_inputs], pos_rois)
        mask_pred = self.mask_head(mask_feats)

        mask_targets = self.mask_head.get_targets(
            sampling_results, gt_masks, self.train_cfg.single_stage_mask)
        pos_labels = torch.cat(
            [res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                        pos_labels)
        losses.update(loss_mask)

        # stuff head
        # stuff subnet
        stuff_head_inputs = bbox_outs[-2]

        stuff_outs = self.stuff_head(
            stuff_head_inputs[:len(self.stuff_head.feat_strides)])
        loss_stuff = self.stuff_head.loss(
            stuff_outs, gt_semantic_seg, img_metas)
        losses.update(loss_stuff)

        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)

        bbox_outs = self.bbox_head(x)
        bbox_inputs = bbox_outs[:2] + (
            img_meta, self.test_cfg.single_stage, rescale)
        det_results = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in det_results
        ]

        # mask subnet
        mask_roi_feats = bbox_outs[-1]
        # det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        _bboxes = []
        for results, meta in zip(det_results, img_meta):
            det_bboxes, _ = results
            scale_factor = meta['scale_factor']
            if rescale and isinstance(scale_factor, (float, np.ndarray)):
                scale_factor = det_bboxes.new_tensor(scale_factor)
            # refer to `bbox_nms.py` for details that we add a zeros bboxes
            # tensor to handle the situation that there is no `det_bboxes`
            _det_bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            _bboxes.append(_det_bboxes)

        mask_rois = bbox2roi(_bboxes)
        mask_feats = self.mask_roi_extractor(
            mask_roi_feats[:len(self.mask_roi_extractor.featmap_strides)],
            mask_rois)
        mask_pred = self.mask_head(mask_feats)
        segm_results = []
        for img_id, meta in enumerate(img_meta):
            idx_img = mask_rois[:, 0] == img_id
            mask_pred_img = mask_pred[idx_img]
            _bboxes_img = _bboxes[img_id]
            det_bboxes_img, det_labels_img = det_results[img_id]
            ori_shape = meta['ori_shape']
            scale_factor = meta['scale_factor']
            segm_result = self.mask_head.get_seg_masks(
                mask_pred_img, _bboxes_img, det_labels_img,
                self.test_cfg.single_stage_mask,
                ori_shape, scale_factor, rescale)
            segm_results.append(segm_result)

        # stuff subnet
        stuff_head_inputs = bbox_outs[-2]

        stuff_outs = self.stuff_head(stuff_head_inputs)
        stuff_results = self.stuff_head.get_stuff_map(
            stuff_outs, img_meta, rescale=rescale)

        bbox_segm_stuff_results = []
        for bbox_result, segm_result, stuff_result in zip(
                bbox_results, segm_results, stuff_results):
            bbox_segm_stuff_results.append(
                (bbox_result, segm_result, stuff_result))
        return bbox_segm_stuff_results[0]


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentation."""
        single_stage_cfg = self.test_cfg.single_stage
        imgs_per_gpu = len(img_metas[0])
        aug_bboxes = [[] for _ in range(imgs_per_gpu)]
        aug_scores = [[] for _ in range(imgs_per_gpu)]
        mask_roi_feats = []
        stuff_outs = []
        for x, img_meta in zip(self.extract_feats(imgs), img_metas):
            bbox_outs = self.bbox_head(x)
            bbox_inputs = bbox_outs[:2] + (img_meta, single_stage_cfg, False)
            results_list = self.bbox_head.get_bboxes(*bbox_inputs)

            # mask subnet
            mask_roi_feats.append(bbox_outs[-1])

            # stuff subnet
            stuff_head_inputs = bbox_outs[-2]
            stuff_out = self.stuff_head(stuff_head_inputs)
            stuff_outs.append(stuff_out)

            for i, results in enumerate(results_list):
                mlvl_bboxes, mlvl_scores = results
                aug_bboxes[i].append(mlvl_bboxes)
                aug_scores[i].append(mlvl_scores)

        aug_img_metas = []
        for i in range(imgs_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)

        det_results = []
        for aug_bbox, aug_score, aug_img_meta in zip(
                aug_bboxes, aug_scores, aug_img_metas):
            merged_bboxes, merged_scores = merge_aug_bboxes(
                aug_bbox, aug_score, aug_img_meta, single_stage_cfg,
                return_mean=False)
            det_bboxes, det_labels = multiclass_nms(
                merged_bboxes, merged_scores, single_stage_cfg.score_thr,
                single_stage_cfg.nms, single_stage_cfg.max_per_img)
            det_results.append((det_bboxes, det_labels))

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in det_results
        ]


        # mask
        imgs_per_gpu = len(img_metas[0])
        aug_masks = [[] for _ in range(imgs_per_gpu)]
        for x, img_meta in zip(mask_roi_feats, img_metas):
            # we should rescale the det bboxes outside `simple_test_mask`
            # because there are `flip` for aug test setting, while in the
            # simple test setting, just have the scale
            scaled_det_results = []
            for results, meta in zip(det_results, img_meta):
                det_bboxes, _ = results
                img_shape = meta['img_shape']
                scale_factor = meta['scale_factor']
                flip = meta['flip']
                scaled_det_bboxes = bbox_mapping(det_bboxes[:, :4],
                                                 img_shape, scale_factor, flip)
                scaled_det_results.append(scaled_det_bboxes)
            mask_rois = bbox2roi(scaled_det_results)
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)],
                mask_rois)
            mask_pred = self.mask_head(mask_feats)
            for img_id, meta in enumerate(img_meta):
                idx_img = mask_rois[:, 0] == img_id
                mask_pred_img = mask_pred[idx_img]
                # convert to numpy array to save memory
                mask_pred_img_np = mask_pred_img.sigmoid().cpu().numpy()
                aug_masks[img_id].append(mask_pred_img_np)

        segm_results = []
        for aug_mask, aug_img_meta in zip(aug_masks, aug_img_metas):
            merged_masks = merge_aug_masks(aug_mask, aug_img_meta,
                                           self.test_cfg.single_stage_mask)
            segm_results.append(merged_masks)

        # stuff
        stuff_results = self.stuff_head.get_stuff_map_aug(
            stuff_outs, img_metas, rescale=rescale)

        bbox_segm_stuff_results = []
        for bbox_result, segm_result, stuff_result in zip(
                bbox_results, segm_results, stuff_results):
            bbox_segm_stuff_results.append(
                (bbox_result, segm_result, stuff_result))
        return bbox_segm_stuff_results[0]
