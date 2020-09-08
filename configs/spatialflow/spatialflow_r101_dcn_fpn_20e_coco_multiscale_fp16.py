_base_ = './spatialflow_r101_fpn_20e_coco.py'
# fp16 settings
fp16 = dict(loss_scale=512.)
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    ),
    bbox_head=dict(
        dcn_cls_convs_idx=[0, 1, 2, 3],
        dcn_mask_convs_idx=[0],
        dcn_stuff_convs_idx=[0, 1, 2, 3],
    )
)
# data
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(
        type='Resize',
        img_scale=[(1600, 400), (1600, 1400)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1.),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1500, 1000), (1800, 1200), (2100, 1400)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# data_root = 'data/coco2017/'
data = dict(
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
    test=dict(
        # ann_file=data_root + 'annotations/instances_test_dev2017.json',
        # img_prefix=data_root + 'test2017/',
        pipeline=test_pipeline),
)
# images_json_file=data_root + 'annotations/instances_test_dev2017.json'
# panoptic settings
segmentations_folder='./work_dirs/spatialflow_r101_dcn_fpn_20e_coco_multiscale_fp16/segmentations_folder_val_pred/'
# segmentations_folder='./work_dirs/spatialflow_r101_dcn_fpn_20e_coco_multiscale_fp16/segmentations_folder_test_pred/'
panoptic_json_file='./work_dirs/spatialflow_r101_dcn_fpn_20e_coco_multiscale_fp16/panoptic_val_pred.json'
# panoptic_json_file='./work_dirs/spatialflow_r101_dcn_fpn_20e_coco_multiscale_fp16/panoptic_test_pred.json'
