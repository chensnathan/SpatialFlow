dataset_type = 'CocoDataset'
data_root = 'data/coco2017/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/panoptic_train2017_detection_format_things_only.json',
        img_prefix=data_root + 'train2017/',
        with_panoptic=True,
        things_other=True,
        pipeline=train_pipeline,
        seg_prefix=data_root +
                   'annotations/panoptic_train2017_semantic_segmentation_things_other_pngs/'
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/panoptic_val2017_detection_format_things_only.json',
        img_prefix=data_root + 'val2017/',
        with_panoptic=True,
        things_other=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/panoptic_val2017_detection_format_things_only.json',
        img_prefix=data_root + 'val2017/',
        with_panoptic=True,
        things_other=True,
        pipeline=test_pipeline))
evaluation = dict(metric=['panoptic'])
# panoptic settings
images_json_file=data_root + 'annotations/panoptic_val2017_detection_format_things_only.json'
categories_json_file=data_root + 'annotations/panoptic_coco_categories.json'
gt_json_file=data_root + 'annotations/panoptic_val2017.json'
gt_folder=data_root + 'annotations/panoptic_val2017/'
