_base_ = [
    '../_base_/models/spatialflow_r50_fpn.py',
    '../_base_/datasets/coco_panoptic.py',
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# panoptic settings
segmentations_folder='./work_dirs/spatialflow_r50_fpn_20e_coco/segmentations_folder_val_pred/'
panoptic_json_file='./work_dirs/spatialflow_r50_fpn_20e_coco/panoptic_val_pred.json'
