_base_ = './spatialflow_r50_fpn_20e_coco.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
# panoptic settings
segmentations_folder='./work_dirs/spatialflow_r101_fpn_20e_coco/segmentations_folder_val_pred/'
panoptic_json_file='./work_dirs/spatialflow_r101_fpn_20e_coco/panoptic_val_pred.json'
