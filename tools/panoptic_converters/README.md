# Converters for panoptic segmentation.

## Cityscapes

For cityscape, we only need to convert the val split to panoptic format,
which is used for evaluating.Because we convert the train split to COCO style in `convert_cityscape_instance.py`,
and we do not have the ground truth of test split.

To convert cityscape val split:
```shell
# current path: root path of SpatialFlow.
python ./tools/panoptic_converters/cityscape_panoptic_converter.py
```

## COCO

Convert panoptic annotation on train and val split.

* convert the panoptic annotation to detection annotation
```shell
# current path: root path of SpatialFlow.
python ./tools/panoptic_converters/panoptic2detection_coco_format.py
--input_json_file /data/coco2017/annotations/panoptic_val(train)2017.json
--output_json_file /data/coco2017/annotations/panoptic_val(train)2017_detection_format(_things_only).json
--categories_json_file ../panoptic_coco_categories.json (--things_only)
```

* convert the panoptic annotation to semantic segmentation annotation (pngs)
```shell
# current path: root path of SpatialFlow.
python ./tools/panoptic_converters/panoptic2semantic_segmentation.py
--input_json_file /data/coco2017/annotations/panoptic_val(train)2017.json
--semantic_seg_folder /data/coco2017/annotations/panoptic_val(train)2017_semantic_segmentation(_things_other)_pngs
--categories_json_file ../panoptic_coco_categories.json (--things_other)
```
