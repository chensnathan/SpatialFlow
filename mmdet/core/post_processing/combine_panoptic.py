"""The script uses a simple procedure to combine semantic segmentation and
instance
segmentation predictions. The procedure is described in section 7 of the
panoptic segmentation paper https://arxiv.org/pdf/1801.00868.pdf.

On top of the procedure described in the paper. This script remove from
prediction small segments of stuff semantic classes. This addition allows to
decrease number of false positives.
"""
from __future__ import (print_function, absolute_import, division,
                        unicode_literals)

from collections import defaultdict
import copy
import os
import os.path as osp

import argparse
import mmcv
import multiprocessing
import numpy as np
import PIL.Image as Image
from pycocotools import mask as COCOmask
import time

from ..utils import id2rgb, IdGenerator, MyJsonEncoder


def combine_to_panoptic(proc_id, img_ids, img_id2img, inst_by_image,
                        sem_by_image, bbox_by_image, segmentations_folder,
                        bbox_overlap_thr, overlap_thr, stuff_area_limit,
                        categories, using_bbox=False):
    panoptic_json = []
    id_generator = IdGenerator(categories)

    for idx, img_id in enumerate(img_ids):
        img = img_id2img[img_id]

        if idx % 100 == 0:
            print('Core: {}, {} from {} images processed.'.
                  format(proc_id, idx, len(img_ids)))

        pan_segm_id = np.zeros((img['height'], img['width']),
                               dtype=np.uint32)
        used = None
        annotation = {}
        annotation['image_id'] = img_id
        annotation['file_name'] = img['file_name'].replace('.jpg', '.png')

        segments_info = []
        for ann_id, ann in enumerate(inst_by_image[img_id]):
            area = COCOmask.area(ann['segmentation'])
            if area == 0:
                continue
            if used is None:
                intersect = 0
                used = copy.deepcopy(ann['segmentation'])
            else:
                intersect = COCOmask.area(
                    COCOmask.merge([used, ann['segmentation']], intersect=True)
                )
            if intersect / area > overlap_thr:
                continue
            used = COCOmask.merge([used, ann['segmentation']], intersect=False)

            mask = COCOmask.decode(ann['segmentation']) == 1
            if intersect != 0:
                mask = np.logical_and(pan_segm_id == 0, mask)
            segment_id = id_generator.get_id(ann['category_id'])
            panoptic_ann = {}
            panoptic_ann['id'] = segment_id
            panoptic_ann['category_id'] = ann['category_id']
            pan_segm_id[mask] = segment_id
            if using_bbox:
                bbox_by_image[img_id][ann_id].update(
                    dict(segment_id=segment_id))
            segments_info.append(panoptic_ann)

        for ann in sem_by_image[img_id]:
            mask = COCOmask.decode(ann['segmentation']) == 1
            mask_left = np.logical_and(pan_segm_id == 0, mask)
            if mask_left.sum() < stuff_area_limit:
                continue
            segment_id = id_generator.get_id(ann['category_id'])
            panoptic_ann = {}
            panoptic_ann['id'] = segment_id
            panoptic_ann['category_id'] = ann['category_id']
            pan_segm_id[mask_left] = segment_id
            segments_info.append(panoptic_ann)

        if using_bbox:
            for ann in bbox_by_image[img_id]:
                bbox = [int(ann['bbox'][0]), int(ann['bbox'][1]),
                        int(ann['bbox'][0] + ann['bbox'][2]),
                        int(ann['bbox'][1] + ann['bbox'][3])]
                mask = np.zeros((img['height'], img['width']), dtype=np.uint8)
                mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
                mask_left = np.logical_and(pan_segm_id == 0, mask == 1)
                overlap = mask_left.sum() / mask.sum()
                if overlap < bbox_overlap_thr:
                    continue
                generated_id = id_generator.get_id(ann['category_id'])
                segment_id = ann.get('segment_id', generated_id)
                panoptic_ann = {}
                panoptic_ann['id'] = segment_id
                panoptic_ann['category_id'] = ann['category_id']
                pan_segm_id[mask_left] = segment_id
                segments_info.append(panoptic_ann)

        annotation['segments_info'] = segments_info
        panoptic_json.append(annotation)
        save_image_path = osp.join(
            segmentations_folder, annotation['file_name'])
        dir_name = osp.dirname(osp.abspath(save_image_path))
        mmcv.mkdir_or_exist(dir_name)

        Image.fromarray(id2rgb(pan_segm_id)).save(save_image_path)

    return panoptic_json


def combine_panoptic_predictions(semseg_json_file,
                                 instseg_json_file,
                                 bbox_json_file,
                                 images_json_file,
                                 categories_json_file,
                                 segmentations_folder,
                                 panoptic_json_file,
                                 confidence_thr,
                                 bbox_overlap_thr,
                                 overlap_thr,
                                 stuff_area_limit,
                                 using_bbox=False):
    start_time = time.time()

    sem_results = mmcv.load(semseg_json_file)
    inst_results = mmcv.load(instseg_json_file)
    bbox_results = mmcv.load(bbox_json_file)
    assert len(bbox_results) == len(inst_results)
    images_d = mmcv.load(images_json_file)
    img_id2img = {img['id']: img for img in images_d['images']}

    categories_list = mmcv.load(categories_json_file)
    categories = {el['id']: el for el in categories_list}

    if segmentations_folder is None:
        segmentations_folder = panoptic_json_file.rsplit('.', 1)[0]
    if not osp.isdir(segmentations_folder):
        print("Creating folder {} for panoptic segmentation PNGs".
              format(segmentations_folder))
        os.mkdir(segmentations_folder)

    print("Combining:")
    print("Semantic segmentation:")
    print("\tJSON file: {}".format(semseg_json_file))
    print("and")
    print("Instance segmentations:")
    print("\tJSON file: {}".format(instseg_json_file))
    print("into")
    print("Panoptic segmentations:")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(panoptic_json_file))
    print("List of images to combine is takes from {}".
          format(images_json_file))
    print('\n')

    inst_by_image = defaultdict(list)
    for inst in inst_results:
        if inst['score'] < confidence_thr:
            continue
        inst_by_image[inst['image_id']].append(inst)
    for img_id in inst_by_image.keys():
        inst_by_image[img_id] = sorted(inst_by_image[img_id],
                                       key=lambda el: -el['score'])
    bbox_by_image = defaultdict(list)
    for bbox in bbox_results:
        if bbox['score'] < confidence_thr:
            continue
        bbox_by_image[bbox['image_id']].append(bbox)
    for img_id in bbox_by_image.keys():
        assert len(bbox_by_image[img_id]) == len(inst_by_image[img_id])
        bbox_by_image[img_id] = sorted(bbox_by_image[img_id],
                                       key=lambda el: -el['score'])

    sem_by_image = defaultdict(list)
    for sem in sem_results:
        if (sem['category_id'] == 183) or \
                (categories[sem['category_id']]['isthing'] == 1):
            continue
        sem_by_image[sem['image_id']].append(sem)

    imgs_ids_all = list(img_id2img.keys())
    cpu_num = multiprocessing.cpu_count()
    img_ids_split = np.array_split(imgs_ids_all, cpu_num)
    print("Number of cores: {}, images per core: {}".
          format(cpu_num, len(img_ids_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, img_ids in enumerate(img_ids_split):
        p = workers.apply_async(combine_to_panoptic,
                                (proc_id, img_ids, img_id2img, inst_by_image,
                                 sem_by_image, bbox_by_image,
                                 segmentations_folder, bbox_overlap_thr,
                                 overlap_thr, stuff_area_limit, categories,
                                 using_bbox))
        processes.append(p)
    panoptic_json = []
    for p in processes:
        panoptic_json.extend(p.get())

    coco_d = mmcv.load(images_json_file)
    coco_d['annotations'] = panoptic_json
    coco_d['categories'] = list(categories.values())
    mmcv.dump(coco_d, panoptic_json_file, cls=MyJsonEncoder)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script uses a simple procedure to combine "
                    "semantic segmentation and instance segmentation "
                    "predictions. See this file's head for more "
                    "information.")
    parser.add_argument('--semseg_json_file', type=str,
                        help="JSON file with semantic segmentation "
                             "predictions")
    parser.add_argument('--instseg_json_file', type=str,
                        help="JSON file with instance segmentation "
                             "predictions")
    parser.add_argument('--bbox_json_file', type=str,
                        help="JSON file with object detection predictions")
    parser.add_argument('--images_json_file', type=str,
                        help="JSON file with correponding image set "
                             "information")
    parser.add_argument('--categories_json_file', type=str,
                        help="JSON file with Panoptic COCO categories "
                             "information",
                        default='./panoptic_coco_categories.json')
    parser.add_argument('--panoptic_json_file', type=str,
                        help="JSON file with resulting COCO panoptic "
                             "format prediction")
    parser.add_argument(
        '--segmentations_folder', type=str, default=None,
        help="Folder with panoptic COCO format segmentations. "
             "Default: X if panoptic_json_file is X.json")
    parser.add_argument('--confidence_thr', type=float, default=0.5,
                        help="Predicted segments with smaller "
                             "confidences than the threshold are "
                             "filtered out")
    parser.add_argument('--bbox_overlap_thr', type=float, default=0.5,
                        help="bboxes that have higher that the "
                             "threshold ratio of their area being "
                             "overlapped by segments with higher "
                             "confidence are filtered out")
    parser.add_argument('--overlap_thr', type=float, default=0.5,
                        help="Segments that have higher that the "
                             "threshold ratio of their area being "
                             "overlapped by segments with higher "
                             "confidence are filtered out")
    parser.add_argument('--stuff_area_limit', type=float, default=64 * 64,
                        help="Stuff segments with area smaller that the "
                             "limit are filtered out")
    parser.add_argument('--using_bbox', type=bool, default=False,
                        help="Whether to use bbox predictions.")
    args = parser.parse_args()
    combine_panoptic_predictions(args.semseg_json_file,
                                 args.instseg_json_file,
                                 args.bbox_json_file,
                                 args.images_json_file,
                                 args.categories_json_file,
                                 args.segmentations_folder,
                                 args.panoptic_json_file,
                                 args.confidence_thr,
                                 args.bbox_overlap_thr,
                                 args.overlap_thr,
                                 args.stuff_area_limit,
                                 args.using_bbox)
