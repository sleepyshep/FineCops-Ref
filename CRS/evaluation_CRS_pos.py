"""
Usage Example:

python evaluation_pos.py \
    --prediction_file <path_to_prediction_jsonl> \
    --annotation_file <path_to_annotation_json> \
    --save_dir <output_dir> \
    --groundingdino_file <path_to_groundingdino_json>

All file paths should be set according to your environment.
"""

import ast
import os
import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.utils.data
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
# from prettytable import PrettyTable

import re
import json

# from misc.refcoco.box_ops import generalized_box_iou, box_iou
from torchvision.ops import box_iou
from pycocotools.coco import COCO


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def save_scores_by_cate(data, cate):
    classified = defaultdict(list)
    classified_id_list = defaultdict(list)
    # cate_id_map = {key: idx for idx, key in
    #                enumerate(["negative_level", "positive_level", "negative_type", "tuple_type", "score", 'pos_id'])}
    for item in data:
        try:
            score = max(item['score'])
        except Exception as e:
            print(e)
            print('no score > 0.3')
            continue
        key = tuple((item[name] for name in cate))
        classified[key].append(score)
        classified_id_list[key].append(item['pos_id'])
    sorted_keys = sorted(classified.keys(), key=lambda x: tuple(x[i] for i in range(len(key))))
    sorted_final_dict = {key: classified[key] for key in sorted_keys}
    return sorted_final_dict, classified_id_list


def save_scores_by_cate_and_pos_id(data, cate):
    classified = defaultdict(dict)
    # classified_id_list = defaultdict(list)
    # cate_id_map = {key: idx for idx, key in
    #                enumerate(["negative_level", "positive_level", "negative_type", "tuple_type", "score", 'pos_id'])}
    for item in data:
        pos_id = item['pos_id']
        key = tuple((item[name] for name in cate))
        classified[key].setdefault(pos_id, []).extend(item['score'])
        # classified_id_list[key].append(item[cate_id_map['pos_id']])
    sorted_keys = sorted(classified.keys(), key=lambda x: tuple(x[i] for i in range(len(key))))
    sorted_final_dict = {key: classified[key] for key in sorted_keys}
    return sorted_final_dict


VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000


def resize_bbox(box, image_w=None, image_h=None):
    ratio_w = image_w * 1.0 / VOCAB_IMAGE_W
    ratio_h = image_h * 1.0 / VOCAB_IMAGE_H

    new_box = [int(box[0] * ratio_w), int(box[1] * ratio_h),
               int(box[2] * ratio_w), int(box[3] * ratio_h)]
    return new_box


def decode_bbox_from_caption(text, img_w, img_h, verbose=False):
    entities = []
    boxes = []

    start = 0
    in_brackets = False
    entity = ""
    box = ""

    for i, char in enumerate(text):
        if char == '[':
            in_brackets = True
            entity = text[start:i].strip()
            start = i + 1
        elif char == ']':
            in_brackets = False
            box = text[start:i].strip()
            start = i + 1

            # Convert box string to list of integers
            box_list = list(map(int, box.split(',')))
            try:
                resized_box_list = resize_bbox(box_list, img_w, img_h)
            except Exception as e:
                print(f'unexpected box: {box_list}')
                continue
            entities.append(entity)
            boxes.append(resized_box_list)

            # Skip until the next entity (ignoring periods or other delimiters)
            while start < len(text) and text[start] not in ['.', ',', ';', '!', '?']:
                start += 1
            start += 1  # Skip the delimiter

    return entities, boxes


def find_top_k_indices(pos_score_list, negative_score_list, k):
    combined_scores = negative_score_list + pos_score_list
    combined_scores.sort(reverse=True)

    # 获取前k个值
    top_k_scores = combined_scores[:k]

    # 找到这些值在pos_score_list中的索引
    indices = [idx for idx, score in enumerate(pos_score_list) if score in top_k_scores]

    return indices


def are_phrases_similar(phrase1, phrase2):
    # Step 1: Convert to lower case
    phrase1 = phrase1.lower()
    phrase2 = phrase2.lower()

    # Step 2: Standardize spacing around punctuation
    phrase1 = re.sub(r'\s*([\'",.;!?|:])\s*', r'\1 ', phrase1).strip()
    phrase2 = re.sub(r'\s*([\'",.;!?|:])\s*', r'\1 ', phrase2).strip()

    # Step 3: Remove all punctuation
    phrase1 = re.sub(r'[^\w\s]', '', phrase1)
    phrase2 = re.sub(r'[^\w\s]', '', phrase2)

    # Step 4: Remove extra white spaces
    phrase1 = ' '.join(phrase1.split())
    phrase2 = ' '.join(phrase2.split())

    return phrase1 == phrase2

def extract_bbox(string):
    import re

    # 使用正则表达式提取列表
    pattern = r'\[([\d.,]+)\]'
    match = re.search(pattern, string)
    if match:
        extracted_list = [float(num) for num in match.group(1).split(',')]
        return extracted_list
    return []

def extract_option(answer):
    """
    从字符串中提取独立的选项 A、B、C 或 D。如果不存在则返回 None。
    从后向前匹配，返回第一个匹配到的选项。

    :param answer: 输入字符串
    :return: 提取到的选项（'A', 'B', 'C', 'D'）或 None
    """
    # 使用 finditer 获取所有匹配项
    matches = list(re.finditer(r'\b([A-D])\b', answer))
    
    # 从后向前遍历匹配项
    for match in reversed(matches):
        return match.group(1).upper()
    
    # 如果没有匹配到任何选项，返回 None
    return None

def option_dict(prompt):
    # 使用正则表达式提取选项
    pattern = r'- ([A-D]): \[\[(\d+,\d+,\d+,\d+)\]\]'
    matches = re.findall(pattern, prompt)

    # 将提取的结果转换为字典
    options = {key: f'[[{value}]]' for key, value in matches}
    return options

def find_matching_option(str2, options, option):
    # 从字符串2中提取bounding box
    start = str2.find('[')
    end = str2.rfind(']') + 1
    bounding_box = str2[start:end]
    if option:
        if options[option] == bounding_box:
            return option

    # Iterate through the options to find a match
    for option, box in options.items():
        if str(box) == bounding_box:
            return option
    
    # Return None if no match is found
    return None

# Main evaluation class
class RefExpEvaluatorFromJsonl(object):
    def __init__(self, refexp_gt_path, k=(1, -1), thresh_iou=0.5, score_thrs=0, save_dir=None, save_failure=True,
                 score_type='scores_prob_average', groundingdino_file=None):
        assert isinstance(k, (list, tuple))
        self.k = k
        self.topk = [1, 5, 10]
        self.thresh_iou = thresh_iou
        self.score_thrs = score_thrs
        self.coco = COCO(refexp_gt_path)
        self.save_dir = save_dir
        self.save_failure = save_failure
        self.score_type = score_type
        self.groundingdino_file = groundingdino_file
        os.makedirs(save_dir, exist_ok=True)

    def summarize(self, prediction_file: str, verbose: bool = False):
        # get the predictions
        if os.path.isfile(prediction_file):
            predictions = [json.loads(line) for line in open(prediction_file)]
        elif os.path.isdir(prediction_file):
            predictions = [json.loads(line) for pred_file in os.listdir(prediction_file) for line in open(os.path.join(prediction_file, pred_file))]
        else:
            raise NotImplementedError('Not supported file format.')

        dataset2score = {
            'ovr': {k: 0.0 for k in self.k},
            1: {k: 0.0 for k in self.k},
            2: {k: 0.0 for k in self.k},
        }
        dataset2count = {'ovr': 0.0, 1: 0.0, 2: 0.0}
        out_dict = {}
        in_score = []
        in_score_level = dict()
        results_id_to_iou = dict()
        results_id_to_score = dict()
        failure_list = []
        count_tmp = 0

        # Load groundingdino results
        if self.groundingdino_file is None:
            raise ValueError('groundingdino_file must be provided as a parameter.')
        with open(self.groundingdino_file, 'r') as f:
            groundingdino = json.load(f)

        for prediction in predictions:
            img_id = prediction['image_id']
            ann_ids = self.coco.getAnnIds(imgIds=int(img_id))
            if len(ann_ids) != 1:
                continue  # skip if annotation is not unique
            img_info = self.coco.loadImgs(int(img_id))[0]
            target = self.coco.loadAnns(ann_ids[0])
            img_height = img_info['height']
            img_width = img_info['width']
            caption = img_info.get('caption', '')
            target_bbox = target[0]['bbox']
            converted_bbox = [target_bbox[0], target_bbox[1], target_bbox[2] + target_bbox[0], target_bbox[3] + target_bbox[1]]
            target_bbox = torch.as_tensor(converted_bbox).view(-1, 4)

            option = prediction['answer'][0].upper()
            if len(option) != 1:
                option = extract_option(option)
            if target[0].get('level', 1) == 3:
                target[0]['level'] = 2
            predict_boxes = []
            gd_box = groundingdino.get(str(img_id), {}).get('bbox', [])
            if option == 'A' and len(gd_box) > 0:
                box = gd_box[0]
            elif option == 'B' and len(gd_box) > 1:
                box = gd_box[1]
            elif option == 'C' and len(gd_box) > 2:
                box = gd_box[2]
            elif option == 'D' and len(gd_box) > 3:
                box = gd_box[3]
            else:
                box = [0., 0., 0., 0.]
            predict_boxes.append(box)
            pred_scores = np.array([0])
            if len(predict_boxes) == 0:
                predict_boxes = [[0., 0., 0., 0.]]
                pred_scores = np.array([0])
            sorted_indices = np.argsort(pred_scores)[::-1]
            predict_boxes = [predict_boxes[i] for i in sorted_indices]
            pred_scores = pred_scores[sorted_indices]
            predict_boxes = torch.as_tensor(predict_boxes).view(-1, 4).to(dtype=torch.float32)
            score_thr = pred_scores[pred_scores >= self.score_thrs]
            score = max(pred_scores)
            if target[0].get('negative_type', None) is None:
                predict_boxes = torch.tensor(predict_boxes)
                iou = box_iou(predict_boxes, target_bbox)
                mean_iou = box_iou(predict_boxes.mean(0).view(-1, 4), target_bbox)
                out_dict[img_id] = iou.tolist()[0][0]
                for k in self.k:
                    if k == 'upper bound':
                        if max(iou.reshape(-1)) >= self.thresh_iou:
                            dataset2score[target[0]['level']][k] += 1.0
                    elif k == 'mean':
                        if max(mean_iou) >= self.thresh_iou:
                            dataset2score[target[0]['level']][k] += 1.0
                    else:
                        if iou.reshape(-1)[0] >= self.thresh_iou:
                            dataset2score[target[0]['level']][k] += 1.0
                            dataset2score['ovr'][k] += 1.0
                        else:
                            failure_list.append([img_id, img_info.get('file_name', ''), predict_boxes[0].tolist()])
                dataset2count[target[0]['level']] += 1.0
                dataset2count['ovr'] += 1.0
                in_score_level.setdefault(target[0]['level'], []).append(score)
                in_score.append(score)
                results_id_to_score[str(img_id)] = score_thr
                results_id_to_iou[str(img_id)] = iou
        print(f"Count positive: {dataset2count}")
        for key, value in dataset2score.items():
            for k in self.k:
                try:
                    value[k] /= dataset2count[key]
                except Exception:
                    pass
        save_results = {}
        for key, value in dataset2score.items():
            save_results[key] = sorted([round(v * 100, 2) for k, v in value.items()])
            print(f"Dataset: {key}: {save_results[key][0]}")
        # Optionally, save results to file
        if self.save_dir:
            with open(os.path.join(self.save_dir, 'eval_results.json'), 'w') as f:
                json.dump(save_results, f, indent=2)
        # Optionally, save failure cases
        if self.save_failure and self.save_dir:
            with open(os.path.join(self.save_dir, 'failure_list.json'), 'w') as f:
                json.dump(failure_list, f, indent=2)
        # Optionally, warn if output count mismatches expected
        # print(f"Total evaluated: {len(out_dict)}")
        return save_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate positive samples.')
    parser.add_argument('--prediction_file', required=True, help='Path to the prediction file (jsonl or directory).')
    parser.add_argument('--annotation_file', required=True, help='Path to the annotation file (COCO format).')
    parser.add_argument('--save_dir', default='./eval_results', type=str, help='Directory to save evaluation results.')
    parser.add_argument('--groundingdino_file', required=True, help='Path to groundingdino top-k bbox json file')
    args = parser.parse_args()

    evaluator = RefExpEvaluatorFromJsonl(
        refexp_gt_path=args.annotation_file,
        k=(1, ),
        thresh_iou=0.5,
        save_dir=args.save_dir,
        groundingdino_file=args.groundingdino_file
    )
    evaluator.summarize(args.prediction_file, verbose=False)