"""
Usage:

python evaluation_neg.py \
    --prediction_file <path_to_prediction_file> \
    --annotation_file <path_to_annotation_file> \
    --save_dir <output_directory> \
    --score_type <score_type>

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

import re
import json

from torchvision.ops import box_iou
from pycocotools.coco import COCO
import sklearn.metrics as sk

# ------------------ Metrics Calculation ------------------
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

    y_true = (y_true == pos_label)
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]
    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
    cutoff = np.argmin(np.abs(recall - recall_level))
    return fps[cutoff] / (np.sum(np.logical_not(y_true)))


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """High precision cumsum with stability check."""
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: its last element does not correspond to sum')
    return out

# ------------------ Score Grouping Utilities ------------------
def save_scores_by_cate(data, cate):
    classified = defaultdict(list)
    classified_id_list = defaultdict(list)
    for item in data:
        score = item['score']
        key = tuple((item[name] for name in cate))
        classified[key].append(score)
        classified_id_list[key].append(item['pos_id'])
    sorted_keys = sorted(classified.keys(), key=lambda x: tuple(x[i] for i in range(len(key))))
    sorted_final_dict = {key: classified[key] for key in sorted_keys}
    return sorted_final_dict, classified_id_list

def save_scores_by_cate_and_pos_id(data, cate):
    classified = defaultdict(dict)
    for item in data:
        pos_id = item['pos_id']
        key = tuple((item[name] for name in cate))
        classified[key].setdefault(pos_id, []).append(item['score'])
    sorted_keys = sorted(classified.keys(), key=lambda x: tuple(x[i] for i in range(len(key))))
    sorted_final_dict = {key: classified[key] for key in sorted_keys}
    return sorted_final_dict

# ------------------ BBox Utilities ------------------
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
            box_list = list(map(int, box.split(',')))
            try:
                resized_box_list = resize_bbox(box_list, img_w, img_h)
            except Exception:
                continue
            entities.append(entity)
            boxes.append(resized_box_list)
            while start < len(text) and text[start] not in ['.', ',', ';', '!', '?']:
                start += 1
            start += 1
    return entities, boxes

def find_top_k_indices(pos_score_list, negative_score_list, k):
    combined_scores = negative_score_list + pos_score_list
    combined_scores.sort(reverse=True)
    top_k_scores = combined_scores[:k]
    indices = [idx for idx, score in enumerate(pos_score_list) if score in top_k_scores]
    return indices

def are_phrases_similar(phrase1, phrase2):
    phrase1 = phrase1.lower()
    phrase2 = phrase2.lower()
    phrase1 = re.sub(r'\s*([\'",.;!?|:])\s*', r'\1 ', phrase1).strip()
    phrase2 = re.sub(r'\s*([\'",.;!?|:])\s*', r'\1 ', phrase2).strip()
    phrase1 = re.sub(r'[^\w\s]', '', phrase1)
    phrase2 = re.sub(r'[^\w\s]', '', phrase2)
    phrase1 = ' '.join(phrase1.split())
    phrase2 = ' '.join(phrase2.split())
    return phrase1 == phrase2

def extract_bbox(string):
    pattern = r'\[([\d.,]+)\]'
    match = re.search(pattern, string)
    if match:
        extracted_list = [float(num) for num in match.group(1).split(',')]
        return extracted_list
    return []

def extract_first_option(text):
    match = re.search(r'\b[A-F]\b', text)
    if match:
        return match.group()
    return None

# ------------------ Main Evaluator Class ------------------
class RefExpEvaluatorFromJsonl(object):
    def __init__(self, refexp_gt_path, k=(1, -1), topk=[1, 5, 10], thresh_iou=0.5, scores=0, save_dir=None, save_failure=True,
                 score_type='scores_prob_average'):
        assert isinstance(k, (list, tuple))
        self.k = k
        self.topk = topk
        self.thresh_iou = thresh_iou
        self.scores = scores
        self.coco = COCO(refexp_gt_path)
        self.save_dir = save_dir
        self.save_failure = save_failure
        self.score_type = score_type
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def summarize(self, prediction_file: str, verbose: bool = False):
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
            'neg': {k: 0.0 for k in self.k},
        }
        dataset2count = {'ovr': 0.0, 1: 0.0, 2: 0.0, 'neg': 0.0}
        in_score = []
        out_score = []
        in_score_level = dict()
        negative_data_list = []
        results_id_to_iou = dict()
        results_id_to_score = dict()
        count_tmp = 0
        cnt = 0
        # Please provide your own ground truth file if needed
        # gt_data = json.load(open('path_to_gt_file.json', 'r'))
        for prediction in predictions:
            img_id = prediction['image_id']
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            assert len(ann_ids) == 1
            img_info = self.coco.loadImgs(img_id)[0]
            target = self.coco.loadAnns(ann_ids[0])[0]
            target_bbox = target['bbox']
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            target_bbox = torch.tensor(converted_bbox).unsqueeze(0)
            level = target['level']
            if level == 3:
                level = 2
            answer = prediction['answer']
            gt = prediction['gt']
            option = answer[0].replace(".", "")
            if len(option) > 1:
                option = extract_first_option(option)
            predict_boxes = gt[option]
            pred_scores = np.exp(prediction[self.score_type])
            if len(predict_boxes) == 0:
                predict_boxes = [[0., 0., 0., 0.]]
                pred_scores = np.array([0])
            score = pred_scores
            if target.get('negative_type', None) is None:
                if predict_boxes == 'None':
                    iou = torch.tensor([[0]])
                    count_tmp += 1
                else:
                    iou = box_iou(torch.tensor(predict_boxes).unsqueeze(0), target_bbox)
                for k in self.k:
                    if iou.reshape(-1)[0] >= self.thresh_iou:
                        dataset2score[level][k] += 1.0
                        dataset2score['ovr'][k] += 1.0
                dataset2count[level] += 1.0
                dataset2count['ovr'] += 1.0
                in_score_level.setdefault(level, []).append(score)
                in_score.append(score)
                results_id_to_score[str(img_id)] = score
                results_id_to_iou[str(img_id)] = iou
            else:
                dataset2count['neg'] += 1.0
                if predict_boxes == 'None':
                    score = 1 - score
                    dataset2score['neg'][1] += 1.0
                else:
                    cnt += 1
                out_score.append(score)
                negative_level = target['negative_level']
                negative_data = {"negative_level": negative_level, "positive_level": level,
                                 "negative_type": target['negative_type'], "tuple_type": target['tuple_type'],
                                 "score": score, "pos_id": str(target['positive_id']),
                                 "negative_cate": target['negative_cate']}
                negative_data_list.append(negative_data)

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
            print(f" Dataset: {key} - Precision @ 1, all: {save_results[key]} \n")
        precision_df = pd.DataFrame(save_results).T
        precision_df.columns = ["Precision @ 1"]
        precision_df.insert(loc=0, column='level', value=['ovr', 1, 2, 3])
        if self.save_dir is not None:
            precision_df.to_csv(os.path.join(self.save_dir, f'precision.csv'), index=False)
        recall_dict = {}
        cate = ["negative_type", "negative_level", "negative_cate"]
        negative_score_dict = save_scores_by_cate_and_pos_id(negative_data_list, cate)
        for neg_type, neg_score_dict in negative_score_dict.items():
            recall_count = 0
            recall = {k: 0.0 for k in self.topk}
            for pos_id, neg_score_list in neg_score_dict.items():
                pos_score_list = [results_id_to_score[str(pos_id)]]
                iou = results_id_to_iou[str(pos_id)]
                for k in self.topk:
                    indices = find_top_k_indices(pos_score_list, neg_score_list, k)
                    max_index = min(len(indices), k)
                    if max_index > 0 and max(iou[:max_index]) >= self.thresh_iou:
                        recall[k] += 1.0
                recall_count += 1.0
            save_recall = {}
            for key, value in recall.items():
                save_recall[key] = round(value / recall_count * 100, 2)
            recall_dict[neg_type] = list(save_recall.values()) + [len(neg_score_dict)]
        recall_df = pd.DataFrame(recall_dict).T
        recall_df.reset_index(inplace=True)
        recall_df.columns = ['type', 'level', 'cate', "Recall @ 1", "count"]
        recall_df.sort_values(by=['cate', 'type', 'level'], ascending=[False, True, True], inplace=True)
        recall_df['count'] = recall_df['count'].astype(int)
        if self.save_dir is not None:
            recall_df.to_csv(os.path.join(self.save_dir, f'recall_{cate}.csv'), index=False)
        auroc, aupr, fpr = get_measures(in_score_level[1], out_score)
        count_negative = dict()
        overall_dict = {tuple(['overall', 0, 0]): [round(auroc * 100, 2), round(fpr * 100, 2), round(100 * aupr, 2), len(out_score)]}
        result_dict = dict()
        final_dict, pos_id_dict = save_scores_by_cate(negative_data_list, cate)
        for types, score in final_dict.items():
            count_negative[str(dict(zip(cate, types)))] = len(score)
            pos_id_list = pos_id_dict[types]
            in_score_tmp = [results_id_to_score[str(idx)] for idx in pos_id_list]
            auroc, aupr, fpr = get_measures(in_score_tmp, score)
            result_dict[types] = [round(auroc * 100, 2), round(fpr * 100, 2), round(100 * aupr, 2), len(score)]
        result_dict.update(overall_dict)
        auroc_df = pd.DataFrame(result_dict).T
        auroc_df.reset_index(inplace=True)
        auroc_df.columns = ['type', 'level', 'cate', "AUROC", "FPR95", "AUPR", "count"]
        auroc_df['count'] = auroc_df['count'].astype(int)
        auroc_df.sort_values(by=['cate', 'type', 'level'], ascending=[False, True, True], inplace=True)
        if self.save_dir is not None:
            auroc_df.to_csv(os.path.join(self.save_dir, f'auroc_{cate}.csv'), index=False)
        return precision_df, recall_df, auroc_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate negative samples.')
    parser.add_argument('--prediction_file', required=True, help='Path to the prediction file (jsonl or directory).')
    parser.add_argument('--annotation_file', required=True, help='Path to the annotation file (COCO format).')
    parser.add_argument('--save_dir', default='./eval_results', type=str, help='Directory to save evaluation results.')
    parser.add_argument('--score_type', default='answer_prob_norm', type=str, choices=['prob', 'answer_prob', 'none_prob', 'answer_prob_norm'], help='Score type to use.')
    args = parser.parse_args()

    evaluator = RefExpEvaluatorFromJsonl(
        refexp_gt_path=args.annotation_file,
        k=(1,),
        topk=[1],
        thresh_iou=0.5,
        save_dir=args.save_dir,
        score_type=args.score_type
    )
    evaluator.summarize(args.prediction_file, verbose=False)
