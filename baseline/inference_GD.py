import ast
import json
import os
from argparse import ArgumentParser
from tqdm import tqdm
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes
from pycocotools.coco import COCO
import torch


def find_positive(expression, subject):
    """
    Find the start and end indices of the subject in the expression (case-insensitive).
    Returns [start, end] or None if not found.
    """
    exp_lower = expression.lower()
    subject_lower = subject.lower()
    start_index = exp_lower.find(subject_lower)
    if start_index == -1:
        return None
    else:
        end_index = start_index + len(subject)
        return [start_index, end_index]

def run_inference(args):
    """
    Main inference and evaluation logic for GroundingDINO.
    """
    # Prepare model args
    call_args = vars(args).copy()
    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {k: call_args.pop(k) for k in init_kws}

    # Handle special arguments
    if call_args.get('no_save_vis') and call_args.get('no_save_pred'):
        call_args['out_dir'] = ''
    if init_args['model'] and init_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically assign the model to --weights')
        init_args['weights'] = init_args['model']
        init_args['model'] = None
    if call_args.get('texts') is not None and str(call_args['texts']).startswith('$:'):
        dataset_name = call_args['texts'][3:].strip()
        class_names = get_classes(dataset_name)
        call_args['texts'] = [tuple(class_names)]

    # Load data
    with open(args.subjects_file, 'r') as f:
        subjects = json.load(f)
    coco = COCO(args.coco_anno)
    out_dict = {}
    error_list = []

    inferencer = DetInferencer(**init_args)
    chunked_size = call_args.pop('chunked_size')
    inferencer.model.test_cfg.chunked_size = chunked_size
    cnt = 0

    for key, subject in tqdm(subjects.items()):
        img_info = coco.loadImgs(int(key))[0]
        call_args['inputs'] = os.path.join(args.img_root, img_info['file_name'])
        # Use the caption as expression if available, otherwise use subject
        exp = img_info.get('caption', subject)
        if call_args.get('focus_enhancement'):
            tokens_positive = find_positive(exp, subject)
            call_args['tokens_positive'] = [[tokens_positive]]
        else:
            call_args['tokens_positive'] = '-1'
        call_args['texts'] = exp
        try:
            return_dict = inferencer(**call_args)
        except RuntimeError as e:
            print(e)
            error_list.append(key)
            continue
        out_dict[key] = {
            'scores': return_dict['predictions'][0]['scores'][0],
            'bbox': return_dict['predictions'][0]['bboxes'][0],
        }
    with open(args.output_file, 'w') as f:
        json.dump(out_dict, f)
    print(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    def parse_args():
        parser = ArgumentParser(description="GroundingDINO inference and evaluation script.")
        parser.add_argument('--subjects-file', type=str, required=True, help='Path to the subjects JSON file (COCO image id to text mapping).')
        parser.add_argument('--coco-anno', type=str, required=True, help='Path to the COCO-format annotation file.')
        parser.add_argument('--img-root', type=str, required=True, help='Root directory of images.')
        parser.add_argument('--output-file', type=str, default='outputs/tokens_positive.json', help='Path to save the output JSON file.')
        parser.add_argument('--model', type=str, default='configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py', help='Config or checkpoint .pth file or the model name and alias defined in metafile.')
        parser.add_argument('--weights', default='grounding_dino_swin-l_pretrain_all-56d69e78.pth', help='Checkpoint file')
        parser.add_argument('--device', default='cuda:0', help='Device used for inference')
        parser.add_argument('--palette', default='none', choices=['coco', 'voc', 'citys', 'random', 'none'], help='Color palette used for visualization')
        parser.add_argument('--chunked-size', '-s', type=int, default=-1, help='Truncate multiple predictions if categories are large.')
        parser.add_argument('--texts', help='Text prompt, such as "bench . car .", "$: coco"')
        parser.add_argument('--pred-score-thr', type=float, default=0.25, help='bbox score threshold')
        parser.add_argument('--batch-size', type=int, default=1, help='Inference batch size.')
        parser.add_argument('--show', action='store_true', help='Display the image in a popup window.')
        parser.add_argument('--no-save-vis', action='store_true', help='Do not save detection vis results')
        parser.add_argument('--no-save-pred', action='store_true', help='Do not save detection json results')
        parser.add_argument('--print-result', action='store_true', help='Whether to print the results.')
        parser.add_argument('--custom-entities', '-c', action='store_true', help='Whether to customize entity names?')
        parser.add_argument('--focus-enhancement', action='store_true', help='Implement the Focus-enhancement Strategy.')
        args = parser.parse_args()
        return args
    args = parse_args()
    run_inference(args)
