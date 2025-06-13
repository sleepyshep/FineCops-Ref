import ast
import json
import os
from tqdm import tqdm
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes
from pycocotools.coco import COCO
import cv2

def run_detection(args):
    """
    Main detection and level assignment.
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
    if call_args.get('tokens_positive') is not None:
        call_args['tokens_positive'] = ast.literal_eval(call_args['tokens_positive'])

    # Load data
    with open(args.subjects_file, 'r') as f:
        subjects = json.load(f)
    coco = COCO(args.coco_anno)
    out_dict = {}
    error_list = []

    inferencer = DetInferencer(**init_args)
    chunked_size = call_args.pop('chunked_size')
    inferencer.model.test_cfg.chunked_size = chunked_size
    call_args['no_save_vis'] = True
    call_args['no_save_pred'] = True

    for key, subject in tqdm(subjects.items()):
        img_info = coco.loadImgs(int(key))[0]
        call_args['inputs'] = os.path.join(args.img_root, img_info['file_name'])
        call_args['texts'] = subject
        try:
            return_dict = inferencer(**call_args)
        except RuntimeError as e:
            print(e)
            error_list.append(key)
            continue
        out_dict[key] = {
            'scores': return_dict['predictions'][0]['scores'][:10],
            'boxes': return_dict['predictions'][0]['bboxes'][:10],
        }
        indices = cv2.dnn.NMSBoxes(
            out_dict[key]['boxes'],
            out_dict[key]['scores'],
            score_threshold=0.2,
            nms_threshold=0.7
        )
        num = len(indices)
        out_dict[key]['level'] = 1 if num <= 1 else 2
    with open(args.output_file, 'w') as f:
        json.dump(out_dict, f)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    def parse_args():
        parser = ArgumentParser(description="Object detection and instance level assignment using GroundingDINO.")
        parser.add_argument('--subjects-file', type=str, required=True, help='Path to the subjects JSON file (COCO image id to text mapping).')
        parser.add_argument('--coco-anno', type=str, required=True, help='Path to the COCO-format annotation file.')
        parser.add_argument('--img-root', type=str, required=True, help='Root directory of images.')
        parser.add_argument('--output-file', type=str, default='outputs/top_10_subject_qwen.json', help='Path to save the output JSON file.')
        parser.add_argument('--model', type=str, default='checkpoints/grounding_dino_swin-l_pretrain_all.py', help='Config or checkpoint .pth file or the model name and alias defined in metafile.')
        parser.add_argument('--weights', default='checkpoints/grounding_dino_swin-l_pretrain_all-56d69e78.pth', help='Checkpoint file')
        parser.add_argument('--device', default='cuda:0', help='Device used for inference')
        parser.add_argument('--palette', default='none', choices=['coco', 'voc', 'citys', 'random', 'none'], help='Color palette used for visualization')
        parser.add_argument('--chunked-size', '-s', type=int, default=-1, help='Truncate multiple predictions if categories are large.')
        parser.add_argument('--texts', help='Text prompt, such as "bench . car .", "$: coco"')
        parser.add_argument('--pred-score-thr', type=float, default=0.2, help='bbox score threshold')
        parser.add_argument('--batch-size', type=int, default=1, help='Inference batch size.')
        parser.add_argument('--show', action='store_true', help='Display the image in a popup window.')
        parser.add_argument('--no-save-vis', action='store_true', help='Do not save detection vis results')
        parser.add_argument('--no-save-pred', action='store_true', help='Do not save detection json results')
        parser.add_argument('--print-result', action='store_true', help='Whether to print the results.')
        parser.add_argument('--custom-entities', '-c', action='store_true', help='Whether to customize entity names?')
        parser.add_argument('--tokens-positive', '-p', type=str, default='-1', help='Specify which locations in the input text are of interest.')
        args = parser.parse_args()
        return args
    args = parse_args()
    run_detection(args)
