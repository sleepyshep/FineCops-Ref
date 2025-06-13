import json
import os
import cv2
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
from pycocotools.coco import COCO

# Default paths
DEFAULT_MODEL_PATH = './checkpoints/Qwen2-VL-7B-Instruct'
DEFAULT_PROCESSOR_PATH = './checkpoints/Qwen2-VL-7B-Instruct'
DEFAULT_DATA_ROOT = './data'
DEFAULT_OUTPUT_ROOT = './outputs'
DEFAULT_COCO_ANN = './data/fineref/test_expression_all_coco_format.json'
DEFAULT_IMG_ROOT = './data/fineref/images/'
DEFAULT_DET_RESULT = './outputs/fineref_all_top10.json'


def convert_qwen_box(bbox, width, height):
    x1, y1, x2, y2 = bbox
    new_bbox = [
        round((x1 / width) * 1000),
        round((y1 / height) * 1000),
        round((x2 / width) * 1000),
        round((y2 / height) * 1000)
    ]
    return f'({new_bbox[0]},{new_bbox[1]}),({new_bbox[2]},{new_bbox[3]})'


def opt_neg(model_path, ans_path, top_k, processor_path=DEFAULT_PROCESSOR_PATH, coco_ann=DEFAULT_COCO_ANN, img_root=DEFAULT_IMG_ROOT, det_result=DEFAULT_DET_RESULT):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(processor_path)
    groundingdino = json.load(open(det_result, 'r'))
    coco = COCO(coco_ann)
    img_root = img_root if img_root.endswith('/') else img_root + '/'

    if os.path.exists(ans_path):
        print('answers_file already exist!')
        return
    ans_file = open(ans_path, "w")
    multiple_choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    img_ids = coco.getImgIds()
    for img_id in tqdm(img_ids):
        coco_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(img_root, f"{coco_info['file_name']}")
        expression = coco_info['caption']
        width = coco_info['width']
        height = coco_info['height']
        prompt = f'Given a referring expression, select the best-matching bounding box from the options below. Referring expression: {expression}\n'
        add_prompt = "\nAnswer with the option's letter from the given choices directly. If no suitable option exists, please select the option corresponding to 'None'."

        boxes = groundingdino[str(img_id)]['bbox'][:top_k]
        scores = groundingdino[str(img_id)]['scores'][:top_k]
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0, nms_threshold=0.7)
        remain_boxes = [boxes[i] for i in indices]

        gt = dict()
        for idx, i in enumerate(indices):
            gt[multiple_choices[idx]] = boxes[i]
        gt[multiple_choices[len(indices)]] = 'None'
        choice_list = []
        options = dict()
        for idx, box in enumerate(remain_boxes):
            box_str = convert_qwen_box(box, width, height)
            choice_list.append('{}. {}'.format(multiple_choices[idx], box_str.strip()))
            options[multiple_choices[idx]] = box_str.strip()
        options[multiple_choices[len(remain_boxes)]] = 'None'
        choice_list.append('{}. {}'.format(multiple_choices[len(remain_boxes)], 'None'))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt + '\n'.join(choice_list) + add_prompt},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, message="`do_sample` is set to `False`")

        # Inference: Generation of the output
        output_dict = model.generate(**inputs, max_new_tokens=128, return_dict_in_generate=True, output_scores=True, do_sample=False)
        generated_ids, output_score = output_dict.sequences, output_dict.scores
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)

        transition_scores_norm = model.compute_transition_scores(generated_ids, output_score, normalize_logits=True)
        transition_scores = model.compute_transition_scores(generated_ids, output_score, normalize_logits=False)
        input_token_len = inputs.data['input_ids'].shape[1]
        n_diff_input_output = (inputs.data['input_ids'] != generated_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {img_id}: {n_diff_input_output} generated_ids are not the same as the input_ids')

        # option score (token id mapping may need adjustment according to the actual tokenizer)
        token_ids = {'A': 32, 'B': 33, 'C': 34, 'D': 35, 'E': 36, 'F': 37, 'G': 38, 'H': 39, 'I': 40, 'J': 41}
        none_token = token_ids[multiple_choices[len(remain_boxes)]]
        normalized_scores = F.log_softmax(output_score[0], dim=1)
        none_score = output_score[0][0][none_token]
        none_score_norm = normalized_scores[0][none_token]

        ans_file.write(json.dumps({
            "image_id": img_id,
            "prompt": prompt,
            "answer": output_text,
            "gt": gt,
            "answer_prob": transition_scores[0][0].item(),
            "answer_prob_norm": transition_scores_norm[0][0].item(),
            "none_prob": none_score.item(),
            "none_prob_norm": none_score_norm.item(),
        }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-Choice Region Selection with Qwen2-VL (Negative Option)')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help='Path to the model directory')
    parser.add_argument('--processor', default=DEFAULT_PROCESSOR_PATH, help='Path to the processor directory')
    parser.add_argument('--ans', default='fineref_all_neg.jsonl', type=str, help='Output answer file path')
    parser.add_argument('--top_k', default=5, type=int, choices=[3, 5, 8, 10], help='Top K boxes')
    parser.add_argument('--coco_ann', default=DEFAULT_COCO_ANN, type=str, help='COCO annotation file path')
    parser.add_argument('--img_root', default=DEFAULT_IMG_ROOT, type=str, help='Image root directory')
    parser.add_argument('--det_result', default=DEFAULT_DET_RESULT, type=str, help='Detection result JSON file')
    args = parser.parse_args()

    opt_neg(
        model_path=args.model,
        ans_path=args.ans,
        top_k=args.top_k,
        processor_path=args.processor,
        coco_ann=args.coco_ann,
        img_root=args.img_root,
        det_result=args.det_result
    )