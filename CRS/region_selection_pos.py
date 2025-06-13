import json
import os
import time
import cv2
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from pycocotools.coco import COCO   

# Please configure the following default paths as needed, or pass them via command line arguments
DEFAULT_MODEL_PATH = './checkpoints/Qwen2-VL-7B-Instruct'
DEFAULT_PROCESSOR_PATH = './checkpoints/Qwen2-VL-7B-Instruct'
DEFAULT_DATA_ROOT = './data'
DEFAULT_OUTPUT_ROOT = './outputs'

def convert_qwen_box(bbox, width, height):
    x1, y1, x2, y2 = bbox
    new_bbox = [
        round((x1 / width) * 1000),
        round((y1 / height) * 1000),
        round((x2 / width) * 1000),
        round((y2 / height) * 1000)
    ]
    return f'({new_bbox[0]},{new_bbox[1]}),({new_bbox[2]},{new_bbox[3]})'

def opt_pos(model_path, ans_path, top_k, data, processor_path=DEFAULT_PROCESSOR_PATH, data_root=DEFAULT_DATA_ROOT, output_root=DEFAULT_OUTPUT_ROOT):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(processor_path)

    # Data path configuration
    if data == 'fineref':
        groundingdino = json.load(open(os.path.join(output_root, 'top_10.json'), 'r'))
        coco = COCO(os.path.join(data_root, 'fineref/test_expression_pos_coco_format.json'))
        img_root = os.path.join(data_root, 'fineref/images/')
    elif data == 'adv':
        groundingdino = json.load(open(os.path.join(output_root, 'adv_top10.json'), 'r'))
        coco = COCO(os.path.join(data_root, 'adv/ref_adv_coco.json'))
        img_root = os.path.join(data_root, 'adv/images/')
    elif data == 'reasoning':
        groundingdino = json.load(open(os.path.join(output_root, 'reasoning_top10.json'), 'r'))
        coco = COCO(os.path.join(data_root, 'reasoning/ref_reasoning_coco.json'))
        img_root = os.path.join(data_root, 'reasoning/images/')
    else:
        raise ValueError(f"Unknown data type: {data}")

    if os.path.exists(ans_path):
        print('answers_file already exist!')
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
        add_prompt = "\nAnswer with the option's letter from the given choices directly."

        boxes = groundingdino[str(img_id)]['bbox'][:top_k]
        scores = groundingdino[str(img_id)]['scores'][:top_k]
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0, nms_threshold=0.7)
        remain_boxes = [boxes[i] for i in indices]
        gt = dict()
        for id, i in enumerate(indices):
            gt[multiple_choices[id]] = boxes[i]
        choice_list = []
        options = dict()
        for i, box in enumerate(remain_boxes):
            box = convert_qwen_box(box, width, height)
            choice_list.append('{}. {}'.format(multiple_choices[i], box.strip()))
            options[multiple_choices[i]] = box.strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
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
        start_time = time.time()
        output_dict = model.generate(**inputs, max_new_tokens=128, return_dict_in_generate=True, output_scores=True, do_sample=False)
        end_time = time.time()
        cost_time = end_time - start_time
        generated_ids, output_score = output_dict.sequences, output_dict.scores
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        ans_file.write(json.dumps({"image_id": img_id,
                                    "prompt": prompt,
                                    "answer": output_text,
                                    "gt": gt,
                                    "cost_time": cost_time,
                                    }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help='Path to the model directory')
    parser.add_argument('--processor', default=DEFAULT_PROCESSOR_PATH, help='Path to the processor directory')
    parser.add_argument('--ans', default='result.jsonl', type=str, help='Output answer file path')
    parser.add_argument('--top_k', default=5, type=int, choices=[3, 5, 8, 10], help='Top K boxes')
    parser.add_argument('--data', default='fineref', type=str, choices=['fineref', 'adv', 'reasoning'], help='Dataset type')
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT, type=str, help='Root path for datasets')
    parser.add_argument('--output_root', default=DEFAULT_OUTPUT_ROOT, type=str, help='Root path for detection results')
    args = parser.parse_args()
    opt_pos(
        model_path=args.model,
        ans_path=args.ans,
        top_k=args.top_k,
        data=args.data,
        processor_path=args.processor,
        data_root=args.data_root,
        output_root=args.output_root
    )   