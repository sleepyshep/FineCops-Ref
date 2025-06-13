import argparse
import json
import os
import time
import cv2
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def get_score(tokens, transition_scores, transition_scores_norm, tokenizer):
    scores_logits_average = []
    scores_prob_multi = []
    scores_prob_average = []
    tokens_length = []

    start = None
    end = None
    
    # tokenizer.decode(): index->word, tokenizer(): word->index
    for i, char in enumerate(tokens):
        decoded_char = tokenizer.decode(char)
        # Find the first '<|box_start|>'
        if decoded_char == '<|box_start|>' and start is None:
            start = i + 1
        # Find the second '<|box_end|>'
        if decoded_char == '<|box_end|>' and start is not None:
            end = i
            break
    if start is not None and end is not None:
        box_token = tokens[start:end]
        box = tokenizer.decode(box_token)
        # prob_average: mean value
        prob_average = sum(transition_scores[start:end]) / len(transition_scores[start:end])
        # prob_norm_sum: log sum -> log(product)
        prob_norm_sum = sum(transition_scores_norm[start:end])
        token_length = len(tokens[start:end])
        # prob_norm_exp_sum: log sum -> exp -> product -> mean
        prob_norm_exp_sum = sum(torch.exp(transition_scores_norm[start:end])) / len(transition_scores_norm[start:end])
        
        scores_logits_average.append(float(prob_average))
        # scores_prob_multi: log(prob) sum then exp => product of prob
        scores_prob_multi.append(float(torch.exp(prob_norm_sum)))
        scores_prob_average.append(float(prob_norm_exp_sum))
        tokens_length.append(token_length)
        
        return scores_logits_average, scores_prob_multi, scores_prob_average, tokens_length, box
    else:
        return [], [], [], [], []

def eval_model(args):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(args.model_path)

    from pycocotools.coco import COCO
    answers_file = args.output_file
    if os.path.exists(answers_file):
        print('answers_file already exists!')
    ans_file = open(answers_file, "w")
    
    if args.data == 'fineref':
        coco = COCO(args.annotation_file or 'test_expression_pos_coco_format.json')
        img_root = args.image_root or 'images/'
    elif args.data == 'adv':
        coco = COCO(args.annotation_file or 'ref_adv_coco.json')
        img_root = args.image_root or 'train2014/'
    elif args.data == 'reasoning':
        coco = COCO(args.annotation_file or 'ref_reasoning_coco.json')
        img_root = args.image_root or 'images/'
    else:
        raise ValueError(f"Unknown data type: {args.data}")

    with open(args.subjects_file, 'r') as f:
        subjects = json.load(f)

    img_ids = coco.getImgIds()
    for img_id in tqdm(img_ids):
        coco_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(img_root, f"{coco_info['file_name']}")
        expression = coco_info['caption']
        width = coco_info['width']
        height = coco_info['height']
        if args.focus_enhancement:
            prompt = f'Output the bounding box of {expression} in the image. Please focus on {subjects[str(img_id)]}.'
        else:
            prompt = f'Output the bounding box of {expression} in the image.'

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
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
        # Suppress specific warnings
        warnings.filterwarnings("ignore", category=UserWarning, message="`do_sample` is set to `False`")

        # Inference: Generation of the output
        start_time = time.time() 
        output_dict = model.generate(**inputs, max_new_tokens=256, return_dict_in_generate=True, output_scores=True, do_sample=False)
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
        
        transition_scores_norm = model.compute_transition_scores(generated_ids, output_score, normalize_logits=True)
        transition_scores = model.compute_transition_scores(generated_ids, output_score, normalize_logits=False)
        
        scores_logits_average, scores_prob_multi, scores_prob_average, tokens_length, box = get_score(generated_ids_trimmed[0], transition_scores[0], transition_scores_norm[0], processor)
        
        ans_file.write(json.dumps({"image_id": img_id,
                                    "prompt": prompt,
                                    "answer": output_text,
                                    "box": box,
                                    "scores_logits_average": scores_logits_average,
                                    "scores_prob_multi": scores_prob_multi,
                                    "scores_prob_average": scores_prob_average,
                                    "tokens_length": tokens_length,
                                    "cost_time": cost_time
                                    }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='fineref', type=str, choices=['fineref', 'adv', 'reasoning'], help="Dataset type")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained Qwen2VL model")
    parser.add_argument("--output_file", type=str, default='results.jsonl', help="Path to save the output results")
    parser.add_argument("--annotation_file", type=str, default=None, help="Path to COCO annotation file (optional)")
    parser.add_argument("--image_root", type=str, default=None, help="Root directory for images (optional)")
    parser.add_argument('--focus-enhancement', action='store_true', help='Implement the Focus-enhancement Strategy.')
    parser.add_argument('--subjects-file', type=str, required=True, help='Path to the subjects JSON file (COCO image id to text mapping).')
    args = parser.parse_args()
    eval_model(args)