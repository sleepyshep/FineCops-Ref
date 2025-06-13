# This script evaluates the results of the SFA method.
# Please configure the file paths below before running.
import json
import cv2

def count_elements_above_threshold(sorted_list, threshold):
    count = 0
    for element in sorted_list:
        if element > threshold:
            count += 1
        else:
            break
    return count

# === Configure your file paths here ===
SFA_RESULT_PATH = 'path/to/top_10_subject.json'  # SFA detection results
GROUNDINGDINO_PATH = 'path/to/groundingdino.json'  # GroundingDINO results
COGVLM_PATH = 'path/to/cogvlm.json'  # CogVLM results
QWEN_PATH = 'path/to/qwen.json'  # Qwen results
TOKEN_PATH = 'path/to/finecops_subject.json'  # Token file (not used in this script)

# === Load data ===
with open(SFA_RESULT_PATH, 'r') as f:
    result = json.load(f)
with open(GROUNDINGDINO_PATH, 'r') as f:
    groundingdino = json.load(f)
with open(COGVLM_PATH, 'r') as f:
    cogvlm = json.load(f)
with open(QWEN_PATH, 'r') as f:
    qwen = json.load(f)
with open(TOKEN_PATH, 'r') as f:
    t = json.load(f)

cnt = {
    '1_r': 0,  # correct single-object results
    '1_s': 0,  # total single-object results
    '2_r': 0,  # correct multi-object results
    '2_s': 0,  # total multi-object results
}

# Evaluate SFA results
for key, value in result.items():
    indices = cv2.dnn.NMSBoxes(value['boxes'], value['scores'], score_threshold=0.2, nms_threshold=0.7)
    num = len(indices)
    value['level'] = 1 if num <= 1 else 2

for key, value in result.items():
    if value['level'] == 1:
        cnt['1_s'] += 1
        iou = groundingdino[key]['iou']
        if iou > 0.5:
            cnt['1_r'] += 1
    else:
        iou = qwen[key]['iou']
        cnt['2_s'] += 1
        if iou > 0.5:
            cnt['2_r'] += 1

print('SFA Evaluation Results:')
if cnt['1_s'] > 0:
    print('Single-object accuracy: {:.2f}%'.format(cnt['1_r']/cnt['1_s']*100))
else:
    print('Single-object accuracy: N/A')
if cnt['2_s'] > 0:
    print('Multi-object accuracy: {:.2f}%'.format(cnt['2_r']/cnt['2_s']*100))
else:
    print('Multi-object accuracy: N/A')
total_s = cnt['1_s'] + cnt['2_s']
total_r = cnt['1_r'] + cnt['2_r']
if total_s > 0:
    print('Overall accuracy: {:.2f}%'.format(total_r / total_s * 100))
else:
    print('Overall accuracy: N/A')