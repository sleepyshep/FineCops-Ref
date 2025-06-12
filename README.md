# New Dataset and Methods for Fine-Grained Compositional Referring Expression Comprehension via Specialist-MLLM Collaboration  

## 📜 News 

🔥 **[2025/06/12]** We open-sourced the methods developed via **Specialist–MLLM collaboration**, including **Slow-Fast Adaptation (SFA)** and **Candidate Region Selection (CRS)**.

🔥 **[2025/06/01]** Our extended paper was accepted by **TPAMI 2025**.

🔥 **[2024/09/20]** Our previous paper was accepted by **EMNLP 2024**.

🔥 **[2024/06/17]** Released the **[FineCops-Ref dataset](https://github.com/liujunzhuo/FineCops-Ref)** — featuring controlled difficulty levels and challenging negative samples.

## ✒️ Contents

- [News](#news)
- [Contents](#contents)
- [Overview](#overview)
- [Preparation](#preparation)
- [Usage](#usage)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)

## 👀 Overview

**Referring Expression Comprehension (REC)** is a foundational cross-modal task that evaluates the interplay of language understanding, image comprehension, and language-to-image grounding. It serves as an essential testing ground for Multimodal Large Language Models (MLLMs). To advance this field, we introduced a new REC dataset in our previous conference paper, characterized by two key features. First, it is designed with **controllable difficulty levels**, requiring multi-level fine-grained reasoning across object categories, attributes, and multi-hop relationships. Second, it incorporates **negative text and images** generated through fine-grained editing and augmentation, explicitly testing a model’s ability to reject scenarios where the target object is absent—an often-overlooked yet critical challenge in existing datasets. In this extended work, we propose two new methods to tackle the challenges of fine-grained REC by combining the strengths of Specialist Models and MLLMs. **The first method** adaptively assigns simple cases to faster, lightweight models and reserves complex ones for powerful MLLMs, balancing accuracy and efficiency. **The second method** lets a specialist generate a set of possible object regions, and the MLLM selects the most plausible one using its reasoning ability. These collaborative strategies lead to significant improvements on our dataset and other challenging benchmarks. Our results show that combining specialized and general-purpose models offers a practical path toward solving complex real-world vision-language tasks.

<div align=center>
<img width="600" alt="image" src="./assests/method.png">
</div>

## 👨‍💻 Preparation

1. Download REC Benchmarks:

   - Download the **FineCops-Ref** dataset from [FineCops-Ref Dataset](https://github.com/liujunzhuo/FineCops-Ref).

   - Optionally, download [**Ref-Adv**](https://github.com/aws/aws-refcocog-adv) and [**Ref-Reasoning**](https://github.com/sibeiyang/sgmn) datasets for evaluation.

   - Place datasets in the `data/` directory (e.g., `data/finecops-ref/`, `data/ref-adv/`, `data/ref-reasoning/`).

3. Download pre-trained model checkpoints:
   - Obtain checkpoints for Specialist Models (e.g., [Grounding DINO](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino)) and MLLMs (e.g., [Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL), [InternVL](https://github.com/OpenGVLab/InternVL)) from their respective repositories.
   - Place checkpoints in the `checkpoints/` directory.

## 🎯 Usage

The repository supports evaluating SFA and CRS on FineCops-Ref and other REC benchmarks. The `--model_type` parameter specifies the method (`sfa` or `crs`). The `--top_k` parameter in CRS controls the number of candidate bounding boxes (default: 5).

1. Example for evaluating FineCops-Ref with SFA:

```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/finecops_sfa.sh
```

2. Example for evaluating FineCops-Ref with CRS (default top-5 candidates):

```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/finecops_crs.sh
```

3. Example for evaluating Ref-Adv with SFA:

```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/refadv_sfa.sh
```

4. Example for evaluating Ref-Reasoning with CRS:

```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/refreasoning_crs.sh
```

5. Example for instruction tuning CRS on RefCOCO (12k samples):

```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/train/crs_instruction_tuning.sh
```

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use FineCops-Ref or our methods in your research, please cite our work using the following BibTeX entry:

```bibtex
@article{yang2025new,
  title={New Dataset and Methods for Fine-Grained Compositional Referring Expression Comprehension via Specialist-MLLM Collaboration},
  author={Yang, Xuzheng and Liu, Junzhuo and Wang, Peng and Wang, Guoqing and Yang, Yang and Shen, Heng Tao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
}
```

## Acknowledgment

We extend our gratitude to the open-source efforts of [Grounding DINO](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino), [Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL), [InternVL](https://github.com/OpenGVLab/InternVL).
