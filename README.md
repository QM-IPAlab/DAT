



# The Detector Teaches Itself: Lightweight Self-Supervised Adaptation for Open-Vocabulary Object Detection

[![Project Website](https://img.shields.io/badge/Project-Website-blue)]([https://qm-ipalab.github.io/DAT/](https://github.com/QM-IPAlab/DAT/blob/d6640d397a1a42a5585f77e826389eccca9517b3/index.html)) [![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/)

[Yazhe Wan](https://qm-ipalab.github.io/DAT/), [Changjae Oh](https://qm-ipalab.github.io/DAT/)
*Queen Mary University of London, London, UK*

Official code for our paper "The Detector Teaches Itself: Lightweight Self-Supervised Adaptation for Open-Vocabulary Object Detection" (DAT).

## :rocket: News
* **(Recent)**
  * Project website is live at [https://qm-ipalab.github.io/DAT/]([https://qm-ipalab.github.io/DAT/](https://github.com/QM-IPAlab/DAT/blob/d6640d397a1a42a5585f77e826389eccca9517b3/index.html))
  * Code and pre-trained weights for our method (DAT) on Open-Vocabulary Detection and Novel Object Detection have been released.

<hr>

![method-diagram](https://qm-ipalab.github.io/DAT/media/coop_exp-2.pdf)


> **Abstract:** *Open-vocabulary object detection aims to recognize objects from an open set of categories, which leverages vision-language models (VLMs) pre-trained on large-scale image-text data. The cooperative paradigm combines an object detector with a VLM to achieve zero-shot recognition of novel objects. However, VLMs pre-trained on full images often struggle to capture local object details, limiting their effectiveness when applied to region-level detection. We present Decoupled Adaptivity Training (DAT), a self-supervised fine-tuning approach to improve VLMs for cooperative model-based object detection. Given a cooperative model consists of a closed-set detector and a VLM, we first construct a region-aware pseudo-labeled dataset using a pre-trained closed-set object detector, in which regions corresponding to novel objects may be present but remain unlabeled or mislabeled. We then fine-tune the visual backbone of the VLM in a decoupled manner, which enhances local feature alignment while preserving global semantic knowledge via weight interpolation. DAT is a plug-and-play module that requires no inference overhead and fine-tunes less than 0.8M parameters. Experiments on the COCO and LVIS datasets show that DAT consistently improves detection performance on both novel and known categories, establishing a new state of the art in cooperative open-vocabulary detection.*

## :trophy: Achievements and Features

- We establish **new state-of-the-art results (SOTA)** in open-vocabulary detection: achieving **70.1 Novel AP<sub>50</sub>** on COCO OVD and **17.68 Novel AP** on LVIS v1.0.
- **Self-Supervised & Data-Efficient:** The detector "teaches itself" using its own pseudo-labels to adapt the VLM, requiring no extra external manual annotations.
- **Parameter-Efficient:** DAT only fine-tunes less than **0.8M parameters** (using techniques like WiSE-FT and selective visual backbone tuning) on a single GPU in under two hours.
- **Plug-and-Play:** The adapted VLM backbone directly replaces the original one in the cooperative pipeline without incurring any additional architectural or inference-time overhead.

## :hammer_and_wrench: Setup and Installation

We recommend using our provided bash script to automatically set up the conda environment and install all dependencies (including `python=3.8.15`, `torch=1.10.1`, and `Detectron2 v0.6`).

1. Clone this repository into your local machine:
```bash
git clone https://github.com/qm-ipalab/DAT.git
cd DAT
```
2. Run the environment setup script to automatically create the conda environment and install all required libraries (including `detectron2`):
```bash
bash env.sh
```
3. Activate the newly created conda environment:
```bash
# Replace 'dat_env' with the exact environment name defined in your env.sh
conda activate dat_env
```

### Datasets
To download and setup the required datasets used in this work, please follow these steps:
1. Download the COCO2017 dataset from their official website: [https://cocodataset.org/#download](https://cocodataset.org/#download). Specifically, download `2017 Train images`, `2017 Val images`, `2017 Test images`, and their annotation files `2017 Train/Val annotations`.
2. Download the LVIS v1.0 annotations from: [https://www.lvisdataset.org/dataset](https://www.lvisdataset.org/dataset). There is no need to download images from this website as LVIS uses the same COCO2017 images. Specifically download the annotation files corresponding to the training set (1GB), and validation set (192 MB).
3. Download extra/custom annotation files for COCO open-vocabulary splits from: [COCO-OVD-Annotations](YOUR_LINK_HERE).
4. Download extra/custom annotation file for `lvis_val_subset` dataset from: [LVIS-Val-Subset](YOUR_LINK_HERE).
5. Detectron2 requires you to setup the datasets in a specific folder format/structure, for that it uses the environment variable `DETECTRON2_DATASETS`. The file structure should be as follows:
- `coco/`
  - `annotations/`
  - `train2017/`
  - `val2017/`
  - `test2017/`
- `lvis/`
  - `lvis_v1_val.json`
  - `lvis_v1_train.json`
  - `lvis_v1_val_subset.json`

Ensure the `detectron2_dir` variable in our `.json` config files points to the absolute path of this dataset directory.

### Model Weights
Place the pre-trained model weights in the required directories and configure the `params.json` files accordingly. The weights include:

- **GDINO_weights.pth**: Grounding DINO model weights.
- **SAM_weights.pth**: Segment Anything Model (SAM) weights.
- **maskrcnn_v2 / MaskRCNN_COCO_OVD**: Pre-trained Mask-RCNN weights for closed-set region proposals.
- **DAT_SigLIP_weights.pth**: *(New)* Our fine-tuned SigLIP visual backbone weights.

## :mag_right: Open-Vocabulary Detection on LVIS v1.0 Val

Our DAT framework significantly enhances the cooperative baseline across all splits.

| Method               | VLM             | Novel AP | Known AP | All AP  |
|----------------------|-----------------|----------|----------|---------|
| RNCDL                | —               | 5.42     | 25.00    | 6.92    |
| GDINO                | —               | 13.47    | 37.13    | 15.30   |
| CFM (Baseline)       | SigLIP          | 17.42    | 42.08    | 19.33   |
| **Ours (DAT)**       | **SigLIP<sub>ft</sub>** | **17.68**| **45.82**| **20.37**|

**Table 1:** Comparison of object detection performance using mAP on the *lvis_val* dataset.

*To reproduce these results, configure the dataset and weight paths in `scripts/evaluate_lvis/params.json` and run the evaluation script (details to be updated).*

## :medal_military: Open Vocabulary Detection on COCO OVD Dataset

| Method                     | Pre-training                | Detection Training Data| Novel AP<sub>50</sub> | Base AP<sub>50</sub> | All AP<sub>50</sub> |
|-----------------------------|-------------------------|------------------------|-----------------------|----------------------|---------------------|
| BARON                   | SOCO, MAVL              | COCO, CC, CLIP         | 42.7                  | 54.9                 | 51.7                |
| CORA+                   | —                       | COCO, CC, CLIP         | 43.1                  | 60.9                 | 56.2                |
| CFM                     | GDINO, SAM, SigLIP      | COCO                   | 50.3                  | 49.8                 | 49.9                |
| **Ours (DAT)**          | **GDINO, SAM, SigLIP<sub>ft</sub>** | **COCO**       | **70.1**              | **55.5**             | **59.3**            |

**Table 2:** Results on the COCO OVD benchmark. Our method outperforms all prior cooperative and end-to-end approaches, achieving the highest novel-class AP<sub>50</sub> by a large margin.

*To reproduce these results, configure the paths in `scripts/evaluate_coco/params.json` and run the main script.*

## :framed_picture: Qualitative Visualization

Our fine-tuned SigLIP backbone exhibits substantially improved region–text alignment, especially for challenging cases such as small, occluded, or semantically ambiguous objects compared to the original CFM baseline.

| Input Image | CFM Baseline | Ours (DAT) |
|-------------|--------------|------------|
| <img src="media/img_1_input.jpg" width="200"/> | <img src="media/img_1_cfm.jpg" width="200"/> | <img src="media/img_1_ours.jpg" width="200"/> |
| <img src="media/img_2_input.jpg" width="200"/> | <img src="media/img_2_cfm.jpg" width="200"/> | <img src="media/img_2_ours.jpg" width="200"/> |
| <img src="media/img_3_input.jpg" width="200"/> | <img src="media/img_3_cfm.jpg" width="200"/> | <img src="media/img_3_ours.jpg" width="200"/> |

To see more visualizations and details, please visit our [project website](https://qm-ipalab.github.io/DAT/).

## :email: Contact
Should you have any questions, please create an issue in this repository or contact Yazhe Wan at `yazhe.wan@qmul.ac.uk`.

## :pray: Acknowledgement
This work was supported by the Korea Institute for Advancement of Technology (KIAT) grant. We also thank the authors of CFM, GDINO, SAM, and Mask R-CNN for their foundational work and open-source contributions.

## :black_nib: Citation
If you found our work helpful, please consider starring the repository ⭐⭐⭐ and citing our paper:
```bibtex
@inproceedings{wan2026detector,
    author    = {Wan, Yazhe and Oh, Changjae},
    title     = {The Detector Teaches Itself: Lightweight Self-Supervised Adaptation for Open-Vocabulary Object Detection},
    booktitle = {arXiv preprint},
    year      = {2026}
}
```

--- 

