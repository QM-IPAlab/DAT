# The Detector Teaches Itself: Lightweight Self-Supervised Adaptation for Open-Vocabulary Object Detection

[![arXiv](https://img.shields.io/badge/arXiv-Paper-&lt;COLOR&gt;.svg)](https://arxiv.org/abs/XXXX.XXXXX) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-detector-teaches-itself/open-vocabulary-object-detection-on-mscoco)](https://paperswithcode.com/sota/open-vocabulary-object-detection-on-mscoco?p=the-detector-teaches-itself)

[Yazhe Wan](https://github.com/YazheWan) and [Changjae Oh](https://github.com/changjaeoh)

Queen Mary University of London, London, UK

Official code for our paper "The Detector Teaches Itself: Lightweight Self-Supervised Adaptation for Open-Vocabulary Object Detection"

## :rocket: News
* **(Month DD, YYYY)**
  * Paper accepted at [Conference Name YYYY].
* **(Month DD, YYYY)**
  * Project website with additional qualitative visualizations is now live at [https://yazhewan.github.io/dat-ovd/](https://yazhewan.github.io/dat-ovd/)
* **(Month DD, YYYY)**
  * Code for DAT and evaluation scripts has been released.

&lt;hr&gt;

![method-diagram](https://your-website.github.io/media/dat/method.png)
&gt; **Abstract:** *Open-vocabulary object detection aims to recognize objects from an open set of categories, which leverages vision-language models (VLMs) pre-trained on large-scale image-text data. The cooperative paradigm combines an object detector with a VLM to achieve zero-shot recognition of novel objects. However, VLMs pre-trained on full images often struggle to capture local object details, limiting their effectiveness when applied to region-level detection. We present Decoupled Adaptivity Training (DAT), a self-supervised fine-tuning approach to improve VLMs for cooperative model-based object detection. Given a cooperative model consists of a closed-set detector and a VLM, we first construct a region-aware pseudo-labeled dataset using a pre-trained closed-set object detector, in which regions corresponding to novel objects may be present but remain unlabeled or mislabeled. We then fine-tune the visual backbone of the VLM in a decoupled manner, which enhances local feature alignment while preserving global semantic knowledge via weight interpolation. DAT is a plug-and-play module that requires no inference overhead and fine-tunes less than 0.8M parameters. Experiments on the COCO and LVIS datasets show that DAT consistently improves detection performance on both novel and known categories, establishing a new state of the art in cooperative open-vocabulary detection.*

## :trophy: Achievements and Features

- We establish **state-of-the-art results (SOTA)** in cooperative open-vocabulary detection on COCO OVD and LVIS benchmarks.
- We propose a **lightweight, self-supervised, and training-free at inference** approach that adapts VLMs for region-level understanding.
- DAT is a **plug-and-play module** that can be seamlessly integrated into any cooperative detection pipeline (e.g., CFM) with **zero inference overhead**.
- We fine-tune **less than 0.8M parameters** (final LayerNorm and projection weights of SigLIP visual backbone) using a single GPU in under 2 hours.
- Our approach uses **self-generated pseudo-labels** from the detector itself, eliminating the need for external annotations or manual captions.
- DAT employs **weight interpolation (WiSE-FT)** to preserve global semantic knowledge while enhancing local feature adaptivity.

## :hammer_and_wrench: Setup and Installation

We have used `python=3.8.15`, and `torch=1.10.1` for all the code in this repository. It is recommended to follow the below steps and setup your conda environment in the same way to replicate the results mentioned in this paper and repository.

1. Clone this repository into your local machine as follows:
```bash
git clone git@github.com:YazheWan/dat-ovd.git
