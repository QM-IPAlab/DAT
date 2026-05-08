from .coco_ovd_split import categories_seen, categories_unseen # Assuming these are defined in coco_ovd_split.py
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

class McnnCOCOEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, output_dir=None, *, use_fast_impl=True, kpt_oks_sigmas=(), tasks=None):
        """
        Args:
            dataset_name (str): name of the dataset
            output_dir (str): output directory to save some files if needed.
            use_fast_impl (bool): whether to use a faster but not perfectly precise COCOEvaluator
                implementation.
            kpt_oks_sigmas (list[float]): The sigmas used by COCO keypoint OKS metric.
                See pycocotools.cocoeval.COCOeval for details.
            tasks (tuple[str]): tasks to evaluate. Can be ("bbox", "segm", "keypoints").
                By default, will infer from `dataset_name`.
        """
        # Pass all relevant arguments to the parent COCOEvaluator's __init__
        super().__init__(
            dataset_name=dataset_name,
            output_dir=output_dir,
            use_fast_impl=use_fast_impl,
            kpt_oks_sigmas=kpt_oks_sigmas,
            tasks=tasks
        )


    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Additionally plot mAP for 'seen classes' and 'unseen classes'
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        # Ensure categories_seen and categories_unseen are accessible here
        # If they are in another file, make sure to import them correctly.
        seen_names = set([x['name'] for x in categories_seen])
        unseen_names = set([x['name'] for x in categories_unseen])
        results_per_category = []
        results_per_category50 = []
        results_per_category50_seen = []
        results_per_category50_unseen = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))
            precision50 = precisions[0, :, idx, 0, -1]
            precision50 = precision50[precision50 > -1]
            ap50 = np.mean(precision50) if precision50.size else float("nan")
            results_per_category50.append(("{}".format(name), float(ap50 * 100)))
            if name in seen_names:
                results_per_category50_seen.append(float(ap50 * 100))
            if name in unseen_names:
                results_per_category50_unseen.append(float(ap50 * 100))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)


        N_COLS = min(6, len(results_per_category50) * 2)
        results_flatten = list(itertools.chain(*results_per_category50))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP50"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP50: \n".format(iou_type) + table)
        self._logger.info(
            "Seen {} AP50: {}".format(
                iou_type,
                sum(results_per_category50_seen) / len(results_per_category50_seen),
            ))
        self._logger.info(
            "Unseen {} AP50: {}".format(
                iou_type,
                sum(results_per_category50_unseen) / len(results_per_category50_unseen),
            ))

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        # Handle cases where seen/unseen lists might be empty to avoid division by zero
        results["AP50-seen"] = sum(results_per_category50_seen) / len(results_per_category50_seen) if results_per_category50_seen else float('nan')
        results["AP50-unseen"] = sum(results_per_category50_unseen) / len(results_per_category50_unseen) if results_per_category50_unseen else float('nan')
        return results