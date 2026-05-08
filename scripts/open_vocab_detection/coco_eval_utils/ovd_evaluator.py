import os
import json
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from loguru import logger

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import Boxes, Instances, BitMasks
import detectron2.utils.comm as comm

# Import your seen/unseen categories
from scripts.open_vocab_detection.coco_eval_utils.coco_ovd_split import categories_seen, categories_unseen

class OVDEvaluator(DatasetEvaluator):
    """
    Custom Evaluator for Open-Vocabulary Detection (OVD) on COCO-style datasets.
    Calculates AP for seen and unseen classes separately.
    """
    def __init__(self, dataset_name, distributed, output_dir=None, use_fast_impl=True):
        """
        Args:
            dataset_name (str): Name of the dataset, e.g., "coco_2017_val_unseen".
            distributed (bool): Whether the evaluation is distributed.
            output_dir (str, optional): Directory to save results.
            use_fast_impl (bool): Whether to use the faster COCOeval implementation.
        """
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        self._cpu_device = torch.device("cpu")
        self._predictions = [] # Store raw Detectron2 prediction dictionaries

        # Load ground truth COCO annotations for the dataset
        # Assuming your dataset_name maps to a COCO annotation file
        self._gt_json_file = MetadataCatalog.get(dataset_name).json_file
        assert os.path.exists(self._gt_json_file), f"GT JSON file not found: {self._gt_json_file}"
        self._coco_gt = COCO(self._gt_json_file)

        # Get COCO JSON IDs for seen and unseen classes
        self._seen_coco_json_ids = {cat['id'] for cat in categories_seen}
        self._unseen_coco_json_ids = {cat['id'] for cat in categories_unseen}
        
        # Get D2 contiguous IDs to COCO JSON IDs mapping from metadata for filtering
        self._d2_contiguous_to_coco_json_id = MetadataCatalog.get(dataset_name).get("thing_contiguous_id_to_dataset_id")
        self._coco_json_to_d2_contiguous_id = MetadataCatalog.get(dataset_name).get("thing_dataset_id_to_contiguous_id")

        logger.info(f"OVD Evaluator initialized for {dataset_name}. GT JSON: {self._gt_json_file}")
        logger.info(f"Seen COCO JSON IDs: {self._seen_coco_json_ids}")
        logger.info(f"Unseen COCO JSON IDs: {self._unseen_coco_json_ids}")
        logger.info(f"D2 Contiguous to COCO JSON ID Map: {self._d2_contiguous_to_coco_json_id}")

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Process predictions from one image.
        Args:
            inputs (list[dict]): The inputs to the model (from D2 data loader).
            outputs (list[dict]): The outputs from the model (_inference_maskrcnn_siglip's return value).
        """
        for input_per_image, output_per_image in zip(inputs, outputs):
            image_id = input_per_image["image_id"]
            instances = output_per_image["instances"].to(self._cpu_device)

            if len(instances) == 0:
                continue

            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            # Convert D2 contiguous IDs back to COCO JSON IDs for pycocotools
            pred_classes_d2_contiguous = instances.pred_classes.tolist()
            pred_classes_coco_json_id = [self._d2_contiguous_to_coco_json_id[d2_id] for d2_id in pred_classes_d2_contiguous]
            
            # Convert masks to COCO RLE format
            rle_masks = []
            if instances.has("pred_masks"):
                masks = instances.pred_masks.numpy()
                for mask in masks:
                    rle_masks.append(mask_util.encode(np.asfortranarray(mask)))

            for i in range(len(instances)):
                self._predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": pred_classes_coco_json_id[i],
                        "bbox": boxes[i].tolist(),
                        "score": scores[i],
                        "segmentation": rle_masks[i] if rle_masks else None,
                    }
                )

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if not self._predictions:
            logger.warning("No predictions collected for evaluation.")
            return {"segm_AP_seen": 0.0, "segm_AP_unseen": 0.0, "bbox_AP_seen": 0.0, "bbox_AP_unseen": 0.0}

        # Save predictions to a JSON file (optional, but good practice)
        if self._output_dir:
            Path(self._output_dir).mkdir(parents=True, exist_ok=True)
            pred_file = os.path.join(self._output_dir, f"coco_instances_results_{self._dataset_name}.json")
            with open(pred_file, "w") as f:
                json.dump(self._predictions, f)
            logger.info(f"Saved predictions to {pred_file}")

        # Initialize COCO result object
        coco_dt = self._coco_gt.loadRes(self._predictions)

        results = {}

        # --- Evaluate for Seen Classes ---
        logger.info("Evaluating for SEEN classes...")
        coco_eval_seen = COCOeval(self._coco_gt, coco_dt, iouType="segm")
        coco_eval_seen.params.catIds = list(self._seen_coco_json_ids) # Filter by seen categories
        coco_eval_seen.evaluate()
        coco_eval_seen.accumulate()
        coco_eval_seen.summarize()
        results["segm_AP_seen"] = coco_eval_seen.stats[0] # AP @ IoU=0.5:0.95
        results["segm_AP50_seen"] = coco_eval_seen.stats[1] # AP @ IoU=0.5

        # Also for bbox
        coco_eval_seen_bbox = COCOeval(self._coco_gt, coco_dt, iouType="bbox")
        coco_eval_seen_bbox.params.catIds = list(self._seen_coco_json_ids)
        coco_eval_seen_bbox.evaluate()
        coco_eval_seen_bbox.accumulate()
        coco_eval_seen_bbox.summarize()
        results["bbox_AP_seen"] = coco_eval_seen_bbox.stats[0]
        results["bbox_AP50_seen"] = coco_eval_seen_bbox.stats[1]


        # --- Evaluate for Unseen Classes ---
        logger.info("Evaluating for UNSEEN classes...")
        coco_eval_unseen = COCOeval(self._coco_gt, coco_dt, iouType="segm")
        coco_eval_unseen.params.catIds = list(self._unseen_coco_json_ids) # Filter by unseen categories
        coco_eval_unseen.evaluate()
        coco_eval_unseen.accumulate()
        coco_eval_unseen.summarize()
        results["segm_AP_unseen"] = coco_eval_unseen.stats[0]
        results["segm_AP50_unseen"] = coco_eval_unseen.stats[1]

        # Also for bbox
        coco_eval_unseen_bbox = COCOeval(self._coco_gt, coco_dt, iouType="bbox")
        coco_eval_unseen_bbox.params.catIds = list(self._unseen_coco_json_ids)
        coco_eval_unseen_bbox.evaluate()
        coco_eval_unseen_bbox.accumulate()
        coco_eval_unseen_bbox.summarize()
        results["bbox_AP_unseen"] = coco_eval_unseen_bbox.stats[0]
        results["bbox_AP50_unseen"] = coco_eval_unseen_bbox.stats[1]
        
        return results