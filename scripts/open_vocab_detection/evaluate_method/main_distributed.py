import os
import sys
import json
import torch
import warnings
import torch.distributed as dist
import torch.multiprocessing as mp

proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(proj_path)

def setup_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

def main_worker(rank, world_size):

    setup_distributed(rank, world_size)

    import detectron2.data.transforms as T
    from gd.groundingdino.util.inference import load_model
    from scripts.open_vocab_detection.evaluate_method.load_models import (
        load_fully_supervised_trained_model, load_clip_model, load_sam_model
    )
    from scripts.open_vocab_detection.evaluate_method.utils import (
        get_text_prompt_for_g_dino, get_ovd_id_to_coco_id
    )
    from scripts.open_vocab_detection.evaluate_method.evaluator_loop import inference
    from pathlib import Path
    from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts, DatasetMapper
    from detectron2.evaluation import print_csv_format
    from segment_anything.utils.transforms import ResizeLongestSide
    from datasets.register_coco_ovd_dataset import coco_meta  
    from scripts.open_vocab_detection.coco_eval_utils.custom_coco_eval import CustomCOCOEvaluator

    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)


    script_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(script_dir, "params.json")
    with open(params_path, "r") as f:
        params = json.load(f)

    detectron2_dir = params["detectron2_dir"]
    visualize = params["visualize"]
    data_split = params["data_split"]
    cfg_file = params["cfg_file"]
    rcnn_weight_dir = params["rcnn_weight_dir"]
    sam_checkpoint = params["sam_checkpoint"]
    gdino_checkpoint = params["gdino_checkpoint"]

    outputs_dir = os.path.normpath(os.path.join(script_dir, "../../../outputs/coco/no_wise/GT"))
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

    os.environ['DETECTRON2_DATASETS'] = detectron2_dir

    device = torch.device(f"cuda:{rank}")


    model = load_model("cfg/GroundingDINO/GDINO.py", gdino_checkpoint)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    rcnn_model, cfg = load_fully_supervised_trained_model(cfg_file, rcnn_weight_dir)
    ovd_id_to_coco_id = get_ovd_id_to_coco_id()
    clip_model, preprocess, text_features = load_clip_model(device, "weights/CLIP_Weights/GT/In_train_weights/LM_epoch_4_dsize_1000.pth")
    sam = load_sam_model(device, sam_checkpoint)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)


    dataset_dicts = get_detection_dataset_dicts(names=data_split, filter_empty=False)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_dicts, num_replicas=world_size, rank=rank, shuffle=False)

    test_loader = build_detection_test_loader(
        dataset=dataset_dicts,
        mapper=DatasetMapper(
            is_train=False,
            augmentations=[T.ResizeShortestEdge(short_edge_length=800, max_size=1333)],
            image_format="BGR",
        ),
        sampler=sampler,
        num_workers=4,
    )

    tokenizer = model.module.tokenizer if hasattr(model, "module") else model.tokenizer
    text_prompt, positive_map = get_text_prompt_for_g_dino(tokenizer)
    coco_evaluator = CustomCOCOEvaluator(dataset_name=data_split, distributed=True)


    param_dict = {
        "visualize": visualize,
        "out_dir": outputs_dir,
        "data_split": data_split,
        "positive_map": positive_map,
        "rcnn_model": rcnn_model,
        "clip_model": clip_model,
        "preprocess": preprocess,
        "text_features": text_features,
        "device": device,
        "ovd_id_to_coco_id": ovd_id_to_coco_id,
        "sam": sam,
        "resize_transform": resize_transform,
    }


    torch.cuda.empty_cache()
    results = inference(test_loader, coco_evaluator, model, text_prompt, param_dict)


    if rank == 0:
        print_csv_format(results)
        clip_name = os.path.basename("weights/CLIP_Weights/GT/In_train_weights/LM_epoch_4_dsize_1000.pth").replace(".pth", "")
        save_path = os.path.join(outputs_dir, f"{clip_name}_{data_split}.json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[Rank 0] Saved evaluation results to: {save_path}")

    cleanup_distributed()


def main():
    world_size = torch.cuda.device_count()
    print(f"Launching distributed evaluation on {world_size} GPUs...")
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
