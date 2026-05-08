import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 
import sys
import json
import cv2
from loguru import logger
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import open_clip
from detectron2.data import build_detection_train_loader, get_detection_dataset_dicts, DatasetMapper
import detectron2.data.transforms as T

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12231'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def truncate_dataloader(dataloader, num_batches):
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        yield batch

class CroppedImageDataset(Dataset):

    def __init__(self, images, labels, preprocess=None):
        self.images = images
        self.labels = labels
        self.preprocess = preprocess

        assert len(self.images) == len(self.labels), "images and labels must have same length"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_item = self.images[idx]
        label = self.labels[idx]

        if isinstance(img_item, torch.Tensor):
            if img_item.dim() == 4 and img_item.shape[0] == 1:
                img_tensor = img_item.squeeze(0)
            else:
                img_tensor = img_item
            return img_tensor, label
        else:
            if self.preprocess is None:
                img_tensor = transforms.ToTensor()(img_item)
            else:
                img_tensor = self.preprocess(img_item)
            return img_tensor, label


def load_clip_model(device):
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP', pretrained='webli')
    tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP')
    clip_model = clip_model.to(device)
    return clip_model, tokenizer, preprocess


def extract_gt_cropped_images_and_labels_from_batch(raw_batch, preprocess, coco_id_name_map):
    cropped_list = []
    labels_list = []

    for item in raw_batch:
        img_tensor = item.get('image')  
        if img_tensor is None:
            continue

        try:
            img_pil = TF.to_pil_image(img_tensor)  
        except Exception as e:
            img_np = img_tensor.permute(1,2,0).cpu().numpy()
            img_np = img_np.astype('uint8')
            img_pil = Image.fromarray(img_np)

        instances = item.get('instances', None)
        if instances is None:
            continue

        try:
            gt_boxes = instances.gt_boxes.tensor.cpu().numpy()
            gt_classes = instances.gt_classes.cpu().numpy()
        except Exception as e:
            continue

        h, w = img_pil.size[1], img_pil.size[0]  
        for box, cls in zip(gt_boxes, gt_classes):
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cropped = img_pil.crop((x1, y1, x2, y2))

            if preprocess is not None:
                proc = preprocess(cropped).unsqueeze(0).cpu()  
            else:
                proc = transforms.ToTensor()(cropped).unsqueeze(0).cpu()

            cropped_list.append(proc)
            class_name = coco_id_name_map.get(int(cls), "unknown")
            labels_list.append(class_name)

    if len(cropped_list) == 0:
        return None, None

    cropped_tensor = torch.cat(cropped_list, dim=0)  
    return cropped_tensor, labels_list

def main():
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise ValueError("No GPUs detected. Please run on a multi-GPU machine.")
    print(f"Using {world_size} GPUs for training")
    cv2.setNumThreads(0)

    mp.spawn(distributed_main, args=(world_size,), nprocs=world_size, join=True)

def distributed_main(rank, world_size):
    import os
    import sys
    import json
    import time
    import tempfile

    import torch
    import torch.distributed as dist
    from torch.utils.data import DistributedSampler, DataLoader
    from torch.nn.parallel import DistributedDataParallel as DDP

    try:
        import torch.serialization
        from open_clip.tokenizer import HFTokenizer
        _HAS_OPENCLIP_TOKENIZER = True
    except Exception:
        _HAS_OPENCLIP_TOKENIZER = False

    print(f"Rank {rank} starting.")
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    try:
        proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        if proj_path not in sys.path:
            sys.path.append(proj_path)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(script_dir, "tparams.json")
        with open(params_path, "r") as f:
            params = json.load(f)

        cfg_file = params.get("cfg_file")
        rcnn_weight_dir = params.get("rcnn_weight_dir", None)
        data_split = params.get("data_split", ["train2017"])

        coco_id_name_map = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
            34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
            42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
            52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
            61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
            74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
            82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush',100:'background'
        }

        save_path = os.path.join(script_dir, 'temp_data.pt')

        if rank == 0:
            print("Rank 0: Loading CLIP and building detection dataloader for preprocessing.")
            clip_model, tokenizer, preprocess = load_clip_model(device)

            data_loader = build_detection_train_loader(
                dataset=get_detection_dataset_dicts(names=data_split, filter_empty=False),
                mapper=DatasetMapper(
                    is_train=True,
                    augmentations=[T.ResizeShortestEdge(short_edge_length=800, max_size=1333)],
                    image_format="BGR",
                ),
                total_batch_size=8,
                num_workers=0,
            )

            d_size = 5000 
            truncated_loader = truncate_dataloader(data_loader, d_size)

            cropped_images_list = []
            selected_labels_list = []

            for batch_idx, raw_batch in enumerate(tqdm(truncated_loader, desc="Processing batches", unit="batch")):
                cropped_tensor, labels = extract_gt_cropped_images_and_labels_from_batch(raw_batch, preprocess, coco_id_name_map)
                if cropped_tensor is not None and labels is not None:
                    cropped_images_list.append(cropped_tensor)
                    selected_labels_list.extend(labels)
            
            del clip_model
            torch.cuda.empty_cache()

            if cropped_images_list:
                all_crops = torch.cat(cropped_images_list, dim=0)
                all_labels = selected_labels_list

                print(f"Rank0: Total cropped samples: {len(all_labels)}")

                tmp_fd, tmp_name = tempfile.mkstemp(dir=script_dir, prefix='temp_data_', suffix='.pt')
                os.close(tmp_fd)
                try:
                    torch.save({'images': all_crops, 'labels': all_labels, 'd_size': d_size}, tmp_name)

                    try:
                        fd = os.open(tmp_name, os.O_RDONLY)
                        os.fsync(fd)
                        os.close(fd)
                    except Exception:
                        pass

                    os.replace(tmp_name, save_path)

                    try:
                        dirfd = os.open(script_dir, os.O_RDONLY)
                        os.fsync(dirfd)
                        os.close(dirfd)
                    except Exception:
                        pass

                    print(f"Rank0: Saved preprocessed data to {save_path}")
                finally:
                    if os.path.exists(tmp_name):
                        try:
                            os.remove(tmp_name)
                        except Exception:
                            pass
            else:
                logger.info("No valid cropped images found on rank0. Exiting.")
                dist.barrier()
                return
        else:
            print(f"Rank {rank} loading CLIP model (waiting for rank0 to finish preprocessing)...")
            clip_model, tokenizer, preprocess = load_clip_model(device)

        dist.barrier()

        max_retries = 6
        data_dict = None
        for attempt in range(max_retries):
            try:
                if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
                    raise FileNotFoundError(f"temp data file missing or empty: {save_path}")

                if _HAS_OPENCLIP_TOKENIZER:
                    try:
                        torch.serialization.add_safe_globals([HFTokenizer])
                    except Exception as e:
                        print(f"[Rank {rank}] Warning: Could not add HFTokenizer to safe globals: {e}")

                data_dict = torch.load(save_path, map_location='cpu', weights_only=False)
                break
            except (RuntimeError, EOFError, OSError, FileNotFoundError) as e:
                wait = 1.0 * (2 ** attempt)
                if rank == 0:
                    print(f"Rank {rank}: load attempt {attempt} failed with {e}; retrying in {wait:.1f}s")
                time.sleep(wait)
        else:
            data_dict = torch.load(save_path, map_location='cpu', weights_only=False)

        images_tensor = data_dict['images']
        labels_list = data_dict['labels']
        d_size = data_dict.get('d_size', None)

        dataset = CroppedImageDataset(images_tensor, labels_list, preprocess=None)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        batch_per_gpu = max(1, 4 // world_size) 

        dataloader = DataLoader(
            dataset,
            batch_size=batch_per_gpu,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=lambda x: default_collate_with_filter(x)
        )
        
        if rank == 0 and 'clip_model' not in locals():
             clip_model, tokenizer, preprocess = load_clip_model(device) 

        clip_model = DDP(clip_model, device_ids=[rank])

        print(f"Rank {rank} starting fine-tuning.")
        finetune_clip_model_distributed(clip_model, dataloader, tokenizer, d_size, device, rank, world_size, epochs=5, lr=1e-5, alpha=0.5)

    finally:
        cleanup()





def default_collate_with_filter(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = [b[1] for b in batch]
    return images, labels

def finetune_clip_model_distributed(clip_model, dataloader, tokenizer, d_size, device, rank, world_size, epochs=5, lr=1e-5, alpha=0.5):
    clip_model.train()

    if rank == 0:
        zeroshot_weights = clip_model.module.state_dict()
        os.makedirs("./weights/CLIP_Weights", exist_ok=True)
        zeroshot_save_path = "./weights/CLIP_Weights/zeroshot_clip_model.pth"
        torch.save(zeroshot_weights, zeroshot_save_path)

    for param in clip_model.parameters():
        param.requires_grad = False

    try:
        for name, param in clip_model.module.text.named_parameters():
            if 'ln_final' in name or 'text_projection' in name:
                param.requires_grad = True
    except Exception:
        pass

    try:
        for name, param in clip_model.module.visual.named_parameters():
            if 'ln_post' in name or 'proj' in name:
                param.requires_grad = True
    except Exception:
        pass

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, clip_model.parameters()), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()
    
    scaler = torch.cuda.amp.GradScaler()

    losses = []
    accuracies = []

    for epoch in range(epochs):
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            try:
                dataloader.sampler.set_epoch(epoch)
            except Exception:
                pass

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            images, labels = batch 
            images = images.to(device) 

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                image_features = clip_model.module.encode_image(images) 

                texts = tokenizer(labels).to(device)
                text_features = clip_model.module.encode_text(texts)

                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                logits_per_image = image_features @ text_features.T * clip_model.module.logit_scale.exp()

                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                loss = loss_fn(logits_per_image, ground_truth)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            predictions = torch.argmax(logits_per_image, dim=1)
            correct += (predictions == ground_truth).sum().item()
            total += len(images)

            if batch_idx % 100 == 0 and rank == 0:
                print(f'Train epoch:{epoch} batch:{batch_idx} loss:{loss.item():.4f}')

        epoch_loss = total_loss / (len(dataloader) if len(dataloader)>0 else 1)
        epoch_accuracy = correct / total if total > 0 else 0.0

        if rank == 0:
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)
            print(f'Train epoch:{epoch} Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

            os.makedirs("weights/CLIP_Weights/GT/In_train_weights", exist_ok=True)
            torch.save(clip_model.module.state_dict(), f"weights/CLIP_Weights/GT/In_train_weights/LM_epoch_{epoch}_dsize_{d_size}.pth")
            print(f"Rank0: Model weights saved at epoch {epoch}")

        scheduler.step()
        dist.barrier()

    if rank == 0:
        zeroshot_weights = torch.load("./weights/CLIP_Weights/zeroshot_clip_model.pth")
        finetuned_weights = clip_model.module.state_dict()

        os.makedirs("weights/CLIP_Weights/GT/www", exist_ok=True)
        wise_ft_weights = {key: (1 - alpha) * zeroshot_weights[key] + alpha * finetuned_weights[key] for key in zeroshot_weights.keys()}

        clip_model.module.load_state_dict(wise_ft_weights)
        torch.save(clip_model.module.state_dict(), f"weights/CLIP_Weights/GT/www/LM_epoch_{epochs}_dsize_{d_size}.pth")
        print("Rank0: Final model weights saved with WiSE-FT applied.")

    return clip_model


if __name__ == "__main__":
    main()



















