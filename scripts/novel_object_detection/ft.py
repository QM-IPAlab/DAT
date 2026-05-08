import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
import sys
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import open_clip
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2

from detectron2.data import get_detection_dataset_dicts, DatasetMapper
from detectron2.data import build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from scripts.open_vocab_detection.evaluate_method.load_models import load_fully_supervised_trained_model


def setup(rank, world_size, master_port="12359"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()


def truncate_dataloader(dataloader, num_batches):
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        yield batch



def create_clip_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP', pretrained='webli')
    tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP')
    model = model.to(device)
    return model, tokenizer, preprocess



class CroppedImageDataset(Dataset):
    def __init__(self, image_paths, labels, preprocess):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.preprocess(img)
        label = self.labels[idx]
        return img, label



def extract_cropped_images_and_labels(inputs, rcnn_model, device, preprocess, id_name_map):
    rcnn_model.eval()
    with torch.no_grad():
        outputs = rcnn_model(inputs)

    if len(outputs) == 0 or len(outputs[0]["instances"]) == 0:
        return None, None


    inst = outputs[0]["instances"]
    boxes = inst.pred_boxes.tensor.cpu()
    classes = inst.pred_classes.cpu()


    width = inputs[0]["width"]
    height = inputs[0]["height"]

    img_tensor = inputs[0]["image"].cpu()

    img_np = img_tensor.permute(1,2,0).numpy()

    if img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    img_pil = Image.fromarray(img_np)

    crops = []
    labels = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.tolist())
        x1 = max(0, min(x1, width-1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height-1))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img_pil.crop((x1, y1, x2, y2))
        crop_t = preprocess(crop).unsqueeze(0).to(device)
        crops.append(crop_t)
        class_id = int(classes[i].item())
        class_name = id_name_map.get(class_id, str(class_id))
        labels.append(class_name)

    if len(crops) == 0:
        return None, None

    crops_tensor = torch.cat(crops, dim=0)
    return crops_tensor, labels




def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None and item[0] is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)



def finetune_clip_model_distributed(clip_model, dataloader, tokenizer, d_size, device, rank, world_size, epochs=10, lr=1e-5, alpha=0.5, out_dir="weights/CLIP_Weights_LVIS"):
    clip_model.train()

    os.makedirs(out_dir, exist_ok=True)
    zeroshot_save_path = os.path.join(out_dir, "zeroshot_clip_model.pth")

    if rank == 0:
        zeroshot_weights = clip_model.module.state_dict() if isinstance(clip_model, DDP) else clip_model.state_dict()
        torch.save(zeroshot_weights, zeroshot_save_path)


    for param in clip_model.parameters():
        param.requires_grad = False


    for name, param in clip_model.module.text.named_parameters():
        if 'ln_final' in name or 'text_projection' in name:
            param.requires_grad = True

    for name, param in clip_model.module.visual.named_parameters():
        if 'ln_post' in name or 'proj' in name:
            param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, clip_model.parameters()), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    accuracies = []

    for epoch in range(epochs):
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue
            images, labels = batch
            images = images.to(device)

            image_features = clip_model.module.encode_image(images)
            texts = tokenizer(labels).to(device)
            text_features = clip_model.module.encode_text(texts)

            optimizer.zero_grad()

            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            logits_per_image = image_features @ text_features.T * clip_model.module.logit_scale.exp()

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            loss = loss_fn(logits_per_image, ground_truth)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits_per_image, dim=1)
            correct += (preds == ground_truth).sum().item()
            total += len(images)

            if batch_idx % 100 == 0 and rank == 0:
                print(f"[Rank {rank}] Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")

            torch.cuda.empty_cache()

        
        epoch_loss = total_loss / (len(dataloader) if len(dataloader) > 0 else 1)
        epoch_acc = correct / total if total > 0 else 0.0

        if rank == 0:
            losses.append(epoch_loss)
            accuracies.append(epoch_acc)
            print(f"[Rank {rank}] Epoch {epoch} Loss {epoch_loss:.4f} Acc {epoch_acc:.4f}")
            save_p = os.path.join(out_dir, f"In_Train/LM_epoch_{epoch}_dsize_{d_size}.pth")
            torch.save(clip_model.module.state_dict(), save_p)

        scheduler.step()
        dist.barrier()

    
    if rank == 0:
        zeroshot_weights = torch.load("./weights/CLIP_Weights/zeroshot_clip_model.pth")
        finetuned_weights = clip_model.module.state_dict()


        zeroshot_weights = {k: v.cpu() for k, v in zeroshot_weights.items()}
        finetuned_weights = {k: v.cpu() for k, v in finetuned_weights.items()}

        os.makedirs("weights/CLIP_Weights_LVIS/www", exist_ok=True)
        wise_ft_weights = {k: (1 - alpha) * zeroshot_weights[k] + alpha * finetuned_weights[k]
                        for k in zeroshot_weights.keys() if k in finetuned_weights}

        clip_model.module.load_state_dict(wise_ft_weights, strict=False)
        torch.save(clip_model.module.state_dict(),
                f"weights/CLIP_Weights_LVIS/www/LM_epoch_{epochs}_dsize_{d_size}.pth")
        print("Final model weights saved with WiSE-FT applied.")

    return clip_model




def distributed_main(rank, world_size):
    print(f"[Rank {rank}] starting")
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(script_dir, "tparams.json")
    with open(params_path, 'r') as f:
        params = json.load(f)

    cfg_file = params.get('cfg_file')
    rcnn_weight_dir = params.get('rcnn_weight_dir')
    lvis_data_split = params.get('lvis_data_split')
    d_size = params.get('d_size', 5000)
    total_batch_size = params.get('total_batch_size', 16)

    out_img_dir = params.get('out_img_dir', 'cropped_dataset_lvis/images')
    out_label_dir = params.get('out_label_dir', 'cropped_dataset_lvis/labels')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    try:
        
        if rank == 0:
            print('[Rank 0] Loading models for cropping...')
            clip_model_local, tokenizer_local, preprocess_local = create_clip_model(device)

            rcnn_model, cfg = load_fully_supervised_trained_model(cfg_file, rcnn_weight_dir)
            rcnn_model.to(device)
            rcnn_model.eval()

            
            meta = MetadataCatalog.get(lvis_data_split)
            thing_classes = getattr(meta, 'thing_classes', None)
            if thing_classes is None:
                raise RuntimeError(f"Cannot find thing_classes for dataset {lvis_data_split}. Make sure the dataset is registered.")
            lvis_id_name_map = {i: name for i, name in enumerate(thing_classes)}

            
            data_loader = build_detection_train_loader(
                dataset=get_detection_dataset_dicts(names=lvis_data_split, filter_empty=False),
                mapper=DatasetMapper(
                    is_train=True,
                    augmentations=[T.ResizeShortestEdge(short_edge_length=800, max_size=1333)],
                    image_format='BGR'
                ),
                total_batch_size=total_batch_size,
                num_workers=0,
            )

            truncated_loader = truncate_dataloader(data_loader, d_size)

            print(f"[Rank 0] Start cropping {d_size} batches from {lvis_data_split}...")
            for batch_idx, batch in enumerate(tqdm(truncated_loader, desc='Crop batches')):
                
                inputs = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()} for item in batch]

                crops_tensor, crop_labels = extract_cropped_images_and_labels(inputs, rcnn_model, device, preprocess_local, lvis_id_name_map)
                if crops_tensor is None or crop_labels is None:
                    torch.cuda.empty_cache()
                    continue

               
                for img_idx in range(crops_tensor.shape[0]):
                    im_t = crops_tensor[img_idx].cpu()
                    im_pil = TF.to_pil_image(im_t)
                    img_filename = os.path.join(out_img_dir, f"{batch_idx}_{img_idx}.jpg")
                    label_filename = os.path.join(out_label_dir, f"{batch_idx}_{img_idx}.txt")
                    im_pil.save(img_filename)
                    with open(label_filename, 'w') as f:
                        f.write(str(crop_labels[img_idx]))

                
                torch.cuda.empty_cache()

            print('[Rank 0] Cropping finished.')

            
            torch.save({'d_size': d_size}, 'temp_data_lvis.pt')

        else:
            print(f"[Rank {rank}] waiting for rank 0 to finish cropping...")

        
        dist.barrier()

        
        clip_model, tokenizer, preprocess = create_clip_model(device)

        shared = torch.load('temp_data_lvis.pt', map_location='cpu')
        d_size = shared['d_size']

        
        img_dir = out_img_dir
        label_dir = out_label_dir
        image_files = sorted(os.listdir(img_dir))
        label_files = sorted(os.listdir(label_dir))

        image_paths = [os.path.join(img_dir, f) for f in image_files]
        labels = []
        for lf in label_files:
            with open(os.path.join(label_dir, lf), 'r') as f:
                labels.append(f.read().strip())

        
        if len(image_paths) == 0:
            raise RuntimeError('No cropped images found. Did rank 0 produce any crops?')

        dataset = CroppedImageDataset(image_paths, labels, preprocess)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        bs = max(1, total_batch_size // world_size)
        dataloader = DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)

        
        clip_model = DDP(clip_model, device_ids=[rank])

        
        finetune_clip_model_distributed(clip_model, dataloader, tokenizer, d_size, device, rank, world_size, epochs=params.get('epochs', 5), lr=params.get('lr', 1e-5), alpha=params.get('alpha', 0.5), out_dir=params.get('out_weights_dir', 'weights/CLIP_Weights_LVIS'))

    finally:
        cleanup()


def main():
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError('No GPUs available for distributed training')
    print(f'Launching LVIS CLIP fine-tuning on {world_size} GPUs...')
    mp.spawn(distributed_main, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
