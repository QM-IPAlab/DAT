import os
import sys
import json
import time
import tempfile
from pathlib import Path
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import open_clip
from detectron2.data import get_detection_dataset_dicts, DatasetMapper, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog

try:
    from scripts.open_vocab_detection.evaluate_method.load_models import load_fully_supervised_trained_model
except Exception:
    def load_fully_supervised_trained_model(cfg_file, weight_dir):
        print('Warning: load_fully_supervised_trained_model is a placeholder. You need to provide the actual implementation if running preprocessing.')
        class MockModel:
            def to(self, device): return self
            def eval(self): return self
        class MockCfg: pass
        return MockModel(), MockCfg()


def setup(rank, world_size, master_port="12311"):
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = master_port
    

    torch.cuda.set_device(rank)

    
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def cleanup():
    if dist.is_available() and dist.is_initialized():
        try:
            pass
        except Exception:
            pass
        dist.destroy_process_group()

def truncate_dataloader(dataloader, num_batches):
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        yield batch

def load_clip_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP', pretrained='webli')
    tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP')
    model = model.to(device)
    return model, tokenizer, preprocess


def extract_gt_crops_from_batch_lvis(raw_batch, preprocess, id_to_name_map):

    crops = []
    labels = []
    for item in raw_batch:
        img_tensor = item.get('image', None)
        if img_tensor is None:
            continue
        
   
        try:
            pil = TF.to_pil_image(img_tensor)
        except Exception:
            img_np = img_tensor.permute(1,2,0).cpu().numpy()
            if img_np.dtype != 'uint8':
                img_np = (img_np * 255).astype('uint8')
            pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        
        instances = item.get('instances', None)
        if instances is None:
            continue
        

        try:
            gt_boxes = instances.gt_boxes.tensor.cpu().numpy()
            gt_classes = instances.gt_classes.cpu().numpy()
        except Exception:
            continue
        
        width, height = pil.size
        for box, cls in zip(gt_boxes, gt_classes):
            x1, y1, x2, y2 = map(int, box.tolist())
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = pil.crop((x1, y1, x2, y2))
            
            
            if preprocess is not None:
                try:
                    t = preprocess(crop).unsqueeze(0).cpu() 
                except Exception:
                    t = transforms.ToTensor()(crop).unsqueeze(0).cpu()
            else:
                t = transforms.ToTensor()(crop).unsqueeze(0).cpu()
            
            crops.append(t)
            
            cls_idx = int(cls)
            class_name = id_to_name_map.get(cls_idx, str(cls_idx))
            labels.append(class_name)

    if len(crops) == 0:
        return None, None
    
    cropped_tensor = torch.cat(crops, dim=0) 
    return cropped_tensor, labels

class CroppedDatasetFromTensors(Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        t = self.tensors[idx]
        if t.dim() == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        return t, self.labels[idx]

def default_collate_with_filter(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = [b[1] for b in batch]
    return images, labels


def finetune_clip_model_distributed(clip_model, dataloader, tokenizer, d_size, device, rank, world_size, epochs=10, lr=1e-5, alpha=0.5, out_dir='weights/CLIP_Weights_LVIS/GT'):
    clip_model.train()
    os.makedirs(out_dir, exist_ok=True)
    
    zeroshot_save_path = os.path.join(out_dir, 'zeroshot_clip_model.pth')
    
    if rank == 0:
        zeroshot_weights = clip_model.module.state_dict() if isinstance(clip_model, DDP) else clip_model.state_dict()
        zeroshot_backup_path = Path('./weights/CLIP_Weights/') / 'zeroshot_clip_model.pth'
        zeroshot_backup_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(zeroshot_weights, zeroshot_save_path)
        if not zeroshot_backup_path.exists():
             torch.save(zeroshot_weights, zeroshot_backup_path)
        print(f"[Rank 0] Zeroshot weights saved to {zeroshot_save_path}")


    for param in clip_model.parameters():
        param.requires_grad = False
    

    clip_model_unwrapped = clip_model.module
    try:
        for name, param in clip_model_unwrapped.text.named_parameters():
            if 'ln_final' in name or 'text_projection' in name:
                param.requires_grad = True
    except Exception:
        pass
    
    try:
        for name, param in clip_model_unwrapped.visual.named_parameters():
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
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        correct = 0
        total = 0

        data_iterator = dataloader
        if rank == 0:
             data_iterator = tqdm(dataloader, desc=f'Epoch {epoch}/{epochs}')
        
        for batch_idx, batch in enumerate(data_iterator):
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
            
            preds = torch.argmax(logits_per_image, dim=1)
            correct += (preds == ground_truth).sum().item()
            total += len(images)
            
            if batch_idx % 100 == 0 and rank == 0:
                print(f'\n[Rank {rank}] Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}')


            del images, texts, image_features, text_features, logits_per_image, ground_truth
            torch.cuda.empty_cache()
        

        metrics = torch.tensor([total_loss, correct, total], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        global_total_loss = metrics[0].item()
        global_correct = metrics[1].item()
        global_total = metrics[2].item()
        

        num_batches = len(dataloader) 
        epoch_loss = global_total_loss / (num_batches * world_size if num_batches > 0 else 1)
        epoch_acc = global_correct / global_total if global_total > 0 else 0.0

        if rank == 0:
            losses.append(epoch_loss)
            accuracies.append(epoch_acc)
            print(f'[Rank {rank}] Epoch {epoch} Loss {epoch_loss:.4f} Acc {epoch_acc:.4f}')
            

            save_p = os.path.join(out_dir, 'In_Train', f'LM_epoch_{epoch}_dsize_{d_size}.pth')
            os.makedirs(os.path.dirname(save_p), exist_ok=True)
            torch.save(clip_model.module.state_dict(), save_p)
            
        scheduler.step()
        dist.barrier() 


    if rank == 0:
        try:

            if os.path.exists("./weights/CLIP_Weights/zeroshot_clip_model.pth"):
                zeroshot_weights = torch.load("./weights/CLIP_Weights/zeroshot_clip_model.pth", map_location='cpu')
            else:
                 zeroshot_weights = torch.load(zeroshot_save_path, map_location='cpu')
        except FileNotFoundError:
            print("Error: Zeroshot weights file not found for WiSE-FT.")
            return clip_model

        finetuned_weights = clip_model.module.state_dict()
        zeroshot_weights = {k: v.cpu() for k, v in zeroshot_weights.items()}
        finetuned_weights = {k: v.cpu() for k, v in finetuned_weights.items()}
        
        wise_ft_out_dir = os.path.join(out_dir, "www")
        os.makedirs(wise_ft_out_dir, exist_ok=True)
        
        wise_ft_weights = {k: (1 - alpha) * zeroshot_weights[k] + alpha * finetuned_weights[k]
                            for k in zeroshot_weights.keys() if k in finetuned_weights}
        

        temp_model = clip_model.module
        temp_model.load_state_dict(wise_ft_weights, strict=False)
        final_save_path = os.path.join(wise_ft_out_dir, f'LM_epoch_{epochs}_dsize_{d_size}.pth')
        torch.save(temp_model.state_dict(), final_save_path)
        print(f"Rank0: Final model weights saved with WiSE-FT applied to {final_save_path}")


            
    return clip_model


def distributed_main(rank, world_size):
    print(f'[Rank {rank}] starting')
    

    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    sys.path.append(proj_path)


    script_dir = os.path.dirname(os.path.abspath(__file__))
    tpath = os.path.join(script_dir, 'tparams.json')
    ppath = os.path.join(script_dir, 'params.json')

    params_path = None
    if os.path.exists(tpath):
        params_path = tpath
    elif os.path.exists(ppath):
        params_path = ppath
    
    if params_path is None:
        if rank == 0:
            raise FileNotFoundError('找不到 tparams.json 或 params.json，请在脚本目录放置参数文件')
        else:
            dist.barrier()
            return
    
    with open(params_path, 'r') as f:
        params = json.load(f)

    cfg_file = params.get('cfg_file')
    rcnn_weight_dir = params.get('rcnn_weight_dir')
    lvis_data_split = params.get('lvis_data_split')
    d_size = params.get('d_size', 5000)
    total_batch_size = params.get('total_batch_size', 16)
    epochs = params.get('epochs', 5)
    lr = params.get('lr', 1e-5)
    alpha = params.get('alpha', 0.5)
    
    save_path = os.path.join(script_dir, 'temp_data_lvis_gt.pt')

    try:
        if rank == 0:
            print('[Rank 0] Loading models for GT cropping...')
            clip_model_local, tokenizer_local, preprocess_local = load_clip_model(device)
            rcnn_model, cfg = load_fully_supervised_trained_model(cfg_file, rcnn_weight_dir)
            

            rcnn_model.to(device)
            rcnn_model.eval()
            
            meta = MetadataCatalog.get(lvis_data_split)
            thing_classes = getattr(meta, 'thing_classes', None)
            if thing_classes is None:
                raise RuntimeError(f'无法从 MetadataCatalog 获取 thing_classes: {lvis_data_split} 是否已注册?')
            
            id_to_name_map = {i: name for i, name in enumerate(thing_classes)}


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
            all_crops = []
            all_labels = []

            print(f'[Rank 0] Cropping up to {d_size} batches from {lvis_data_split}...')
            for batch_idx, raw_batch in enumerate(tqdm(truncated_loader, desc='GT cropping')):

                crops_tensor, labels = extract_gt_crops_from_batch_lvis(raw_batch, preprocess_local, id_to_name_map)
                
                if crops_tensor is None or labels is None:
                    del raw_batch
                    torch.cuda.empty_cache()
                    continue
                
                all_crops.append(crops_tensor)
                all_labels.extend(labels)
                
                del raw_batch, crops_tensor, labels
                torch.cuda.empty_cache()

            if len(all_crops) == 0:
                raise RuntimeError('Rank0: No GT crops extracted. Check dataset and annotations.')

            all_crops_tensor = torch.cat(all_crops, dim=0)
            print(f'[Rank 0] Total cropped samples: {all_crops_tensor.shape[0]}')


            tmp_fd, tmp_name = tempfile.mkstemp(dir=script_dir, prefix='temp_data_lvis_gt_', suffix='.pt')
            os.close(tmp_fd)
            try:
                torch.save({'images': all_crops_tensor, 'labels': all_labels, 'd_size': d_size}, tmp_name)
                os.replace(tmp_name, save_path)
                print(f'[Rank 0] Saved preprocessed GT data to {save_path}')
            finally:
                if os.path.exists(tmp_name):
                    try:
                        os.remove(tmp_name)
                    except Exception:
                        pass


            del clip_model_local, rcnn_model, preprocess_local, tokenizer_local, all_crops_tensor, all_crops, all_labels
            torch.cuda.empty_cache()
        
        else:
            print(f'[Rank {rank}] waiting for rank0 to ensure GT preprocessing is finished...')
        

        if rank != 0:
            max_wait_time = 3600 
            check_interval = 30 
            start_time = time.time()
            

            while not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError(f"Rank {rank} timed out waiting for data file {save_path}. Rank 0 might have failed.")
                time.sleep(check_interval)
            print(f'[Rank {rank}] Data file found. Proceeding to load.')
            
        dist.barrier() 

        
        clip_model, tokenizer, preprocess = load_clip_model(device)
        

        max_retries = 6
        data_dict = None
        for attempt in range(max_retries):
            try:
                if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
                    raise FileNotFoundError(f'temp data missing or empty: {save_path}')

                data_dict = torch.load(save_path, map_location='cpu') 
                break
            except Exception as e:
                wait = 1.0 * (2 ** attempt)
                if rank == 0:
                    print(f'[Rank {rank}] load attempt {attempt+1} failed: {e}; retry {wait}s')
                time.sleep(wait)
        
        if data_dict is None:
             raise RuntimeError(f"Rank {rank}: Failed to load data from {save_path} after {max_retries} attempts.")
            
        images_tensor = data_dict['images']
        labels_list = data_dict['labels']
        d_size = data_dict.get('d_size', d_size)
        
        dataset = CroppedDatasetFromTensors(images_tensor, labels_list)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        batch_per_gpu = max(1, (total_batch_size // world_size))
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_per_gpu, 
            sampler=sampler, 
            num_workers=0, 
            pin_memory=True, 
            collate_fn=default_collate_with_filter
        )

        
        clip_model = DDP(clip_model, device_ids=[rank], find_unused_parameters=True) 

        
        finetune_clip_model_distributed(
            clip_model, 
            dataloader, 
            tokenizer, 
            d_size, 
            device, 
            rank, 
            world_size, 
            epochs=epochs, 
            lr=lr, 
            alpha=alpha, 
            out_dir=params.get('out_weights_dir', 'weights/CLIP_Weights_LVIS/GT')
        )
        
    except Exception as e:
        if rank == 0:
             print(f"FATAL ERROR in Rank {rank}: {e}", file=sys.stderr)

        try:
            dist.barrier() 
        except Exception:
            pass
        raise 
        
    finally:
        cleanup()

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError('No GPUs found for distributed training')
    
    print(f'Launching LVIS GT CLIP fine-tuning on {world_size} GPUs...')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir) 
    mp.spawn(distributed_main, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()