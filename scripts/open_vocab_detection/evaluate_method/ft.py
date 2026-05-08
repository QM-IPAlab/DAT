import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  
import sys
import cv2
import json
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm
import open_clip
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader, get_detection_dataset_dicts, DatasetMapper
from scripts.open_vocab_detection.evaluate_method.utils import finetune_clip_model 
from scripts.open_vocab_detection.evaluate_method.load_models import load_fully_supervised_trained_model,load_clip_model
import detectron2.data.transforms as T
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torchvision.transforms.functional as TF
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



def setup(rank, world_size):
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12363'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    
    dist.destroy_process_group()

def truncate_dataloader(dataloader, num_batches):
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        yield batch
        

def save_detected_objects(images, targets, batch_idx):

    for img_idx, (image, target) in enumerate(zip(images, targets)):
        image_pil = F.to_pil_image(image.cpu())  
        boxes = target["boxes"].detach().cpu().numpy()  
        labels = target["labels"].detach().cpu().numpy()

        for obj_idx, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = map(int, box)


            cropped_obj = image_pil.crop((x1, y1, x2, y2))


            image_filename = f"cropped_dataset/images/{batch_idx}_{img_idx}_{obj_idx}.jpg"
            cropped_obj.save(image_filename)


            label_filename = f"cropped_dataset/labels/{batch_idx}_{img_idx}_{obj_idx}.txt"
            with open(label_filename, "w") as f:
                f.write(str(label)) 

            print(f"Saved: {image_filename}, Label: {label_filename}")


def extract_cropped_images_and_labels(inputs, rcnn_model, device, preprocess, coco_id_name_map, batch_idx, output_dir="cropped_dataset"):
    rcnn_model.eval()
    with torch.no_grad():
        outputs = rcnn_model(inputs)


    rcnn_boxes = outputs[0]["instances"].pred_boxes.tensor.to("cuda") if len(outputs[0]["instances"]) > 0 else torch.tensor([])
    rcnn_classes = outputs[0]["instances"].pred_classes.to("cuda") if len(outputs[0]["instances"]) > 0 else torch.tensor([])


    if rcnn_classes.numel() == 0:
        print("No objects detected, skipping image.")
        return None, None

    img = inputs[0]["image"].to(device)
    new_height, new_width = img.shape[1], img.shape[2]
    object_crops_list = []
    selected_idx = []

    for bbox_idx, bbox in enumerate(rcnn_boxes):
        x1, y1, x2, y2 = bbox.int()


        x1 = max(0, int(x1 * new_width / inputs[0]['width']))
        x2 = min(new_width, int(x2 * new_width / inputs[0]['width']))
        y1 = max(0, int(y1 * new_height / inputs[0]['height']))
        y2 = min(new_height, int(y2 * new_height / inputs[0]['height']))

        if x2 > x1 and y2 > y1:
            cropped_image = img[:, y1:y2, x1:x2]
            cropped_img_arr = cropped_image.permute(1, 2, 0).cpu().numpy()

            if cropped_img_arr.shape[0] > 0 and cropped_img_arr.shape[1] > 0:
                if cropped_img_arr.shape[2] == 3:
                    cropped_img_arr = cv2.cvtColor(cropped_img_arr, cv2.COLOR_BGR2RGB)  
                img_pil = Image.fromarray(cropped_img_arr)
                image = preprocess(img_pil).unsqueeze(0).to(device)

                object_crops_list.append(image)
                
                class_name = coco_id_name_map.get(rcnn_classes[bbox_idx].item())
                selected_idx.append(class_name)

    if object_crops_list:
        cropped_img_arr = torch.cat(object_crops_list, dim=0)
        return cropped_img_arr, selected_idx
    else:
        return None, None
        


def custom_collate_fn(batch):
    
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    if not batch:
        return None  

    
    return torch.utils.data.dataloader.default_collate(batch)




def load_clip_model(device):
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP', pretrained='webli')
    tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP')
    clip_model = clip_model.to(device)
    return clip_model, tokenizer, preprocess



class CroppedImageDataset(Dataset):
    def __init__(self, images, labels, preprocess):
        self.images = images
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        while True:
            image = self.images[idx]
            label = self.labels[idx]

            
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)

            
            image = self.preprocess(image)

            
            if label is None:
                
                idx = (idx + 1) % len(self.images)
                continue

            return image, label


def main():

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training")
    cv2.setNumThreads(0)      
    
    mp.spawn(
        distributed_main,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


def distributed_main(rank, world_size):
    print(f"Running distributed training on rank {rank}.")
    setup(rank, world_size)
    

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    try:
        proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        sys.path.append(proj_path)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(script_dir, "tparams.json")
        with open(params_path, "r") as f:
            params = json.load(f)

        cfg_file = params["cfg_file"]
        rcnn_weight_dir = params["rcnn_weight_dir"]
        data_split = params["data_split"]
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


        if rank == 0:
            print("Loading CLIP and Faster R-CNN on rank 0...")
            clip_model, tokenizer, preprocess = load_clip_model(device)
            
            
            rcnn_model, cfg = load_fully_supervised_trained_model(cfg_file, rcnn_weight_dir)
            rcnn_model.to(device)
            rcnn_model.eval()


            data_loader = build_detection_train_loader(
                dataset=get_detection_dataset_dicts(names=data_split, filter_empty=False),
                mapper=DatasetMapper(
                    is_train=True,
                    augmentations=[T.ResizeShortestEdge(short_edge_length=800, max_size=1333)],
                    image_format="BGR",
                ),
                total_batch_size=16,
                num_workers=0,
            )

            d_size = 1000
            truncated_loader = truncate_dataloader(data_loader, d_size)


            img_dir = "cropped_dataset/images"
            label_dir = "cropped_dataset/labels"
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            print("Start extracting and saving cropped objects...")
            print("Datasize = ",d_size)
            for batch_idx, batch in enumerate(tqdm(truncated_loader, desc="Processing batches", unit="batch")):
                inputs = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()} for item in batch]

                cropped_images, selected_idx = extract_cropped_images_and_labels(
                    inputs, rcnn_model, device, preprocess, coco_id_name_map, batch_idx
                )

                if cropped_images is not None and selected_idx is not None:
                    for img_idx, (image, label) in enumerate(zip(cropped_images, selected_idx)):
                        image_pil = TF.to_pil_image(image.cpu())
                        image_filename = os.path.join(img_dir, f"{batch_idx}_{img_idx}.jpg")
                        label_filename = os.path.join(label_dir, f"{batch_idx}_{img_idx}.txt")
                        image_pil.save(image_filename)
                        with open(label_filename, "w") as f:
                            f.write(str(label))

                torch.cuda.empty_cache()  

            print("All cropped images saved to disk.")
            
            torch.save({
                'tokenizer': tokenizer,
                'd_size': d_size
            }, 'temp_data.pt')

        else:
            print(f"Rank {rank} waiting for rank 0 to finish data preprocessing...")

        dist.barrier()


        print(f"Rank {rank} loading CLIP model and dataset...")
        clip_model, tokenizer, preprocess = load_clip_model(device)

        shared_data = torch.load('temp_data.pt', map_location='cpu', weights_only=False)
        tokenizer = shared_data['tokenizer']
        d_size = shared_data['d_size']


        image_files = sorted(os.listdir("cropped_dataset/images"))
        label_files = sorted(os.listdir("cropped_dataset/labels"))

        images, labels = [], []
        for img_file, lbl_file in zip(image_files, label_files):
            img_path = os.path.join("cropped_dataset/images", img_file)
            lbl_path = os.path.join("cropped_dataset/labels", lbl_file)
            with open(lbl_path, "r") as f:
                lbl = f.read().strip()
            images.append(Image.open(img_path).convert("RGB"))
            labels.append(lbl)

        dataset = CroppedImageDataset(images, labels, preprocess)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

        dataloader = DataLoader(
            dataset,
            batch_size=16 // world_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

        clip_model = DDP(clip_model, device_ids=[rank])


        print(f"Rank {rank} starting fine-tuning...")
        finetune_clip_model_distributed(
            clip_model, dataloader, tokenizer, d_size, device,
            rank, world_size, epochs=5, lr=1e-5, alpha=0.5
        )

    finally:
        cleanup()



def finetune_clip_model_distributed(clip_model, dataloader, tokenizer, d_size, device, rank, world_size, epochs=10, lr=1e-5, alpha=0.5):
    clip_model.train()
    
    
    if rank == 0:
        zeroshot_weights = clip_model.module.state_dict()
        zeroshot_save_path = "./weights/CLIP_Weights/zeroshot_clip_model.pth"
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
        
        dataloader.sampler.set_epoch(epoch)
        
        total_loss = 0
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
            
            logits_per_image = image_features @ text_features.T * clip_model.module.logit_scale.exp()

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            loss = loss_fn(logits_per_image, ground_truth)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            predictions = torch.argmax(logits_per_image, dim=1)
            correct += (predictions == ground_truth).sum().item()
            total += len(images)

            if batch_idx % 100 == 0 and rank == 0:
                print(f'Train epoch:{epoch} batch:{batch_idx} loss:{loss.item():.4f}')


        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct / total
        
        
        if rank == 0:
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)

            print(f'Train epoch:{epoch} Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
            

            torch.save(clip_model.module.state_dict(), f"weights/CLIP_Weights/In_train_weights/LM_epoch_{epoch}_dsize_{d_size}.pth")
            print(f"Model weights saved at epoch {epoch}")

        scheduler.step()
        

        dist.barrier()


    if rank == 0:

        zeroshot_weights = torch.load("./weights/CLIP_Weights/zeroshot_clip_model.pth")
        finetuned_weights = clip_model.module.state_dict()
        
        wise_ft_weights = {
            key: (1 - alpha) * zeroshot_weights[key] + alpha * finetuned_weights[key]
            for key in zeroshot_weights.keys()
        }

        clip_model.module.load_state_dict(wise_ft_weights)
        torch.save(clip_model.module.state_dict(), f"weights/CLIP_Weights/www/LM_epoch_{epochs}_dsize_{d_size}.pth")
        print("Final model weights saved with WiSE-FT applied.")

    return clip_model


if __name__ == "__main__":
    main()