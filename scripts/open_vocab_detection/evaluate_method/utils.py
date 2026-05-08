import numpy as np
import matplotlib.colors as mcolors
import torch
import torch.nn.functional as F
import os
import matplotlib as mpl
import random
import torch.distributed as dist

from detectron2.data import MetadataCatalog
from torch import nn
from detectron2.utils.visualizer import _create_text_labels, Visualizer
from typing import List

from detectron2.utils.visualizer import _create_text_labels, Visualizer
from scripts.open_vocab_detection.coco_eval_utils.coco_ovd_split import categories_seen, categories_unseen

class BBoxVisualizer(Visualizer):
    colors = list(mcolors.BASE_COLORS.keys())
    
    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): an :class:`Instances` object with fields "pred_boxes", "pred_classes", and "scores"

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes.tensor
        scores = predictions.scores
        classes = predictions.pred_classes
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        for idx, (box, label) in enumerate(zip(boxes, labels)):
            color = self.colors[idx % len(self.colors)]
            box = box.cpu().detach().numpy()
            x0, y0, x1, y1 = box
            self.draw_box((x0, y0, x1, y1), edge_color = color, linewidth = 4.0)
            # Draw label at the top-left corner of the bounding box
            self.draw_text(label, (x0, y0), horizontal_alignment="left", color = color, font_size = 12)
        return self.output

    def draw_box(self, box_coord, linewidth, alpha=0.5, edge_color="g", line_style="-"):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

def build_captions_and_token_span(cat_list, force_lowercase):
    """
    Return:
        captions: str
        cat2tokenspan: dict
            {
                'dog': [[0, 2]],
                ...
            }
    """

    cat2tokenspan = {}
    captions = ""
    for catname in cat_list:
        class_name = catname
        if force_lowercase:
            class_name = class_name.lower()
        if "/" in class_name:
            class_name_list: List = class_name.strip().split("/")
            class_name_list.append(class_name)
            class_name: str = random.choice(class_name_list)

        tokens_positive_i = []
        subnamelist = [i.strip() for i in class_name.strip().split(" ")]
        for subname in subnamelist:
            if len(subname) == 0:
                continue
            if len(captions) > 0:
                captions = captions + " "
            strat_idx = len(captions)
            end_idx = strat_idx + len(subname)
            tokens_positive_i.append([strat_idx, end_idx])
            captions = captions + subname

        if len(tokens_positive_i) > 0:
            captions = captions + " ."
            cat2tokenspan[class_name] = tokens_positive_i

    return captions, cat2tokenspan

def create_positive_map_from_span(tokenized, token_span, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j
    Input:
        - tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        - token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
    for j, tok_list in enumerate(token_span):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            if os.environ.get("SHILONG_DEBUG_ONLY_ONE_POS", None) == "TRUE":
                positive_map[j, beg_pos] = 1
                break
            else:
                positive_map[j, beg_pos : end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def get_ovd_id_to_coco_id():
    seen_names = [x['name'] for x in categories_seen]
    unseen_names = [x['name'] for x in categories_unseen]

    coco_ovd_classes = seen_names + unseen_names

    all_coco_classes = MetadataCatalog.get("coco_2017_val").get("thing_classes")
    ovd_id_to_coco_id = {}

    for i, coco_class in enumerate(all_coco_classes):
        if coco_class in coco_ovd_classes:
            ovd_id_to_coco_id[coco_ovd_classes.index(coco_class)] = i

    return ovd_id_to_coco_id

def get_text_prompt_for_g_dino(tokenizer):
    seen_names = [x['name'] for x in categories_seen]
    unseen_names = [x['name'] for x in categories_unseen]

    coco_ovd_classes = seen_names + unseen_names

    coco_ovd_classes = [i.lower() for i in coco_ovd_classes]
    coco_ovd_classes = [s.replace("_", " ") for s in coco_ovd_classes] # replace _ with space

    captions, cat2tokenspan = build_captions_and_token_span(coco_ovd_classes, True)
    tokenspanlist = [cat2tokenspan[cat] for cat in coco_ovd_classes]
    positive_map = create_positive_map_from_span(tokenizer(captions), tokenspanlist) # shape: (num_categories, 256)

    return captions, positive_map

def get_clip_preds(img, clip_model, text_features):
    """
    img: torch.Size([N, 3, 224, 224])
    text_features: torch.Size([768, 1203])
    """
    with torch.no_grad(), torch.cuda.amp.autocast():
        img_features = clip_model.encode_image(img) # features shape: torch.Size([50, N, 768])
        img_features = F.normalize(img_features, dim=-1) # features shape: torch.Size([50, N, 768])

        text_probs = torch.sigmoid(img_features @ text_features.T * clip_model.logit_scale.exp() + clip_model.logit_bias) # shape: torch.Size([N, 1203])

        values, indices = text_probs.topk(1)
    
    return values, indices

def article(name):
  return 'an' if name[0] in 'aeiou' else 'a'

def processed_name(name, rm_dot=False):
  # _ for lvis
  # / for obj365
  res = name.replace('_', ' ').replace('/', ' or ').lower()
  if rm_dot:
    res = res.rstrip('.')
  return res



def finetune_clip_model(clip_model, dataloader,tokenizer, d_size, device, epochs=10, lr=1e-5, alpha=0.5):
    clip_model.train()
    zeroshot_weights = clip_model.state_dict()
    zeroshot_save_path = "./weights/CLIP_Weights/zeroshot_clip_model.pth"
    torch.save(zeroshot_weights, zeroshot_save_path)

    for param in clip_model.parameters():
        param.requires_grad = False

    for name, param in clip_model.text.named_parameters():
        if 'ln_final' in name or 'text_projection' in name:
            param.requires_grad = True

    for name, param in clip_model.visual.named_parameters():
        if 'ln_post' in name or 'proj' in name:
            param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, clip_model.parameters()), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    accuracies = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            image_features = clip_model.encode_image(images)

            # 将类别名称（字符串）转换为文本嵌入
            texts = tokenizer(labels).to(device)  
            text_features = clip_model.encode_text(texts)

            optimizer.zero_grad()
 
            image_features = F.normalize(image_features, dim=-1)
            

            logits_per_image = image_features @ text_features.T * clip_model.logit_scale.exp()

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)  # 对角线匹配
            loss = loss_fn(logits_per_image, ground_truth)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            predictions = torch.argmax(logits_per_image, dim=1)
            correct += (predictions == ground_truth).sum().item()
            total += len(images)

            if batch_idx % 100 == 0:
                print(f'Train epoch:{epoch} batch:{batch_idx} loss:{loss.item():.4f}')

        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        print(f'Train epoch:{epoch} Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        scheduler.step()

        torch.save(clip_model.state_dict(), f"weights/CLIP_Weights/In_train_weights/LM_epoch_{epochs}_dsize_{d_size}.pth")
        print(f"Model weights saved at epoch {epoch}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), accuracies, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.legend()

    if d_size is not None:
        plt.savefig(f"Training_Results/LM_epoch_{epochs}_dsize_{d_size}.png")
    else:
        plt.savefig(f"Training_Results/training_curve.png")
    print(f"Training curve saved as training_curve_epoch_{epochs}_dsize_{d_size}.png")

    finetuned_weights = clip_model.state_dict()
    wise_ft_weights = {
        key: (1 - alpha) * zeroshot_weights[key] + alpha * finetuned_weights[key]
        for key in zeroshot_weights.keys()
    }

    clip_model.load_state_dict(wise_ft_weights)
    torch.save(clip_model.state_dict(), f"weights/CLIP_Weights/www/LM_epoch_{epochs}_dsize_{d_size}.pth")
    print("Final model weights saved with WiSE-FT applied.")

    return clip_model









def dis_finetune_clip_model(clip_model, dataloader, tokenizer, d_size, device, epochs=10, lr=1e-5, alpha=0.5):
    # 检查是否是 DDP 环境
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0

    clip_model.train()
    zeroshot_weights = clip_model.module.state_dict() if hasattr(clip_model, "module") else clip_model.state_dict()

    if rank == 0:
        zeroshot_save_path = "./weights/CLIP_Weights/zeroshot_clip_model.pth"
        torch.save(zeroshot_weights, zeroshot_save_path)

    # 仅 rank=0 打印
    if rank == 0:
        print("Starting CLIP fine-tuning ...")

    # ========== 冻结部分参数 ==========
    for param in clip_model.parameters():
        param.requires_grad = False

    model_to_train = clip_model.module if hasattr(clip_model, "module") else clip_model

    for name, param in model_to_train.text.named_parameters():
        if 'ln_final' in name or 'text_projection' in name:
            param.requires_grad = True
    for name, param in model_to_train.visual.named_parameters():
        if 'ln_post' in name or 'proj' in name:
            param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, clip_model.parameters()),
        lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = torch.tensor(0.0, device=device)
        correct = 0
        total = 0

        dataloader.sampler.set_epoch(epoch) if distributed else None

        for images, labels in dataloader:
            images = images.to(device)
            texts = tokenizer(labels).to(device)

            image_features = clip_model(images, None, encode_image=True)
            text_features = clip_model(None, texts, encode_text=True)

            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)

            logits = image_features @ text_features.T * clip_model.module.logit_scale.exp() if hasattr(clip_model, "module") else image_features @ text_features.T * clip_model.logit_scale.exp()
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            loss = loss_fn(logits, ground_truth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach()

        # 同步 loss across GPUs
        if distributed:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss / world_size

        if rank == 0:
            print(f"Epoch {epoch}: loss={total_loss.item():.4f}")

        scheduler.step()

    if rank == 0:
        torch.save(model_to_train.state_dict(), "weights/CLIP_Weights/final_ddp_clip.pth")
        print("✅ Final model saved on rank 0.")

    return clip_model