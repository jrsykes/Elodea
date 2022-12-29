
#%%
root = "/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat"
import os
#os.environ['TRANSFORMERS_CACHE'] = root + "/TRANSFORMERS_CACHE"

os.environ['WANDB_MODE'] = 'offline'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torchvision
from transformers import DetrFeatureExtractor
import numpy as np
import os
from PIL import Image, ImageDraw
import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection, AutoConfig
from transformers import DetrFeatureExtractor
import torch
import wandb
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import json
from pathlib import Path

#%%

batch_size = 8
name = "mongoose_detr_seg"


class CocoPanoptic(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_folder, ann_file, feature_extractor):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])
        # sanity check
#        print(self.coco['annotations'][0])
        #if "annotations" in self.coco:
         #   for img, ann in zip(self.coco['images'], self.coco['annotations']):
          #      assert img['file_name'][:-4] == ann['file_name'][:-4]

        self.img_folder = img_folder
        self.ann_folder = Path(ann_folder)
        self.ann_file = ann_file
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        ann_info = self.coco['annotations'][idx] if "annotations" in self.coco else self.coco['images'][idx]
        img_path = Path(self.img_folder) / ann_info['file_name']#.replace('.png', '.jpg')

        img = Image.open(img_path).convert('RGB')
        print(ann_info['image_id'])
        print(ann_info['file_name'])
        print(ann_info['segments_info'])
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        encoding = self.feature_extractor(images=img, annotations=ann_info, masks_path=self.ann_folder, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

    def __len__(self):
        return len(self.coco['images'])

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic", size=500, max_size=600)

# train_dataset = CocoPanoptic(img_folder='/local/scratch/jrs596/dat/Mongoose/Dans_data/train/images_reduced', 
#                              ann_folder='/local/scratch/jrs596/dat/Mongoose/Dans_data/train/masks',
#                              ann_file='/local/scratch/jrs596/dat/Mongoose/Dans_data/train/custom_train.json',
#                              feature_extractor=feature_extractor)

train_dataset = CocoPanoptic(img_folder='/local/scratch/jrs596/dat/balloon/train/', 
                             ann_folder='/local/scratch/jrs596/dat/Mongoose/Dans_data/train/masks',
                             ann_file='/local/scratch/jrs596/dat/balloon/train/custom_train.json',
                             feature_extractor=feature_extractor)

pixel_values, target = train_dataset[2]
print(pixel_values.shape)
print(target.keys())
#%%
def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic", size=500, max_size=600)
  encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

class Detr(pl.LightningModule):
  def __init__(self, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4):
    super().__init__()
    # replace COCO classification head with custom head
    
    config = AutoConfig.from_pretrained("/jmain02/home/J2AD016/jjw02/jjs00-jjw02/.cache/huggingface/hub/models--facebook--detr-resnet-50/snapshots/c783425425d573f30483efb0660bf6207deea991/config.json")
    config.id2label = id2label
    config.label2id = label2id
    self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", ignore_mismatched_sizes=True, config=config)
    # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
    self.lr = lr
    self.lr_backbone = lr_backbone
    self.weight_decay = weight_decay

  def forward(self, pixel_values, pixel_mask):
      outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
      return outputs

  def common_step(self, batch, batch_idx):
      pixel_values = batch["pixel_values"]
      pixel_mask = batch["pixel_mask"]
      labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

      outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

      loss = outputs.loss
      loss_dict = outputs.loss_dict

      return loss, loss_dict

  def training_step(self, batch, batch_idx):
      loss, loss_dict = self.common_step(batch, batch_idx)     
      # logs metrics for each training_step,
      # and the average across the epoch
      #self.log("training_loss", loss)
      wandb.log({'train/loss' :loss})
      for k,v in loss_dict.items():
        #self.log("train_" + k, v.item())
        wandb.log({'train/' + k : v.item()})
        return {'loss': loss}

#   def validation_step(self, batch, batch_idx):
#       loss, loss_dict = self.common_step(batch, batch_idx)     
#       #self.log("validation_loss", loss)
#       wandb.log({'val/loss' :loss})
#       for k,v in loss_dict.items():
#         #self.log("validation_" + k, v.item())
#         wandb.log({'val/' + k : v.item()})
#         return {'loss': loss}

  def configure_optimizers(self):
      param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
      ]
      optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

      return optimizer

  def train_dataloader(self):
      return train_dataloader

#   def val_dataloader(self):
#       return val_dataloader
# %%
def main():
  
  
  
  model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

  outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
  outputs.logits.shape

  trainer = Trainer(gpus=8, gradient_clip_val=0.1, default_root_dir=os.path.join(root, 'ElodeaProject', name), max_epochs=10000, log_every_n_steps=3)
  trainer.fit(model)
#%%
wandb.init(project="Rudder2", entity="frankslab")

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic", size=500, max_size=600)
#train_dataset = CocoDetection(img_folder='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat"/ElodeaProject/BB4_combined_split/train', feature_extractor=feature_extractor)
#val_dataset = CocoDetection(img_folder='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/ElodeaProject/BB4_combined_split/val', feature_extractor=feature_extractor, train=False)

train_dataset = CocoDetection(img_folder='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/Mongoose/data/train', feature_extractor=feature_extractor)

# %%
print("Number of training examples:", len(train_dataset))
#print("Number of validation examples:", len(val_dataset))
# %%
# image_ids = train_dataset.coco.getImgIds()
# image_id = image_ids[np.random.randint(0, len(image_ids))]
# print('Image n°{}'.format(image_id))
# image = train_dataset.coco.loadImgs(image_id)[0]
# image = Image.open(os.path.join('/local/scratch/jrs596/dat/ElodeaProject/BB3_combined_split/train/rudder', image['file_name']))

# annotations = train_dataset.coco.imgToAnns[image_id]
# draw = ImageDraw.Draw(image, "RGBA")

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}
label2id = {v:k for k,v in id2label.items()}

# for annotation in annotations:
#   box = annotation['bbox']
#   class_idx = annotation['category_id']
#   x,y,w,h = tuple(box)
#   draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
#   draw.text((x, y), id2label[class_idx], fill='white')

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=batch_size)
#val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=batch_size)
batch = next(iter(train_dataloader))

# %%
batch.keys()
pixel_values, target = train_dataset[0]
pixel_values.shape


# %%
if __name__ == "__main__":
    main()