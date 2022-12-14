
#%%
import torchvision
import os
from transformers import DetrFeatureExtractor
import numpy as np
import os
from PIL import Image, ImageDraw
import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
import torch
import wandb
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
#%%

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor
    def __getitem__(self, idx):
       # read in PIL image and target in COCO format
      img, target = super(CocoDetection, self).__getitem__(idx)
       
      # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
      image_id = self.ids[idx]
      target = {'image_id': image_id, 'annotations': target}
      encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
      pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
      target = encoding["labels"][0] # remove batch dimension
      return pixel_values, target

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
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
    self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=len(id2label), ignore_mismatched_sizes=True)
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

  def validation_step(self, batch, batch_idx):
      loss, loss_dict = self.common_step(batch, batch_idx)     
      #self.log("validation_loss", loss)
      wandb.log({'val/loss' :loss})
      for k,v in loss_dict.items():
        #self.log("validation_" + k, v.item())
        wandb.log({'val/' + k : v.item()})
        return {'loss': loss}

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

  def val_dataloader(self):
      return val_dataloader
# %%
def main():
  
  
  root = "/local/scratch/jrs596/dat/ElodeaProject"
  name = "rudder_detr"

  model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

  outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
  outputs.logits.shape

  trainer = Trainer(gpus=2, gradient_clip_val=0.1, default_root_dir=os.path.join(root, name))
  trainer.fit(model)
#%%
wandb.init(project="Rudder2", entity="frankslab")

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
train_dataset = CocoDetection(img_folder='/local/scratch/jrs596/dat/ElodeaProject/BB4_combined_split/train', feature_extractor=feature_extractor)
val_dataset = CocoDetection(img_folder='/local/scratch/jrs596/dat/ElodeaProject/BB4_combined_split/val', feature_extractor=feature_extractor, train=False)
# %%
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))
# %%
image_ids = train_dataset.coco.getImgIds()
image_id = image_ids[np.random.randint(0, len(image_ids))]
print('Image n°{}'.format(image_id))
image = train_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join('/local/scratch/jrs596/dat/ElodeaProject/BB3_combined_split/train/rudder', image['file_name']))

annotations = train_dataset.coco.imgToAnns[image_id]
draw = ImageDraw.Draw(image, "RGBA")

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

for annotation in annotations:
  box = annotation['bbox']
  class_idx = annotation['category_id']
  x,y,w,h = tuple(box)
  draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
  draw.text((x, y), id2label[class_idx], fill='white')

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=4)
batch = next(iter(train_dataloader))

# %%
batch.keys()
pixel_values, target = train_dataset[0]
pixel_values.shape


# %%
if __name__ == "__main__":
    main()