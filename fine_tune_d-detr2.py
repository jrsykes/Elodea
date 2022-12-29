#%%
root = "/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/ElodeaProject"
import os
os.environ['TRANSFORMERS_CACHE'] = root + "/TRANSFORMERS_CACHE"

os.environ['WANDB_MODE'] = 'offline'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torchvision

import numpy as np
import pytorch_lightning as pl
from transformers import DeformableDetrForObjectDetection, AutoImageProcessor
from transformers import AutoConfig
import torch
import wandb
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import time
#%%



wandb.init(project="Rudder2", entity="frankslab")



model_name = "rudder_ddetr"
batch_size = 2

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
  #feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
  feature_extractor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")

  encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

class DDetr(pl.LightningModule):
  def __init__(self, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, config=None):
    super().__init__()
    # replace COCO classification head with custom head
    self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr",  ignore_mismatched_sizes=True, config=config)
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
      wandb.log({'train/loss' :loss})
      for k,v in loss_dict.items():
        wandb.log({'train/' + k : v.item()})
      return {'loss': loss}

  def validation_step(self, batch, batch_idx):
      loss, loss_dict = self.common_step(batch, batch_idx)     
      #self.log("validation_loss", loss)
      wandb.log({'val/loss' :loss})
      for k,v in loss_dict.items():
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
  cats = train_dataset.coco.cats
  id2label = {k: v['name'] for k,v in cats.items()}
  
  #invert id2label
  label2id = {v:k for k,v in id2label.items()}
  
  config = AutoConfig.from_pretrained(root + "/TRANSFORMERS_CACHE/models--SenseTime--deformable-detr/snapshots/a30ee67eda2a60e12d2df31bcd63e57e19fc25bd/config.json")
  config.id2label = id2label
  config.label2id = label2id
  
  model = DDetr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, config=config)

  #outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
  
  #print(outputs.logits.shape)
 
  
  trainer = Trainer(gpus=1, gradient_clip_val=0.1, default_root_dir=os.path.join(root, model_name), max_epochs=10000, log_every_n_steps=10)

  trainer.fit(model)
#%%


#feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")



#train_dataset = CocoDetection(img_folder=os.path.join(root, 'BB4_combined_split/train'), feature_extractor=image_processor)
#val_dataset = CocoDetection(img_folder=os.path.join(root, 'BB4_combined_split/val'), feature_extractor=image_processor, train=False)

train_dataset = CocoDetection(img_folder=os.path.join(root, 'balloon/train'), feature_extractor=image_processor)
val_dataset = CocoDetection(img_folder=os.path.join(root, 'balloon/val'), feature_extractor=image_processor, train=False)
# %%
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))
# %%
# image_ids = train_dataset.coco.getImgIds()
# image_id = image_ids[np.random.randint(0, len(image_ids))]
# print('Image n°{}'.format(image_id))
# image = train_dataset.coco.loadImgs(image_id)[0]
# image = Image.open(os.path.join('/local/scratch/jrs596/dat/ElodeaProject/BB4_combined_split/train/rudder', image['file_name']))

# annotations = train_dataset.coco.imgToAnns[image_id]
# draw = ImageDraw.Draw(image, "RGBA")

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

# for annotation in annotations:
#   box = annotation['bbox']
#   class_idx = annotation['category_id']
#   x,y,w,h = tuple(box)
#   draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
#   draw.text((x, y), id2label[class_idx], fill='white')

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=batch_size)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=batch_size)
#batch = next(iter(train_dataloader))


# %%
#batch.keys()
#pixel_values, target = train_dataset[0]
#pixel_values.shape


# %%
if __name__ == "__main__":
    main()
# %%
