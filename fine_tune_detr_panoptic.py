#%%
import torch
import json
from pathlib import Path
from PIL import Image
from transformers import DetrFeatureExtractor
import numpy as np
#%%

class CocoPanoptic(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_folder, ann_file, feature_extractor):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco['images'] = sorted(self.coco['images'], key=lambda x: x['id'])
        # sanity check
        if "annotations" in self.coco:
            for img, ann in zip(self.coco['images'], self.coco['annotations']):
                assert img['file_name'][:-4] == ann['file_name'][:-4]

        self.img_folder = img_folder
        self.ann_folder = Path(ann_folder)
        self.ann_file = ann_file
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        ann_info = self.coco['annotations'][idx] if "annotations" in self.coco else self.coco['images'][idx]
        img_path = Path(self.img_folder) / ann_info['file_name'].replace('.png', '.jpg')

        img = Image.open(img_path).convert('RGB')
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        encoding = self.feature_extractor(images=img, annotations=ann_info, masks_path=self.ann_folder, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

    def __len__(self):
        return len(self.coco['images'])
# %%
# we reduce the size and max_size to be able to fit the batches in GPU memory
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic", size=500, max_size=600)

dataset = CocoPanoptic(img_folder='/local/scratch/jrs596/dat/Cats/val', 
                             ann_folder='/local/scratch/jrs596/dat/Cats/annotations',
                             ann_file='/local/scratch/jrs596/dat/Cats/annotations/via_region_data.json',
                             feature_extractor=feature_extractor)

print(dataset)
#%%
# let's split it up into very tiny training and validation sets using random indices
np.random.seed(42)
indices = np.random.randint(low=0, high=len(dataset), size=169)
train_dataset = torch.utils.data.Subset(dataset, indices[:152])
val_dataset = torch.utils.data.Subset(dataset, indices[152:])
# %%
