#%%
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

coco_annotation_file_path = "/local/scratch/jrs596/dat/Kats/Dans_data/train/custom_train.json"
#coco_annotation_file_path = "/local/scratch/jrs596/dat/Kats/Dans_data/train/coco_train.json"
coco_annotation = COCO(annotation_file=coco_annotation_file_path)
#%%
cat_ids = coco_annotation.getCatIds()
cats = coco_annotation.loadCats(cat_ids)
cat_names = [cat["name"] for cat in cats]

catIds = coco_annotation.getCatIds(catNms=cat_names)

imgIds_ = coco_annotation.getImgIds(imgIds = imgIds_[203])

print(imgIds_)

#%%
for i in imgIds_:
    imgIds = coco_annotation.getImgIds(imgIds = i)

    img = coco_annotation.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    #I = io.imread('/local/scratch/jrs596/dat/Kats/Dans_data/train/images/' + img['file_name'])
    plt.clf()
    #plt.imshow(I); plt.axis('off')

    annIds = coco_annotation.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco_annotation.loadAnns(annIds)
    coco_annotation.showAnns(anns)

    mask = coco_annotation.annToMask(anns[0])
    for j in range(len(anns)):
        try:
            mask += coco_annotation.annToMask(anns[j])
        except:
            pass
    #plt.imshow(mask)
    #save mask to file
    plt.imsave('/local/scratch/jrs596/dat/Kats/Dans_data/train/masks/' + img['file_name'], mask)
# %%
