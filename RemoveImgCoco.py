#%%
import json
import os
import glob

root = "/local/scratch/jrs596/dat/Mongoose/Dans_data/train"

file_path = os.path.join(root, "coco_train.json")
with open(file_path) as f:
    data = json.load(f)

#%%
new_coco = {}

new_coco['info'] = data['info']
new_coco['licenses'] = data['licenses']
new_coco['images'] = []
new_coco['annotations'] = []
new_coco['categories'] = data['categories']

#List files in root directory with png extension
path = r'/local/scratch/jrs596/dat/Mongoose/Dans_data/train/images_reduced/*.png'
png_files = glob.glob(path)
png_files = [os.path.basename(png_file) for png_file in png_files]


for i in data['images']:
    if i['file_name'] in png_files:
        print(i['file_name'])  
        new_image = i
        #print(new_image)
        for j in data['annotations']:
            #print(j)
            if j['image_id'] == i['id']:
                new_ann = j
                #new_ann['segments_info'] = j['segmentation'][0]
                
                #new_ann['file_name'] = i['file_name']
                new_coco['annotations'].append(new_ann)
    
        new_coco['images'].append(new_image)
        #data['annotations']['id']


#remove duplicate dictionaries from new_coco['images']
new_coco['images'] = [i for n, i in enumerate(new_coco['images']) if i not in new_coco['images'][n + 1:]]
#remove duplicate dictionaries from new_coco['annotations']
new_coco['annotations'] = [i for n, i in enumerate(new_coco['annotations']) if i not in new_coco['annotations'][n + 1:]]

# %%
#segments_info
print(len(new_coco['annotations']))
# %%
#save new_coco to json file
with open(os.path.join(root, "custom_train.json"), 'w') as f:
    json.dump(new_coco, f)
# %%
