#%%
import os
os.chdir("/home/userfs/j/jrs596/scripts/Elodea")

from torch.utils.data import DataLoader
from transformers import DetrFeatureExtractor, DetrConfig
from tqdm.notebook import tqdm
import torch
from fine_tune_detrGPU5 import Detr, CocoDetection, collate_fn
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
#from transformers import DetrConfig, DetrForObjectDetection

#set TRANSFORMERS_CACHE to a directory where you want to store the model weights
os.environ['TRANSFORMERS_CACHE'] = "/local/scratch/jrs596/TRANSFORMERS_CACHE"
# %%


feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

val_dataset = CocoDetection(img_folder='/local/scratch/jrs596/dat/ElodeaProject/Elodea_BB/val', feature_extractor=feature_extractor, train=False)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)

#os.chdir("/home/userfs/j/jrs596/scripts/detr")
from datasets import get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator

base_ds = get_coco_api_from_dataset(val_dataset)

iou_types = ['bbox']

coco_evaluator = CocoEvaluator(base_ds, iou_types) # initialize evaluator with ground truths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
checkpoint_pth = "/local/scratch/jrs596/dat/ElodeaProject/rudder_detr/epoch=1047-step=7336.ckpt"


#model = Detr()

model = Detr.load_from_checkpoint(checkpoint_pth)

model.to(device)
model.eval()
#%%
print("Running evaluation...")

for idx, batch in enumerate(tqdm(val_dataloader)):
    # get the inputs
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized
    # forward pass
    outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
    res = {target['image_id'].item(): output for target, output in zip(labels, results)}
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()
# %%
image = Image.open('/local/scratch/jrs596/dat/ElodeaProject/BB4_combined_split/val/1670514682.2031014.jpeg')

convert_tensor = transforms.ToTensor()
resize_tensor = transforms.Resize((2500, 2500))
pixel_values = convert_tensor(image).unsqueeze(0).to(device)
pixel_values = resize_tensor(pixel_values)

outputs = model(pixel_values=pixel_values, pixel_mask=None)


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        id2label = {0: 'rudder'}
        text = f'{id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def visualize_predictions(image, outputs, threshold=0.000000):
  # keep only predictions with confidence >= threshold
  probas = outputs.logits.softmax(-1)[0, :, :-1]
  #keep = probas.max(-1).values > threshold
  #keep 10 best boxes
  #keep = probas.max(-1).values.topk(5).indices
  #keep only max prob
  keep = probas.max(-1).values == probas.max(-1).values.max()
  # convert predicted boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

  # plot results
  plot_results(image, probas[keep], bboxes_scaled)


#image = Image.open('/local/scratch/jrs596/dat/ElodeaProject/BB3_combined/rudder/1670514682.5410795.jpeg')
visualize_predictions(image, outputs)
     
# %%
