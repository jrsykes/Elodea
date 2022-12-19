#%%
from __future__ import print_function
from __future__ import division
import torch
#import torch.nn as nn
#import torch.optim as optim
#import numpy as np
import torchvision
#from torchvision import datasets, models, transforms
import time
import os
import copy
#from torch.utils.tensorboard import SummaryWriter
import pickle
#import numpy as np
from sklearn import metrics
from progress.bar import Bar
#from torch.utils.mobile_optimizer import optimize_for_mobile

os.environ['TRANSFORMERS_CACHE'] = "/scratch/staff/jrs596/TRANSFORMERS_CACHE"

from transformers import DeformableDetrForObjectDetection, DetrFeatureExtractor
import wandb

import sys
import argparse
from torch.utils.data import DataLoader

#%%
parser = argparse.ArgumentParser('encoder decoder examiner')
parser.add_argument('--model_name', type=str, default='rudder_ddetr',
                        help='save name for model')
parser.add_argument('--root', type=str, default='/scratch/staff/jrs596/dat/',
                        help='location of all data')
parser.add_argument('--data_dir', type=str, default='balloon',
                        help='location of all data')
parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
parser.add_argument('--min_epochs', type=int, default=100,
                        help='n epochs before loss is assesed for early stopping')
parser.add_argument('--patience', type=int, default=50,
                        help='n epochs to run without improvment in loss')
parser.add_argument('--beta', type=float, default=1.005,
                        help='minimum required per cent improvment in validation loss')
parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
parser.add_argument('--lr_backbone', type=float, default=1e-6,
                        help='learning rate for backbone')
parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
parser.add_argument('--eps', type=float, default=1e-8,
                        help='epsilon')



args = parser.parse_args()
print(args)



#Define some variable and paths
#data_dir = os.path.join(args.root, args.data_dir)
data_dir = args.root + "/" + args.data_dir

model_path = args.root + '/models'
log_dir= model_path + "/logs" + "/logs_" + args.model_name

#print n files in each directory
print('Train images: ' + str(len(os.listdir(data_dir + '/train'))-1))
print('Val images: ' + str(len(os.listdir(data_dir + '/val'))-1))

#writer = SummaryWriter(log_dir=log_dir)
#Define traning loop
def train_model(model, dataloaders_dict, optimizer, patience):
    since = time.time()
    
    val_loss_history = []
    best_recall = 0.0
    best_recall_acc = 0.0
    
    #Save current weights as 'best_model_wts' variable. 
    #This will be reviewed each epoch and updated with each improvment in validation recall
    best_model_wts = copy.deepcopy(model.state_dict())

    #######################################################################
    #Initialise dataloader and set decaying batch size

    #initial_patience = patience
    epoch = args.min_epochs
    while epoch > 0: 
        epoch -= 1
        print('\nEpoch {}'.format(epoch))
        print('-' * 10)

        #Ensure minimum number of epochs is met before patience is allow to reduce
        if len(val_loss_history) > args.min_epochs:
            #If the current loss is not at least 0.5% less than the lowest loss recorded, reduce patiece by one epoch
            if val_loss_history[-1] > min(val_loss_history)*args.beta:
                patience -= 1
            else:
                #If validation loss improves by at least 0.5%, reset patient to initial value
                patience = args.patience
        print('Patience: ' + str(patience) + '/' + str(args.patience))
        

        #Training and validation loop
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode

            elif phase == 'val':
                #Model quatisation with quantisation aware training

                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            running_precision = 0
            running_recall = 0
            running_f1 = 0

            # Iterate over data.
            #Get size of whole dataset split
            n = len(dataloaders_dict[phase].dataset)
            #Begin training
            print(phase)
            with Bar('Learning...', max=n/args.batch_size+1) as bar:
                
                for batch in dataloaders_dict[phase]:
                   
                    #Load images and lables from current batch onto GPU(s)
                 
                    # zero the parameter gradients
                    optimizer.zero_grad()   

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # In train mode we calculate the loss by summing the final output and the auxiliary output
                        # but in testing we only consider the final output.
                        
                        #Get predictionas from regular or quantised model
                        #outputs = model(inputs)

                        pixel_values = batch["pixel_values"].to(device)
                        pixel_mask = batch["pixel_mask"].to(device)
                        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

                        print(pixel_values.shape)
                        print(pixel_mask.shape)
                        print(labels)
                    
                        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                        print(outputs)
                     
                        #Calculate loss and other model metrics
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)    
                        stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), digits=4, output_dict = True, zero_division = 0)
                        stats_out = stats['weighted avg']
                        
                        #Add precision, recall or f1-score to loss with weight. Experimental. Doesn't seem to work in practice.
                        #loss += (1-stats_out['recall'])#*0.4

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()  

                            if args.quantise == True:
                                if epoch > 3:
                                    model.apply(torch.quantization.disable_observer)
                                if epoch > 2:
                                    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

                    #Calculate statistics
                    #Here we multiply the loss and other metrics by the number of lables in the batch and then divide the 
                    #running totals for these metrics by the total number of training or validation samples. This controls for 
                    #the effect of batch size and the fact that the size of the last batch will less than args.batch_size
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data) 
                    running_precision += stats_out['precision'] * inputs.size(0)
                    running_recall += stats_out['recall'] * inputs.size(0)
                    running_f1 += stats_out['f1-score'] * inputs.size(0)

                    #Move progress bar.
                    bar.next()

            #Calculate statistics for epoch
            n = len(dataloaders_dict[phase].dataset)
            epoch_loss = float(running_loss / n)
            epoch_acc = float(running_corrects.double() / n)
            epoch_precision = (running_precision) / n         
            epoch_recall = (running_recall) / n        
            epoch_f1 = (running_f1) / n

        
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(phase, epoch_precision, epoch_recall, epoch_f1))
 
            # Save statistics to tensorboard log
            # if phase == 'train':
            #     writer.add_scalar("Loss/train", epoch_loss, epoch)
            #     writer.add_scalar("Accuracy/train", epoch_acc, epoch)
            #     writer.add_scalar("Precision/train", epoch_precision , epoch)
            #     writer.add_scalar("Recall/train", epoch_recall, epoch)
            #     writer.add_scalar("F1/train", epoch_f1, epoch)
            # else:
            #     writer.add_scalar("Loss/val", epoch_loss, epoch)
            #     writer.add_scalar("Accuracy/val", epoch_acc, epoch)
            #     writer.add_scalar("Precision/val", epoch_precision , epoch)
            #     writer.add_scalar("Recall/val", epoch_recall, epoch)
            #     writer.add_scalar("F1/val", epoch_f1, epoch)
              
            
            # Save model and update best weights only if recall has improved
            if phase == 'val' and epoch_recall > best_recall:
                best_recall = epoch_recall
                best_recall_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # Save only the model weights for easy loading into a new model
                final_out = {
                    'model': best_model_wts,
                    '__author__': 'Jamie R. Sykes',
                    '__model_name__': args.model_name,
                    '__model_parameters__': args                    
                    }       
                 
                PATH = os.path.join(model_path, args.model_name)
                
                with open(PATH + '.pkl', 'wb') as f:
                    pickle.dump(final_out, f)   

                    # Save the whole model with pytorch save function
                torch.save(model.module, PATH + '.pth')


            if phase == 'val':
                val_loss_history.append(epoch_loss)
    
        epoch += 1
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Acc of saved model: {:4f}'.format(best_recall_acc))
    print('Recall of saved model: {:4f}'.format(best_recall))
    
    # load best model weights and save
    #model.load_state_dict(best_model_wts)
    
    #Flush and close tensorbaord writer
    #writer.flush()
    #writer.close()


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

#  for i in range(len(labels)):
 #   labels[i]['class_labels'] = torch.tensor([0], dtype=torch.int64)

  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

#Instantiate model and load dataset etc

model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr",  num_labels=1, ignore_mismatched_sizes=True)

#wandb.init(project="Rudder2", entity="frankslab")


feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

train_dataset = CocoDetection(img_folder=os.path.join(args.root, args.data_dir, 'train'), feature_extractor=feature_extractor)
val_dataset = CocoDetection(img_folder=os.path.join(args.root, args.data_dir, 'val'), feature_extractor=feature_extractor, train=False)

print("Number of training examples loaded:", len(train_dataset))
print("Number of validation examples loaded:", len(val_dataset))

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2, shuffle=True, num_workers=1)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=4)

dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

#model = nn.DataParallel(model)
device = torch.device("cuda")

model = model.to(device)

params_to_update = model.parameters()
optimizer = torch.optim.Adam(params_to_update, lr=args.lr,
                                          weight_decay=args.weight_decay, eps=args.eps)

#Train and evaluate

model = train_model(model=model, dataloaders_dict=dataloaders_dict, optimizer=optimizer, patience=args.patience)




# %%
