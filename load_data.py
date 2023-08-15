import torch
import numpy as np
import cv2
import pandas as pd
import os

class EmbeddingRecallDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, px_mean= [123.675,116.28,103.53],px_std= [58.395,57.12,57.375],input_size = 224):
        self.px_mean = torch.tensor(px_mean, dtype=torch.float)
        self.px_mean = self.px_mean[:, None, None]
        self.px_std = torch.tensor(px_std, dtype=torch.float)
        self.px_std = self.px_std[:, None, None]
        self.input_size = input_size
        self.dataset = dataset
        query_file = os.path.join('datasets',self.dataset,'query.csv')
        gallery_file = os.path.join('datasets',self.dataset,'gallery.csv')
        query = pd.read_csv(query_file)
        query['type'] = query.apply(lambda row: 0, axis=1)
        if os.path.exists(gallery_file):
            gallery = pd.read_csv(gallery_file)
            gallery['type'] = gallery.apply(lambda row: 1, axis=1)
            self.samples = pd.concat([gallery,query], ignore_index=True)
        else:
            self.samples = query
    
    def __len__(self):
        return self.samples.shape[0]
    
    def __getitem__(self, idx,):
        if self.dataset == 'Logo-2K+':
            img_path,local_id,label,type_ = self.samples.iloc[idx]
            img_array = cv2.imread(os.path.join('datasets',self.dataset,img_path))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif self.dataset == 'FoodLogoDet-1500':
            img_path,img_width,img_height,local_id,bbox,label,type_ = self.samples.iloc[idx]
            img_array = cv2.imread(os.path.join('datasets',self.dataset,img_path))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            x1, y1, x2, y2 = eval(bbox)
            img_array = img_array[y1: y2, x1: x2]
        elif self.dataset == 'BelgaLogos':
            img_path,inst_id,local_id,bbox,label,type_ = self.samples.iloc[idx]
            print(os.path.join('datasets',self.dataset,'images',img_path))
            img_array = cv2.imread(os.path.join('datasets',self.dataset,'images',img_path))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            x1, y1, x2, y2 = eval(bbox)
            img_array = img_array[y1: y2, x1: x2]
        elif self.dataset == 'FlickrLogos-32':
            img_path,local_id,bbox,label,type_ = self.samples.iloc[idx]
            img_array = cv2.imread(os.path.join('datasets',self.dataset,img_path))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            x1, y1, x2, y2 = eval(bbox)
            img_array = img_array[y1: y2, x1: x2]
        elif self.dataset == 'toplogo10':
            img_path,local_id,label,type_ = self.samples.iloc[idx]
            img_array = cv2.imread(os.path.join('datasets',self.dataset,img_path))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("dataset {} is illegal.".format(self.dataset))
        
        img = cv2.resize(img_array, dsize=(224,224), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)
        img = img.sub_(self.px_mean).div_(self.px_std)
        return img, local_id, type_