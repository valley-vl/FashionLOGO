import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm 
from transformers import CLIPModel
import argparse
from utils import distributed_concat, get_metric
from load_data import EmbeddingRecallDataset


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--model_path', default='', type=str, help='path to model checkpoint')
parser.add_argument('--predict_batch_size', default=64, type=int, help='predict batch size')
parser.add_argument('--num_workers', default=8, type=int, help='num workers')
parser.add_argument('--dataset', default='Logo-2K+', choices=['Logo-2K+', 'FoodLogoDet-1500', 'BelgaLogos', 'FlickrLogos-32', 'toplogo10'], help='Test dataset') 
args = parser.parse_args()
torch.distributed.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

### load model ###
model  = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
pretrain = torch.load(args.model_path, map_location="cpu")
state_dict_pretrain = pretrain['state_dict']
state_dict = { k.replace('backbone.', '') : v for k, v in state_dict_pretrain.items()}
state_dict_new = OrderedDict()
for key, value in state_dict.items():
    if key in model.state_dict().keys():
        state_dict_new[key] = value
model.load_state_dict(state_dict_new)
model.cuda(args.local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

### load data ####
predict_dataset = EmbeddingRecallDataset(dataset = args.dataset)
predict_sampler = torch.utils.data.distributed.DistributedSampler(predict_dataset)
predict_dataloader= torch.utils.data.DataLoader(predict_dataset,sampler = predict_sampler, batch_size=args.predict_batch_size ,num_workers=args.num_workers, shuffle=False)

### start predict! ###
model.eval()
with torch.no_grad():
    labels = []
    types = []
    embeds = []
    for idx, batch in tqdm(enumerate(predict_dataloader)):
        images, label, type_ = batch
        images, label, type_ = images.to(args.local_rank), label.to(args.local_rank), type_.to(args.local_rank)
        emb  = F.normalize(model.module.get_image_features(images))
        embeds.extend(emb)
        labels.extend(label)
        types.extend(type_)
        torch.distributed.barrier()
    embeds = distributed_concat(torch.stack(embeds),  len(predict_sampler.dataset))
    labels = distributed_concat(torch.stack(labels), len(predict_sampler.dataset))
    types = distributed_concat(torch.stack(types), len(predict_sampler.dataset))
    gallery_emb = embeds[types==1]
    query_emb = embeds[types==0]
    gallery_labels = labels[types==1]
    query_labels = labels[types==0]

    val_metric_dict = get_metric(query_emb, query_labels, gallery_emb, gallery_labels)
    print(torch.distributed.get_rank(),val_metric_dict)