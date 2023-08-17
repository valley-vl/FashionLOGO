# FashionLOGO

## 1. Introduction
This repo is an implementation of our paper: "FashionLOGO: Prompting Multimodal Large Language Models for Fashion Logo Embeddings".

## 2. Data Preparation

**Related Datasets**
* [BelgaLogos](https://www-sop.inria.fr/members/Alexis.Joly/BelgaLogos/BelgaLogos.html)
* [FlickrLogos-32](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/flickrlogos/)
* [FoodLogoDet-1500](https://github.com/hq03/FoodLogoDet-1500-Dataset)
* [Logo-2K+](https://github.com/msn199959/Logo-2k-plus-Dataset)
* [TopLogo-10](https://hangsu0730.github.io/qmul-openlogo/)

1. In "datasets/\[*dataset*\]" directory, we have prepared query and gallery files for each \[*dataset*\].
2. Download related datasets in [datasets collection](https://drive.google.com/file/d/1BFSwKdwg783aQtfHNAqavjZG7FHHdn6X/view?usp=sharing), for FlickrLogos-32 dataset, you need permission from [link](https://www.uni-augsburg.de/de/fakultaet/fai/informatik/prof/mmc/research/datensatze/flickrlogos/).

## 3. Download Pretrained Model
Our model is released in [model](https://drive.google.com/file/d/1h--xQHHVrguSeycgThzk0FGsBdCnmFZQ/view?usp=drive_link), download and put it into "checkpoints" folder.

## 4. Inference
```
python3 -m torch.distributed.launch predict.py --model_path checkpoints/your_model_checkpoint
```