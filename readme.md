# FashionLOGO

## Data Preparation
1. In "datasets/\[*dataset*\]" directory, we have prepared query and gallery files for each \[*dataset*\].
2. Download datasets **BelgaLogos**, **FlickrLogos-32**, **FoodLogoDet-1500**, **Logo-2K+**, **toplogo10**, put them into corresponding folder.

## Download Pretrained Model
Our model is released in <https://drive.google.com/file/d/1h--xQHHVrguSeycgThzk0FGsBdCnmFZQ/view?usp=drive_link>, download and put it into "checkpoints" folder.

## Predict
```
python3 -m torch.distributed.launch predict.py --model_path checkpoints/your_model_checkpoint
```


