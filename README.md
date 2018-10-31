# Multiple Granularity Network
This is a non-official pytorch re-production of paper:[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)


## Dependencies

- Python >= 3.5
- PyTorch >= 0.4.0
- torchvision
- scipy
- numpy
- scikit_learn



## Current Progress

### re-rank
| backbone |  mAP | rank1 | rank3 | rank5 | rank10 |  
| :------: |  :------: | :------: | :------: | :------: |  :------: |   
| resnet50 |  94.28 | 95.67 | 97.45 | 97.89 | 98.60 |  
| resnet101 |  94.50 | 95.87 | 97.62 | 98.16 | 98.57 | 
| paper |  94.20 | 96.60 | - | - | - | 

### no re-rank
| backbone |  mAP | rank1 | rank3 | rank5 | rank10 |  
| :------: |  :------: | :------: | :------: | :------: |  :------: |   
| resnet50 |  85.96 | 94.63 | 97.54 | 98.28 | 98.96 |  
| resnet101 |  86.20 | 95.07 | 97.57 | 98.34 | 98.90 | 
| paper |  86.90 | 95.70 | - | 98.30 | 99.00 | 


## Data

Download Market1501 training data from [here](http://www.liangzheng.org/Project/project_reid.html)

## Weights

Pretrained weights with resnet50 backbone can download from [here](https://drive.google.com/open?id=1ECkD2js0kA82ptod3QfzNDpP9fkt3LZB)  
Pretrained weights with resnet101 backbone can download from [here](https://drive.google.com/open?id=1LUo8JAHOcCKr2qwJdD0UYZsoVvW3BD60)

## Train

You can specify more parameters in opt.py

```
python3 train_eval.py --data_path <path/to/Market-1501-v15.09.15> --mode train
```

## Test

Use pretrained weight or your trained weight.

```
python3 train_eval.py --data_path <path/to/Market-1501-v15.09.15> --weight <path/to/weight_name.pt> 
--backbone <correspond to weight,choices=['resnet50', 'resnet101']>--mode evaluate
```


## Citation

```text
@ARTICLE{2018arXiv180401438W,
    author = {{Wang}, G. and {Yuan}, Y. and {Chen}, X. and {Li}, J. and {Zhou}, X.},
    title = "{Learning Discriminative Features with Multiple Granularities for Person Re-Identification}",
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1804.01438},
    primaryClass = "cs.CV",
    keywords = {Computer Science - Computer Vision and Pattern Recognition},
    year = 2018,
    month = apr,
    adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180401438W},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
