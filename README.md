# Multiple Granularity Network
Implement of paper:[Learning Discriminative Features with Multiple Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438v1)

## Dependencies

- Python >= 3.5
- PyTorch >= 0.4.0
- torchvision
- scipy
- numpy
- scikit_learn



## Current Result

| Re-Ranking| backbone |  mAP | rank1 | rank3 | rank5 | rank10 |  
| :------: | :------: |  :------: | :------: | :------: | :------: |  :------: |   
| yes | resnet50 |  94.33 | 95.58 | 97.54 | 97.92 | 98.46 | 
| no | resnet50 |  86.15 | 94.95 | 97.42 | 98.07 | 98.93 | 



## Data

The data structure would look like:
```
data/
    bounding_box_train/
    bounding_box_test/
    query/
```
#### Market1501 
Download from [here](http://www.liangzheng.org/Project/project_reid.html)

#### DukeMTMC-reID
Download from [here](http://vision.cs.duke.edu/DukeMTMC/)

#### CUHK03
1. Download cuhk03 dataset from "http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html"
2. Unzip the file and you will get the cuhk03_release dir include cuhk-03.mat
3. Download "cuhk03_new_protocol_config_detected.mat" from "https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03"
and put it with cuhk-03.mat. We need this new protocol to split the dataset.
```
python utils/transform_cuhk03.py --src <path/to/cuhk03_release> --dst <path/to/save>
```

NOTICE:You need to change num_classes in network depend on how many people in your train dataset! e.g. 751 in Market1501

## Weights

Pretrained weight download from [google drive](https://drive.google.com/open?id=16V7ZsflBbINHPjh_UVYGBVO6NuSxEMTi)
or [baidu drive](https://pan.baidu.com/s/12AkumLX10hLx9vh_SQwdyw) password:mrl5
## Train

You can specify more parameters in opt.py

```
python main.py --mode train --data_path <path/to/Market-1501-v15.09.15> 
```

## Evaluate

Use pretrained weight or your trained weight

```
python main.py --mode evaluate --data_path <path/to/Market-1501-v15.09.15> --weight <path/to/weight_name.pt> 
```

## Visualize

Visualize rank10 query result of one image(query from bounding_box_test)

Extract features will take a few munutes, or you can save features as .mat file for multiple uses

![image](https://s1.ax1x.com/2018/11/27/FV9xyj.png)

```
python main.py --mode vis --query_image <path/to/query_image> --weight <path/to/weight_name.pt> 
```


## Citation

```text
@ARTICLE{2018arXiv180401438W,
    author = {{Wang}, G. and {Yuan}, Y. and {Chen}, X. and {Li}, J. and {Zhou}, X.},
    title = "{Learning Discriminative Features with Multiple Granularities for Person Re-Identification}",
    journal = {ArXiv e-prints},
    year = 2018,
}
```
