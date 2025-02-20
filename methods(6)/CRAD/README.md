# Continuous Memory Representation for Anomaly Detection (ECCV 2024)
### Joo Chan Lee*, Taejune Kim*, Eunbyung Park, Simon S. Woo, Jong Hwan Ko


## Requirements
To install environment:
```setup
conda env create -f environment.yml
```

## Construct the data structure as follows:
```
|-- data
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
            |--bottle
            |--cable
            |-- ...
        |-- train.json
        |-- test4.json
```

## Download pretrained models and put it in the experiments folder.
- Download pretrained weights from [Baidu Netdisk](https://pan.baidu.com/s/1VgkFn9jyswwZpWBwflznzg)
- extraction code: 2222

## Evaluation
To evaluate a trained model, run:
```eval
cd experiments/
bash eval_torch.sh config.yaml 1 7
```

## BibTeX
```
@article{lee2024crad,
title={Continuous Memory Representation for Anomaly Detection},
author={Lee, Joo Chan and Kim, Taejune and Park, Eunbyung and Woo, Simon S. and Ko, Jong Hwan},
journal={arXiv preprint arXiv:2402.18293},
year={2024}
}
```
