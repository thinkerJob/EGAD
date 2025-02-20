# MSFlow: Multi-Scale Normalizing Flows for Unsupervised Anomaly Detection

## Enviroment

```shell
conda env create -f environment.yml
```

##  Construct the data structure as follows:

```shell
MVTec
├── bottle
│   ├── ground_truth
│   │   ├── broken_large
│   │   └── ...
│   ├── test
│   │   ├── good
│   │   ├── broken_large
│   │   └── ...
│   └── train
│       └── good
├── cable
└── ...
```

## Download pretrained models and put it in the project root folder
- Download pretrained weights from [Baidu Netdisk](https://pan.baidu.com/s/1gcAJrbk_8Ed3JlF82to_tA)
- extraction code: 4444

## Testing

```shell
bash teatAll.sh
```


## Citation

If you find this work useful for your research, please cite our paper. The formal citation of TNNLS will be updated soon.

```bibtex
@article{zhou2023msflow,
  title={MSFlow: Multi-Scale Flow-based Framework for Unsupervised Anomaly Detection},
  author={Zhou, Yixuan and Xu, Xing and Song, Jingkuan and Shen, Fumin and Shen, Heng Tao},
  journal={arXiv preprint arXiv:2308.15300},
  year={2023}
}
```
