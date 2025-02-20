# DeSTSeg


## Installation

Please install the environment:

```
conda env create -f environment.yml
```

## Pretrained Checkpoints

- Download pretrained checkpoints [here](https://pan.baidu.com/s/1tYpklT3xNmx6vldMPHWTVg) and put the checkpoints under `<project_dir>/`
- extraction code: 3333

##  Testing

To test the performance of the model, users can run the following command:

```
python eval.py --gpu_id 0 --num_workers 16
```

## Citation

```
@inproceedings{zhang2023destseg,
  title={DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection},
  author={Zhang, Xuan and Li, Shiyu and Li, Xi and Huang, Ping and Shan, Jiulong and Chen, Ting},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3914--3923},
  year={2023}
}
```
