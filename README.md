# Pytorch Implementation of PointNet and PointNet++ 

This repo is implementation for [PointNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) in pytorch.

## Install
The latest codes are tested on Ubuntu 20.04, CUDA11.1, PyTorch 1.7 and Python 3.7:
```shell
conda install pytorch==1.7.1 cudatoolkit=11.1 -c pytorch
```
## Semantic Segmentation (Points Data)
### Data Preparation
Prepare 3D point dataset.
```
cd data_utils
python collect_indoor3d_data.py
```

### Run
```
## Check model in ./models 
## e.g., pointnet2_ssg
python bys_train_semseg_gan.py 
python test_semseg.py --log_dir 2023-05-02_09-37(your log) --test_area 5 --visual
```
Visualization results will save in `log/sem_seg/pointnet2_sem_seg/visual/` and you can visualize these .obj file by [MeshLab](http://www.meshlab.net/).
## Visualization
### Using show3d_balls.py
```
## build C++ code for visualization
cd visualizer
bash build.sh 
## run one example 
python show3d_balls.py
```
![](/visualizer/pic.png)
### Using MeshLab
![](/visualizer/pic2.png)


## Reference By
[Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)<br>

## Citation
If you find this repo useful in your research, please consider citing it and our other works:
```
@article{Author = {Luo Jiangnan},
      Title = {Detection and Pose Measurement of Underground Drill Pipes Based on GA-PointNet++},
      Journal = {https://github.com/caijinyu0609/GA-Point},
      Year = {2024}
}
```
