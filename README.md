# RigidFlow++
This is the PyTorch code for [Self-Supervised 3D Scene Flow Estimation and Motion Prediction using Local Rigidity Prior (T-PAMI 2024)](https://www.computer.org/csdl/journal/tp/5555/01/10530455/1WWdXdJBbTW). In this repository, we apply RigidFlow++ to  self-supervised 3D scene flow estimation. For the codes in self-supervised motion prediction, please refer to [RigidFlowPP-Motion](https://github.com/L1bra1/RigidFlowPP-Motion). The code is created by Ruibo Li (ruibo001@e.ntu.edu.sg).

## Prerequisites
- Python 3.7.16, NVIDIA GPU + CUDA CuDNN, PyTorch (torch == 1.9.0),

- tqdm, sklearn, pptk, yaml, numba, thop



Create a conda environment for RigidFlow++:
```bash
conda create -n RigidPP python=3.7.16
conda activate RigidPP
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm pptk PyYAML numba thop
```

Compile the furthest point sampling, grouping, and gathering operation for PyTorch. We use the operation from this [repo](https://github.com/sshaoshuai/Pointnet2.PyTorch).
```bash
cd lib
python setup.py install
cd ../
```


Install & compile supervoxel segmentation method:
```bash
cd Supervoxel_utils
g++ -std=c++11 -fPIC -shared -o main.so main.cc
cd ../
```
More details about the supervoxel segmentation method, please refer to [Supervoxel-for-3D-point-clouds](https://github.com/yblin/Supervoxel-for-3D-point-clouds).

## Data preprocess
By default, the datasets are stored in `SAVE_PATH`.
### FlyingThings3D
1. FlyingThings3D data provided by [HPLFlowNet](https://github.com/laoreja/HPLFlowNet)

    Download and unzip the "Disparity", "Disparity Occlusions", "Disparity change", "Optical flow", "Flow Occlusions" for DispNet/FlowNet2.0 dataset subsets from the [FlyingThings3D website](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (we used the paths from [this file](https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_all_download_paths.txt), now they added torrent downloads). They will be unzipped into the same directory, `RAW_DATA_PATH`. Then run the following script for 3D reconstruction:

    ```bash
    python data_preprocess/process_flyingthings3d_subset.py --raw_data_path RAW_DATA_PATH --save_path SAVE_PATH/FlyingThings3D_subset_processed_35m --only_save_near_pts
    ```

    This dataset is denoted FT3D<sub>s</sub> in our paper.

2. FlyingThings3D data provided by [FlowNet3D](https://github.com/xingyul/flownet3d)

    Download and unzip [data](https://drive.google.com/file/d/1CMaxdt-Tg1Wct8v8eGNwuT7qRSIyJPY-/view) processed by FlowNet3D to directory `SAVE_PATH`. This dataset is denoted FT3D<sub>o</sub> in our paper.

    For the experiments on FT3D<sub>o</sub>, we generate supervoxels for training samples offline:
    ```bash
    python data_preprocess/generate_voxel_for_FT3D_O.py --data_root SAVE_PATH --num_supervoxels 30
    ```

### KITTI
1. KITTI scene flow data provided by [HPLFlowNet](https://github.com/laoreja/HPLFlowNet)

    Download and unzip [KITTI Scene Flow Evaluation 2015](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) to directory `RAW_DATA_PATH`.
    Run the following script for 3D reconstruction:
    ```bash
    python data_preprocess/process_kitti.py RAW_DATA_PATH SAVE_PATH/KITTI_processed_occ_final
    ```
    This dataset is denoted KITTI<sub>s</sub> in our paper.

2. KITTI scene flow data provided by [FlowNet3D](https://github.com/xingyul/flownet3d)

    Download and unzip [data](https://drive.google.com/open?id=1XBsF35wKY0rmaL7x7grD_evvKCAccbKi) processed by FlowNet3D to directory `SAVE_PATH`. This dataset is denoted KITTI<sub>o</sub> in our paper.

3. Unlabeled KITTI raw data

    In our paper, we use [raw data](http://www.cvlibs.net/datasets/kitti/raw_data.php) from KITTI for self-supervised scene flow learning.
    We release the unlabeled training data [here](https://drive.google.com/file/d/12S69dpuz3PDujVZIcrDP_8H5QmbWZP9m/view?usp=sharing) for download. This dataset is denoted KITTI<sub>r</sub> in our paper.

The final data directory will be organized as follows:
```
SAVE_PATH
|-- FlyingThings3D_subset_processed_35m (FT3D_s)
|-- data_processed_maxcut_35_20k_2k_8192 (FT3D_o)
|-- data_processed_maxcut_35_20k_2k_8192_30 (supervoxel for each pc1 in FT3D_o)
|-- data_processed_maxcut_35_20k_2k_8192_30_back (supervoxel for each pc2 in FT3D_o)
|-- KITTI_processed_occ_final (KITTI_s)
|-- kitti_rm_ground (KITTI_o)
|-- KITTI_Raw (KITTI_r)
```


## Evaluation
Set `data_root` in each configuration file to `SAVE_PATH` in the data preprocess section.

### Trained models
We adopt [Bi-PointFlowNet](https://github.com/cwc1260/BiFlow) as the scene ﬂow estimation model.
Our trained models can be downloaded from [Model trained on FT3D<sub>s</sub>](https://drive.google.com/file/d/17SQrZmqgCn0lsu2aFNOsiQtuL4btukW1/view?usp=drive_link), [Model trained on FT3D<sub>o</sub>](https://drive.google.com/file/d/1x8yhsm0YB017IpOfaDXXDVMGwUDR0mjk/view?usp=drive_link), and [Model trained on KITTI<sub>r</sub>](https://drive.google.com/file/d/1iYOLG-0Gs0q0MfirFB2o0eUc7kKmB1dq/view?usp=drive_link).


### Testing

* Model trained on non-occluded FT3D<sub>s</sub>

    When evaluating this pre-trained model on FT3D<sub>s</sub> testing data, set `dataset` to `FlyingThings3DSubset`.  And when evaluating this pre-trained model on KITTI<sub>s</sub> data, set `dataset` to `KITTI`.
Then run:
    ```bash
    python BiFlow-test/evaluate_bid_pointconv.py config_eval_bid_pointconv.yaml
    ```

* Model trained on occluded FT3D<sub>o</sub>

    When evaluating this pre-trained model on FT3D<sub>o</sub> testing data, set `dataset` to `FlyingThings3D_OCC`.  And when evaluating this pre-trained model on KITTI<sub>o</sub> data, set `dataset` to `KITTI_OCC`. Then run:
    ```bash
    python BiFlow-test/evaluate_bid_pointconv_Occ.py config_eval_bid_pointconv_Occ.yaml
    ```

* Model trained on raw KITTI<sub>r</sub>  
    Evaluate this pre-trained model on KITTI<sub>o</sub>:
    ```bash
    python BiFlow-test/evaluate_bid_pointconv_KITTI.py config_eval_bid_pointconv_KITTI.yaml
    ```

## Training
Set `data_root` in each configuration file to `SAVE_PATH` in the data preprocess section.

* Train model on FT3D<sub>s</sub>:
    ```bash
   python BiFlow/train_bid_pointconv.py config_train_bid_pointconv.yaml
    ```

* Train model on FT3D<sub>o</sub>:
    ```bash
   python BiFlow/train_bid_pointconv_Occ.py config_train_bid_pointconv_Occ.yaml
    ```

* Train model on KITTI<sub>r</sub>:
  ```bash
  python BiFlow/train_bid_pointconv_KITTI_R.py config_train_bid_pointconv_KITTI.yaml
  ```

## Citation

If you find this code useful, please cite our paper:
```
@article{li2024self,
  title={Self-Supervised 3D Scene Flow Estimation and Motion Prediction using Local Rigidity Prior},
  author={Li, Ruibo and Zhang, Chi and Wang, Zhe and Shen, Chunhua and Lin, Guosheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```
```
@inproceedings{li2022rigidflow,
  title={RigidFlow: Self-Supervised Scene Flow Learning on Point Clouds by Local Rigidity Prior},
  author={Li, Ruibo and Zhang, Chi and Lin, Guosheng and Wang, Zhe and Shen, Chunhua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16959--16968},
  year={2022}
}
```

## Acknowledgement

Our code is based on [RigidFlow](https://github.com/L1bra1/RigidFlow/), [HPLFlowNet](https://github.com/laoreja/HPLFlowNet), [FlowNet3D](https://github.com/xingyul/flownet3d), [PointPWC](https://github.com/DylanWusee/PointPWC), [Bi-PointFlowNet](https://github.com/cwc1260/BiFlow), and [flownet3d_pytorch](https://github.com/hyangwinter/flownet3d_pytorch).
The supervoxel segmentation method is based on [Supervoxel-for-3D-point-clouds](https://github.com/yblin/Supervoxel-for-3D-point-clouds).
