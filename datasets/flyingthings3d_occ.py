"""
Modified based on
FlowNet3D: https://github.com/xingyul/flownet3d/blob/master/flying_things_dataset.py
FLOT: https://github.com/valeoai/FLOT/blob/master/flot/datasets/flyingthings3d_flownet3d.py
"""

import sys, os
import os.path as osp
import numpy as np
# import pptk

import torch.utils.data as data

__all__ = ['FlyingThings3D_OCC']
import glob


lib = np.ctypeslib.load_library('main.so', '../')
c_test = lib.main
c_test.restype = None
c_test.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=2),
]

def compute_sp(input_pos, n_sp):
    input_pos = input_pos.astype(np.float32)
    num_input_points = input_pos.shape[0]

    output_label = np.random.rand(num_input_points)
    output_label = output_label.astype(np.int32)

    num_points = np.array([num_input_points, n_sp])
    num_points = num_points.astype(np.int32)

    output_color = np.random.rand(num_input_points, 3)
    output_color = output_color.astype(np.int32)

    c_test(input_pos, num_points, output_label, output_color)
    return output_label


class FlyingThings3D_OCC(data.Dataset):
    """
    """
    def __init__(self,
                 train,
                 num_points,
                 data_root,
                 num_supervoxels = 30,
                 use_flip=False):
        self.root = osp.join(data_root, 'data_processed_maxcut_35_20k_2k_8192')
        self.train = train
        self.num_points = num_points
        self.num_supervoxels = num_supervoxels
        self.use_flip = use_flip

        assert (num_supervoxels == 30)

        if self.train:
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))

        # from FLOT
        # Remove one sample containing a nan value in train set
        scan_with_nan_value = os.path.join(
            self.root, "TRAIN_C_0140_left_0006-0.npz"
        )
        if scan_with_nan_value in self.datapath:
            self.datapath.remove(scan_with_nan_value)

        # Remove samples with all points occluded in train set
        scan_with_points_all_occluded = [
            "TRAIN_A_0364_left_0008-0.npz",
            "TRAIN_A_0364_left_0009-0.npz",
            "TRAIN_A_0658_left_0014-0.npz",
            "TRAIN_B_0053_left_0009-0.npz",
            "TRAIN_B_0053_left_0011-0.npz",
            "TRAIN_B_0424_left_0011-0.npz",
            "TRAIN_B_0609_right_0010-0.npz",
        ]
        for f in scan_with_points_all_occluded:
            if os.path.join(self.root, f) in self.datapath:
                self.datapath.remove(os.path.join(self.root, f))

        # Remove samples with all points occluded in test set
        scan_with_points_all_occluded = [
            "TEST_A_0149_right_0013-0.npz",
            "TEST_A_0149_right_0012-0.npz",
            "TEST_A_0123_right_0009-0.npz",
            "TEST_A_0123_right_0008-0.npz",
        ]
        for f in scan_with_points_all_occluded:
            if os.path.join(self.root, f) in self.datapath:
                self.datapath.remove(os.path.join(self.root, f))


        if len(self.datapath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        fn = self.datapath[index]
        # print(fn)
        with open(fn, "rb") as fp:
            data = np.load(fp)
            pos1 = data["points1"]
            pos2 = data["points2"]
            color1 = data["color1"] / 255
            color2 = data["color2"] / 255
            flow = data["flow"]
            mask1 = data["valid_mask1"]


        if self.train:
            tmp = np.random.rand(1)
            if self.use_flip and tmp>0.5:
                tmp = pos2.copy()
                pos2 = pos1.copy()
                pos1 = tmp.copy()

                tmp = color2.copy()
                color2 = color1.copy()
                color1 = tmp.copy()
                voxel_list = np.load(self.root + '_30_back/' + fn.split('/')[-1].split('.')[0] + '.npy')
            else:
                voxel_list = np.load(self.root + '_30/' + fn.split('/')[-1].split('.')[0] + '.npy')

            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.num_points, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.num_points, replace=False)

            pos1_ = np.copy(pos1[sample_idx1, :])
            pos2_ = np.copy(pos2[sample_idx2, :])
            color1_ = np.copy(color1[sample_idx1, :])
            color2_ = np.copy(color2[sample_idx2, :])
            flow_ = np.copy(flow[sample_idx1, :])
            mask1_ = np.copy(mask1[sample_idx1])
            voxel_list_ = np.copy(voxel_list[sample_idx1])
            # voxel_list_ = compute_sp(pos1_, self.num_supervoxels)
        else:
            pos1_ = np.copy(pos1[: self.num_points, :])
            pos2_ = np.copy(pos2[: self.num_points, :])
            color1_ = np.copy(color1[: self.num_points, :])
            color2_ = np.copy(color2[: self.num_points, :])
            flow_ = np.copy(flow[: self.num_points, :])
            mask1_ = np.copy(mask1[: self.num_points])
            voxel_list_ = np.zeros_like(pos1_)



        return pos1_.astype(np.float32), pos2_.astype(np.float32), \
               color1_.astype(np.float32), color2_.astype(np.float32), \
               flow_.astype(np.float32), mask1_.astype(np.float32),\
               self.datapath[index], voxel_list_.astype(np.int32)


