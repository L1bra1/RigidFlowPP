"""
References:
flownet3d: https://github.com/xingyul/flownet3d/blob/master/kitti_dataset.py
FLOT: https://github.com/valeoai/FLOT/blob/master/flot/datasets/kitti_flownet3d.py
RigidFlow: https://github.com/L1bra1/RigidFlow/blob/main/datasets/KITTI_o_test.py
"""

import sys, os
import os.path as osp
import numpy as np
# import pptk

import torch.utils.data as data

__all__ = ['KITTI_OCC']
import glob

class KITTI_OCC(data.Dataset):
    """
    """
    def __init__(self,
                 train,
                 num_points,
                 data_root, limit_35m = False):
        self.train = train
        assert (self.train is False)
        self.num_points = num_points
        self.limit_35m = limit_35m

        self.root = osp.join(data_root, "kitti_rm_ground")
        self.datapath = glob.glob(osp.join(self.root, "*.npz"))
        self.datapath = sorted(self.datapath)

        if len(self.datapath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):

        data = np.load(self.datapath[index])
        pos1 = data['pos1']
        pos2 = data['pos2']
        flow = data['gt']

        if self.limit_35m:
            loc1 = pos1[:, 0] < 35.0
            loc2 = pos2[:, 0] < 35.0
            pos1 = pos1[loc1]
            flow = flow[loc1]
            pos2 = pos2[loc2]

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        if n1 >= self.num_points:
            sample_idx1 = np.random.choice(n1, self.num_points, replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.num_points - n1, replace=True)),
                                         axis=-1)
        if n2 >= self.num_points:
            sample_idx2 = np.random.choice(n2, self.num_points, replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.num_points - n2, replace=True)),
                                         axis=-1)

        pos1 = pos1[sample_idx1, :]
        pos2 = pos2[sample_idx2, :]
        flow = flow[sample_idx1, :]
        mask1 = np.ones_like(pos1[:, 0])
        color1 = np.zeros([self.num_points, 3])
        color2 = np.zeros([self.num_points, 3])
        voxel_list = np.zeros_like(pos1)

        return pos1.astype(np.float32), pos2.astype(np.float32), \
               color1.astype(np.float32), color2.astype(np.float32), \
               flow.astype(np.float32), mask1.astype(np.float32), \
               self.datapath[index], voxel_list.astype(np.int32)