"""
Modified based on HPLFlowNet:
https://github.com/laoreja/HPLFlowNet/blob/master/datasets/flyingthings3d_subset.py
"""

import sys, os
import os.path as osp
import numpy as np
# import pptk

import torch.utils.data as data

__all__ = ['FlyingThings3DSubset']

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

class FlyingThings3DSubset(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        args:
    """
    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 full=True,
                 use_flip=False,
                 num_supervoxels = 30):
        self.root = osp.join(data_root, 'FlyingThings3D_subset_processed_35m')
        self.train = train
        self.transform = transform
        self.num_points = num_points
        self.num_supervoxels = num_supervoxels
        self.use_flip = use_flip

        self.samples = self.make_dataset(full)

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        #import ipdb; ipdb.set_trace()
        #fn = '/media/wenxuan/Large/Code2019_9/HPLFlowNet/data_preprocess/FlyingThings3D_subset_processed_35m/train/0004781'
        #pc1_loaded, pc2_loaded = self.pc_loader(fn)
        pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)

        if self.train:
            tmp = np.random.rand(1)
            if self.use_flip and (tmp > 0.5):
                tmp = pc2_transformed.copy()
                pc2_transformed = pc1_transformed.copy()
                pc1_transformed = tmp.copy()

        pc1_norm = pc1_transformed
        pc2_norm = pc2_transformed
        if self.train:
            voxel_list = compute_sp(pc1_transformed, self.num_supervoxels)
        else:
            voxel_list = np.ones_like(pc1_transformed)

        return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples[index], voxel_list.astype(np.int32)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def make_dataset(self, full):
        root = osp.realpath(osp.expanduser(self.root))
        root = osp.join(root, 'train') if self.train else osp.join(root, 'val')

        all_paths = os.walk(root)
        useful_paths = sorted([item[0] for item in all_paths if len(item[1]) == 0])

        print(len(useful_paths))

        try:
            if self.train:
                assert (len(useful_paths) == 19640)
            else:
                assert (len(useful_paths) == 3824)
        except AssertionError:
            print('len(useful_paths) assert error', len(useful_paths))
            sys.exit(1)

        if not full:
            res_paths = useful_paths[::4]
        else:
            res_paths = useful_paths

        return res_paths

    def pc_loader(self, path):
        """
        Args:
            path: path to a dir, e.g., home/xiuye/share/data/Driving_processed/35mm_focallength/scene_forwards/slow/0791
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(osp.join(path, 'pc1.npy'))
        pc2 = np.load(osp.join(path, 'pc2.npy'))
        # multiply -1 only for subset datasets
        pc1[..., -1] *= -1
        pc2[..., -1] *= -1
        pc1[..., 0] *= -1
        pc2[..., 0] *= -1

        return pc1, pc2