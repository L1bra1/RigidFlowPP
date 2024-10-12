import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import numpy as np
import os.path as osp

import argparse
from datasets.flyingthings3d_occ import FlyingThings3D_OCC

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

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='/path_to/', type=str, required=True, help="path to the FT3D_O training data")
parser.add_argument('--num_supervoxels', default=30)

args = parser.parse_args()


if __name__ == '__main__':
    train_dataset = FlyingThings3D_OCC(
        train=True,
        num_points=8192,
        data_root=args.data_root,
        num_supervoxels=args.num_supervoxels,
    )

    file_list = train_dataset.datapath
    train_dataset_root = train_dataset.root

    save_root = os.path.join(args.data_root, train_dataset_root.split('/')[-1] + '_' + str(args.num_supervoxels))
    os.makedirs(save_root, exist_ok=True)

    back_save_root = os.path.join(args.data_root, train_dataset_root.split('/')[-1] + '_' + str(args.num_supervoxels)+ '_back')
    os.makedirs(back_save_root, exist_ok=True)

    for index in range(len(file_list)):
        print(index)
        fn = file_list[index]
        with open(fn, "rb") as fp:
            data = np.load(fp)
            pos1 = data["points1"]
            pos2 = data["points2"]

        voxel_list = compute_sp(pos1, args.num_supervoxels)
        save_name = os.path.join(save_root, fn.split('/')[-1].split('.')[0] + '.npy')
        np.save(save_name, voxel_list)

        voxel_list = compute_sp(pos2, args.num_supervoxels)
        save_name = os.path.join(back_save_root, fn.split('/')[-1].split('.')[0] + '.npy')
        np.save(save_name, voxel_list)