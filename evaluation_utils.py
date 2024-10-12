"""
Evaluation metrics
Borrowed from HPLFlowNet
Date: May 2020

@inproceedings{HPLFlowNet,
  title={HPLFlowNet: Hierarchical Permutohedral Lattice FlowNet for
Scene Flow Estimation on Large-scale Point Clouds},
  author={Gu, Xiuye and Wang, Yijie and Wu, Chongruo and Lee, Yong Jae and Wang, Panqu},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2019 IEEE International Conference on},
  year={2019}
}
"""

import numpy as np


def evaluate_3d(sf_pred, sf_gt, mask = None):
    """
    sf_pred: (N, 3)
    sf_gt: (N, 3)
    """
    if mask is None:
        mask = np.ones_like(sf_pred[:, :, 0])
    mask = np.expand_dims(mask, -1)

    # error = np.sqrt(np.sum((sf_pred - sf_gt) ** 2, 2, ) + 1e-20)  # B 2048
    error = np.sqrt(np.abs(np.sum((sf_pred - sf_gt) ** 2, 2, )) + 1e-10)  # B 2048

    error = np.expand_dims(error, 2)
    mask_sum = np.sum(mask, 1)  # B 1

    EPE3D = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE3D = np.mean(EPE3D)
    # print(EPE3D)
    # sf_norm = np.sqrt(np.sum(sf_gt * sf_gt, 2) + 1e-20)  # B,N
    sf_norm = np.sqrt(np.abs(np.sum(sf_gt * sf_gt, 2)) + 1e-10)  # B,N
    sf_norm = np.expand_dims(sf_norm, 2)
    relative_err = error / sf_norm

    acc3d_strict = np.sum(np.logical_or((error < 0.05) * mask, (relative_err < 0.05) * mask), axis=1)
    acc3d_relax = np.sum(np.logical_or((error < 0.1) * mask, (relative_err < 0.1) * mask), axis=1)
    outlier = np.sum(np.logical_or((error > 0.3) * mask, (relative_err > 0.1) * mask), axis=1)

    acc3d_strict = acc3d_strict[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc3d_strict = np.mean(acc3d_strict)

    acc3d_relax = acc3d_relax[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc3d_relax = np.mean(acc3d_relax)

    outlier = outlier[mask_sum > 0] / mask_sum[mask_sum > 0]
    outlier = np.mean(outlier)
    return EPE3D, acc3d_strict, acc3d_relax, outlier


def evaluate_2d(flow_pred, flow_gt):
    """
    flow_pred: (N, 2)
    flow_gt: (N, 2)
    """

    epe2d = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    epe2d_mean = epe2d.mean()

    flow_gt_norm = np.linalg.norm(flow_gt, axis=-1)
    relative_err = epe2d / (flow_gt_norm + 1e-5)

    acc2d = (np.logical_or(epe2d < 3., relative_err < 0.05)).astype(np.float).mean()

    return epe2d_mean, acc2d
