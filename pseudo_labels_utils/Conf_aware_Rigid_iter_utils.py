import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from pointnet2 import pointnet2_utils
# from pointconv_util import knn_point


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, C, N]
        idx: sample index data, [B, S, 1]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    points = points.permute(0, 2, 1)
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx.detach().long(), :]
    return new_points.permute(0, 3, 1, 2).contiguous()


def knn_search_pos_flow(warped_pc1_t, pc2_t, back_pred_t):
    _, idx_1 = pointnet2_utils.knn(1, warped_pc1_t.permute(0, 2, 1).contiguous(), pc2_t.permute(0, 2, 1).contiguous())

    nn_pc2 = index_points(pc2_t.contiguous(), idx_1.int()).squeeze(-1).permute(0, 2, 1).contiguous()
    nn_back_pred = index_points(back_pred_t.contiguous(), idx_1.int()).squeeze(-1).permute(0, 2, 1).contiguous()

    return nn_pc2, nn_back_pred


def rigid_transformation_gen(pc1, pc2, weight):
    pc1_mean = torch.sum(torch.mul(pc1, weight), dim=1, keepdim=True) / torch.sum(weight)
    pc2_mean = torch.sum(torch.mul(pc2, weight), dim=1, keepdim=True) / torch.sum(weight)

    # Eq.(11)
    pc1_moved = pc1 - pc1_mean
    pc2_moved = pc2 - pc2_mean

    X = pc1_moved
    Y = pc2_moved

    # Eq.(12)
    inner = torch.matmul(torch.matmul(X, torch.diag(weight.squeeze())), Y.transpose(1, 0))

    # Eq.(13)
    [U, S, V] = torch.svd(inner)
    R_ = torch.matmul(V, U.transpose(1, 0))

    mid = torch.eye(3, device=weight.device)
    mid[2,2] = torch.det(R_)
    R = torch.matmul(torch.matmul(V, mid), U.transpose(1, 0))

    # Eq.(14)
    t = pc2_mean - torch.matmul(R, pc1_mean)
    return R, t


def pseudo_label_gen_per_sample(pc1, nn_pc2, voxel_label, intial_pseudo_gt, weight_mask, label_validity_mask,
                                max_num_points_in_voxel=600, min_num_points_in_voxel=20):

    num_voxel = torch.max(voxel_label) + 1
    pseudo_gt = torch.zeros_like(intial_pseudo_gt)

    # Updating pseudo labels for each voxel
    for index_voxel in range(num_voxel):
        mask = torch.where(voxel_label == index_voxel)[0]
        num_points_in_voxel = len(mask)
        voxel_weight_mask = weight_mask[:, mask]

        if (num_points_in_voxel > min_num_points_in_voxel):
            points_in_voxel = pc1[:, mask]
            nn_points_in_pc2 = nn_pc2[:, mask]

            if num_points_in_voxel > max_num_points_in_voxel:
                selected_points_in_voxel = points_in_voxel[:, 0:max_num_points_in_voxel]
                selected_nn_points_in_pc2 = nn_points_in_pc2[:, 0:max_num_points_in_voxel]
                selected_weight_mask = voxel_weight_mask[:, 0:max_num_points_in_voxel]
            else:
                selected_points_in_voxel = points_in_voxel
                selected_nn_points_in_pc2 = nn_points_in_pc2
                selected_weight_mask = voxel_weight_mask

            # Updating rigid transformation estimate
            [R, t] = rigid_transformation_gen(selected_points_in_voxel.cpu(),
                                              selected_nn_points_in_pc2.cpu(), selected_weight_mask.cpu())
            R, t = R.cuda(), t.cuda()
            # Generating pseudo labels for each voxel
            pseudo_gt[:, mask] = torch.matmul(R, points_in_voxel) + t - points_in_voxel
        else:
            pseudo_gt[:, mask] = intial_pseudo_gt[:, mask]
            label_validity_mask[:, mask] = 0

    return pseudo_gt, label_validity_mask



def Conf_aware_Rigid_Reg(pc1, flow_pred, back_flow_pred, voxel_label_1, pc2,
                         conf_aware, iter=1, theta0=0.01, flow_thr=0.2, pos_thr=0.1):
    """
    Pseudo label generation
    ----------
    Input:
        pc1, pc2: Input points position, [B, N, 3]
        flow_pred, back_flow_pred: Scene flow prediction, [B, N, 3]
        voxel_label_1: Supervoxel label for each point in pc1, [B, N]
        iter: Iteration number
        conf_aware: using confidence mechanism or not
    -------
    Returns:
        pseudo_gt: Pseudo labels, [B, N, 3]
        label_validity_mask: indicate the validity of the pseudo labels, [B, N]
        pos_diff: the distance between warped point and corresponding point, [B, N]
        flow_diff: the mismatch between the forward ﬂow and the reversed backward ﬂow, [B, N]
        label_validity_mask, pos_diff, and flow_diff will be used to generate the final binary validity mask for training
    """

    pc1 = pc1.permute(0, 2, 1).contiguous()
    pc2 = pc2.permute(0, 2, 1).contiguous()
    flow_pred = flow_pred.permute(0, 2, 1).contiguous()
    batch_size = pc1.size(0)

    # Initializing by predicted flow.
    pseudo_gt = flow_pred.clone().detach()
    flow_forward = flow_pred.clone().detach()
    flow_backward = back_flow_pred.clone().detach().permute(0, 2, 1).contiguous()

    # Iteratively generate pseudo labels
    for index_reg in range(iter):
        # Updating point mapping by nearest neighbor search
        nn_pc2, nn_back_pred = knn_search_pos_flow(pc1 + pseudo_gt, pc2, flow_backward)
        nn_pc2 = nn_pc2.permute(0, 2, 1).contiguous()
        nn_back_pred = nn_back_pred.permute(0, 2, 1).contiguous()

        if conf_aware:
            # Eq.(8)
            flow_diff = torch.sum((flow_forward + nn_back_pred) ** 2, 1) + 1e-6
            pos_diff = torch.sqrt(torch.sum((pc1 + pseudo_gt - nn_pc2) ** 2, 1) + 1e-6)

            # Eq.(9)
            conf_score = torch.exp(torch.neg(flow_diff / theta0)) + 1e-6
            conf_score = conf_score.unsqueeze(1)

            # Eq.(10)
            conf_weight = conf_score * (flow_diff.unsqueeze(1) < flow_thr).float()\
                                * (pos_diff.unsqueeze(1) < pos_thr).float() + 1e-6
        else:
            conf_weight = torch.ones_like(pc1[:, 0:1, :]) + 1e-6

        label_validity_mask = torch.ones_like(pc1[:, 0:1, :])
        # Updating pseudo labels per sample
        for index in range(batch_size):
            pseudo_gt[index, :, :], label_validity_mask[index, :, :] = \
                pseudo_label_gen_per_sample(pc1[index, :, :], nn_pc2[index, :, :],
                                            voxel_label_1[index, :], pseudo_gt[index, :, :],
                                            conf_weight[index, :, :],
                                            label_validity_mask[index, :, :])

    ''' pos diff'''
    pseudo_gt_pc2 = pc1 + pseudo_gt
    nn_pc2, nn_back_pred = knn_search_pos_flow(pseudo_gt_pc2, pc2, flow_backward)
    pos_diff = torch.sum((pseudo_gt_pc2.permute(0, 2, 1).contiguous() - nn_pc2)**2, -1) + 1e-6
    pos_diff = torch.sqrt(pos_diff)
    nn_back_pred = nn_back_pred.permute(0, 2, 1).contiguous()

    ''' flow diff'''
    flow_diff = torch.sum((flow_forward + nn_back_pred) ** 2, 1) + 1e-6
    flow_diff = torch.sqrt(flow_diff + 1e-6)

    return pseudo_gt.permute(0, 2, 1).contiguous(), label_validity_mask.squeeze(1), pos_diff, flow_diff


class Conf_aware_Label_Gen_module(torch.nn.Module):
    def __init__(self, iter=2, theta0=0.01, flow_thr=0.2, pos_thr=0.1):
        super(Conf_aware_Label_Gen_module, self).__init__()
        self.iter = iter
        self.theta0 = theta0
        self.flow_thr = flow_thr
        self.pos_thr = pos_thr

    def forward(self, pc1, flow_pred, back_flow_pred, voxel_label_1, pc2, conf_aware):
        with torch.no_grad():
            pseudo_gt, label_validity_mask, pos_diff, flow_diff = \
                Conf_aware_Rigid_Reg(pc1, flow_pred, back_flow_pred, voxel_label_1, pc2,
                                     conf_aware, iter=self.iter, theta0=self.theta0,
                                     flow_thr=self.flow_thr, pos_thr=self.pos_thr)
        return pseudo_gt, label_validity_mask, pos_diff, flow_diff

