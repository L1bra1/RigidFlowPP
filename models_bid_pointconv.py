"""
Modified based on Bi-pointﬂownet: Bidirectional learning for point cloud based scene ﬂow estimation (ECCV'2022).

In our work, we apply our self-supervised learning method to BiFlow model.

Reference:
BiFlow: https://github.com/cwc1260/BiFlow/blob/new1/models_bid_pointconv.py
"""

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow, CrossLayerLight as CrossLayer
from pointconv_util import SceneFlowEstimatorResidual
from pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance
import time

scale = 1.0


class PointConvBidirection(nn.Module):
    def __init__(self):
        super(PointConvBidirection, self).__init__()

        flow_nei = 32
        feat_nei = 16
        self.scale = scale
        #l0: 8192
        self.level0 = Conv1d(3, 32)
        self.level0_1 = Conv1d(32, 32)
        # self.level0_1_t1 = Conv1d(32, 32)
        # self.level0_1_t2 = Conv1d(32, 32)
        self.cross0 = CrossLayer(flow_nei, 32 + 32 , [32, 32], [32, 32])
        self.flow0 = SceneFlowEstimatorResidual(32 + 64, 32)
        self.level0_2 = Conv1d(32, 64)

        #l1: 2048
        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64)
        self.cross1 = CrossLayer(flow_nei, 64 + 32, [64, 64], [64, 64])
        self.flow1 = SceneFlowEstimatorResidual(64 + 64, 64)
        self.level1_0 = Conv1d(64, 64)
        # self.level1_0_t1 = Conv1d(64, 64)
        # self.level1_0_t2 = Conv1d(64, 64)
        self.level1_1 = Conv1d(64, 128)

        #l2: 512
        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128)
        self.cross2 = CrossLayer(flow_nei, 128 + 64, [128, 128], [128, 128])
        self.flow2 = SceneFlowEstimatorResidual(128 + 64, 128)
        self.level2_0 = Conv1d(128, 128)
        # self.level2_0_t1 = Conv1d(128, 128)
        # self.level2_0_t2 = Conv1d(128, 128)
        self.level2_1 = Conv1d(128, 256)

        #l3: 256
        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256)
        self.cross3 = CrossLayer(flow_nei, 256 + 64, [256, 256], [256, 256])
        self.flow3 = SceneFlowEstimatorResidual(256, 256)
        self.level3_0 = Conv1d(256, 256)
        # self.level3_0_t1 = Conv1d(256, 256)
        # self.level3_0_t2 = Conv1d(256, 256)
        self.level3_1 = Conv1d(256, 512)

        #l4: 64
        self.level4 = PointConvD(64, feat_nei, 512 + 3, 256)

        #deconv
        # self.deconv4_3_t1 = Conv1d(256, 64)
        # self.deconv4_3_t2 = Conv1d(256, 64)
        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 64)
        self.deconv2_1 = Conv1d(128, 32)
        self.deconv1_0 = Conv1d(64, 32)

        #warping
        self.warping = PointWarping()

        #upsample
        self.upsample = UpsampleFlow()

    def forward(self, xyz1, xyz2, color1, color2):
       
        #xyz1, xyz2: B, N, 3
        #color1, color2: B, N, 3

        #l0
        pc1_l0 = xyz1.permute(0, 2, 1)
        pc2_l0 = xyz2.permute(0, 2, 1)
        color1 = color1.permute(0, 2, 1) # B 3 N
        color2 = color2.permute(0, 2, 1) # B 3 N
        feat1_l0 = self.level0(color1)
        # feat1_l0 = self.level0_1_t1(feat1_l0)
        feat1_l0 = self.level0_1(feat1_l0)
        feat1_l0_1 = self.level0_2(feat1_l0)

        feat2_l0 = self.level0(color2)
        # feat2_l0 = self.level0_1_t2(feat2_l0)
        feat2_l0 = self.level0_1(feat2_l0)
        feat2_l0_1 = self.level0_2(feat2_l0)

        #l1
        pc1_l1, feat1_l1, fps_pc1_l1 = self.level1(pc1_l0, feat1_l0_1)
        # feat1_l1 = self.level1_0_t1(feat1_l1)
        feat1_l1 = self.level1_0(feat1_l1)
        feat1_l1_2 = self.level1_1(feat1_l1)

        pc2_l1, feat2_l1, fps_pc2_l1 = self.level1(pc2_l0, feat2_l0_1)
        # feat2_l1 = self.level1_0_t2(feat2_l1)
        feat2_l1 = self.level1_0(feat2_l1)
        feat2_l1_2 = self.level1_1(feat2_l1)

        #l2
        pc1_l2, feat1_l2, fps_pc1_l2 = self.level2(pc1_l1, feat1_l1_2)
        # feat1_l2 = self.level2_0_t1(feat1_l2)
        feat1_l2 = self.level2_0(feat1_l2)
        feat1_l2_3 = self.level2_1(feat1_l2)

        pc2_l2, feat2_l2, fps_pc2_l2 = self.level2(pc2_l1, feat2_l1_2)
        # feat2_l2 = self.level2_0_t2(feat2_l2)
        feat2_l2 = self.level2_0(feat2_l2)
        feat2_l2_3 = self.level2_1(feat2_l2)

        #l3
        pc1_l3, feat1_l3, fps_pc1_l3 = self.level3(pc1_l2, feat1_l2_3)
        # feat1_l3 = self.level3_0_t1(feat1_l3)
        feat1_l3 = self.level3_0(feat1_l3)
        feat1_l3_4 = self.level3_1(feat1_l3)

        pc2_l3, feat2_l3, fps_pc2_l3 = self.level3(pc2_l2, feat2_l2_3)
        # feat2_l3 = self.level3_0_t2(feat2_l3)
        feat2_l3 = self.level3_0(feat2_l3)
        feat2_l3_4 = self.level3_1(feat2_l3)

        #l4
        pc1_l4, feat1_l4, _ = self.level4(pc1_l3, feat1_l3_4)
        feat1_l4_3 = self.upsample(pc1_l3, pc1_l4, feat1_l4)
        # feat1_l4_3 = self.deconv4_3_t1(feat1_l4_3)
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)

        pc2_l4, feat2_l4, _ = self.level4(pc2_l3, feat2_l3_4)
        feat2_l4_3 = self.upsample(pc2_l3, pc2_l4, feat2_l4)
        # feat2_l4_3 = self.deconv4_3_t2(feat2_l4_3)
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)

        #l3
        c_feat1_l3 = torch.cat([feat1_l3, feat1_l4_3], dim = 1)
        c_feat2_l3 = torch.cat([feat2_l3, feat2_l4_3], dim = 1)
        feat1_new_l3, feat2_new_l3, cross3 = self.cross3(pc1_l3, pc2_l3, c_feat1_l3, c_feat2_l3)
        feat3, flow3 = self.flow3(pc1_l3, feat1_l3, cross3)

        feat1_l3_2 = self.upsample(pc1_l2, pc1_l3, feat1_new_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)

        feat2_l3_2 = self.upsample(pc2_l2, pc2_l3, feat2_new_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)

        c_feat1_l2 = torch.cat([feat1_l2, feat1_l3_2], dim = 1)
        c_feat2_l2 = torch.cat([feat2_l2, feat2_l3_2], dim = 1)

        #l2
        up_flow2 = self.upsample(pc1_l2, pc1_l3, self.scale * flow3)
        pc2_l2_warp = self.warping(pc1_l2, pc2_l2, up_flow2)
        feat1_new_l2, feat2_new_l2, cross2 = self.cross2(pc1_l2, pc2_l2_warp, c_feat1_l2, c_feat2_l2)

        feat3_up = self.upsample(pc1_l2, pc1_l3, feat3)
        new_feat1_l2 = torch.cat([feat1_l2, feat3_up], dim = 1)
        feat2, flow2 = self.flow2(pc1_l2, new_feat1_l2, cross2, up_flow2)

        feat1_l2_1 = self.upsample(pc1_l1, pc1_l2, feat1_new_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)

        feat2_l2_1 = self.upsample(pc2_l1, pc2_l2, feat2_new_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)

        c_feat1_l1 = torch.cat([feat1_l1, feat1_l2_1], dim = 1)
        c_feat2_l1 = torch.cat([feat2_l1, feat2_l2_1], dim = 1)

        #l1
        up_flow1 = self.upsample(pc1_l1, pc1_l2, self.scale * flow2)
        pc2_l1_warp = self.warping(pc1_l1, pc2_l1, up_flow1)
        feat1_new_l1, feat2_new_l1, cross1 = self.cross1(pc1_l1, pc2_l1_warp, c_feat1_l1, c_feat2_l1)

        feat2_up = self.upsample(pc1_l1, pc1_l2, feat2)
        new_feat1_l1 = torch.cat([feat1_l1, feat2_up], dim = 1)
        feat1, flow1 = self.flow1(pc1_l1, new_feat1_l1, cross1, up_flow1)

        feat1_l1_0 = self.upsample(pc1_l0, pc1_l1, feat1_new_l1)
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)

        feat2_l1_0 = self.upsample(pc2_l0, pc2_l1, feat2_new_l1)
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)

        c_feat1_l0 = torch.cat([feat1_l0, feat1_l1_0], dim = 1)
        c_feat2_l0 = torch.cat([feat2_l0, feat2_l1_0], dim = 1)

        #l0
        up_flow0 = self.upsample(pc1_l0, pc1_l1, self.scale * flow1)
        pc2_l0_warp = self.warping(pc1_l0, pc2_l0, up_flow0)
        _, _, cross0 = self.cross0(pc1_l0, pc2_l0_warp, c_feat1_l0, c_feat2_l0)

        feat1_up = self.upsample(pc1_l0, pc1_l1, feat1)
        new_feat1_l0 = torch.cat([feat1_l0, feat1_up], dim = 1)
        _, flow0 = self.flow0(pc1_l0, new_feat1_l0, cross0, up_flow0)

        flows = [flow0, flow1, flow2, flow3]
        pc1 = [pc1_l0, pc1_l1, pc1_l2, pc1_l3]
        pc2 = [pc2_l0, pc2_l1, pc2_l2, pc2_l3]
        fps_pc1_idxs = [fps_pc1_l1, fps_pc1_l2, fps_pc1_l3]
        fps_pc2_idxs = [fps_pc2_l1, fps_pc2_l2, fps_pc2_l3]

        return flows, fps_pc1_idxs, fps_pc2_idxs, pc1, pc2

def multiScaleLoss(pred_flows, gt_flow, fps_idxs, alpha = [0.02, 0.04, 0.08, 0.16]):

    #num of scale
    num_scale = len(pred_flows)
    offset = len(fps_idxs) - num_scale + 1

    #generate GT list and mask1s
    gt_flows = [gt_flow]
    for i in range(1, len(fps_idxs) + 1):
        fps_idx = fps_idxs[i - 1]
        sub_gt_flow = index_points(gt_flows[-1], fps_idx) / scale
        gt_flows.append(sub_gt_flow)

    total_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        diff_flow = pred_flows[i].permute(0, 2, 1) - gt_flows[i + offset]
        total_loss += alpha[i] * torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()

    return total_loss


def multiScaleLoss_mask(pred_flows, gt_flow, fps_idxs, mask1, alpha = [0.02, 0.04, 0.08, 0.16]):

    #num of scale
    num_scale = len(pred_flows)
    offset = len(fps_idxs) - num_scale + 1

    #generate GT list and mask1s
    gt_flows = [gt_flow]
    gt_masks = [mask1]
    for i in range(1, len(fps_idxs) + 1):
        fps_idx = fps_idxs[i - 1]
        sub_gt_flow = index_points(gt_flows[-1], fps_idx) / scale
        gt_flows.append(sub_gt_flow)

        sub_gt_mask = index_points(gt_masks[-1].unsqueeze(-1), fps_idx) / scale
        gt_masks.append(sub_gt_mask.squeeze(-1))


    total_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        diff_flow = pred_flows[i].permute(0, 2, 1) - gt_flows[i + offset]
        truncated_mask = (torch.norm(diff_flow, dim=2) < 5.0).float()

        # total_loss += alpha[i] * torch.norm(diff_flow, dim = 2).sum(dim = 1).mean()
        # total_loss += alpha[i] * torch.sum(torch.norm(diff_flow, dim=2) * gt_masks[i + offset], 1).mean()
        total_loss += alpha[i] * torch.sum(torch.norm(diff_flow, dim=2) * truncated_mask * gt_masks[i + offset], 1).mean()

    return total_loss

from thop import profile, clever_format
if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((1,8192,3)).float().cuda()
    model = PointConvBidirection().cuda()
    # print(model)
    output = model(input,input,input,input)
    macs, params = profile(model, inputs=(input,input,input,input))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

    dump_input = torch.randn((1,8192,3)).float().cuda()
    traced_model = torch.jit.trace(model, (dump_input, dump_input, dump_input, dump_input))
    timer = 0

    timer = 0
    for i in range(100):
        t = time.time()
        _ = traced_model(input,input,input,input)
        timer += time.time() - t
    print(timer / 100.0)
