"""
To evaluate models trained on unlabeled KITTI raw data.

Modified based on BiFlow:
https://github.com/cwc1260/BiFlow/blob/new1/evaluate_bid_pointconv.py
"""

import argparse
import sys 
import os 
sys.path.append('../')

import torch, numpy as np, glob, math, torch.utils.data, multiprocessing as mp
import torch.nn.functional as F
import time
import torch.nn as nn
import pickle 
import datetime
import logging

from tqdm import tqdm 
from models_bid_pointconv import PointConvBidirection
from models_bid_pointconv import multiScaleLoss
from pathlib import Path
from collections import defaultdict

import transforms
import datasets
import cmd_args 
from main_utils import *
from utils import geometry
from evaluation_utils import evaluate_3d

def main():

    #import ipdb; ipdb.set_trace()
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'

    '''CREATE DIR'''
    experiment_dir = Path('./Evaluate_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sFlyingthings3d-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = PointConvBidirection()

    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        num_points=args.num_points,
        data_root = args.data_root,
        limit_35m=False
    )
    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    #load pretrained model
    pretrain = args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)
    logger.info('load model %s'%pretrain)

    model.cuda()

    epe3ds_full = AverageMeter()
    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()

    total_loss = 0
    total_seen = 0
    total_epe = 0
    metrics = defaultdict(lambda:list())

    num_test = 10
    for _ in range(num_test):
        for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
            pos1, pos2, flow, _, voxel_list = data

            #move to cuda
            pos1 = pos1.cuda()
            pos2 = pos2.cuda()
            norm1 = pos1
            norm2 = pos2
            flow = flow.cuda()

            model = model.eval()
            with torch.no_grad():
                pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)
                full_flow = pred_flows[0].permute(0, 2, 1)

            pc1_np = pos1.cpu().numpy()
            pc2_np = pos2.cpu().numpy()
            sf_np = flow.cpu().numpy()
            pred_sf = full_flow.cpu().numpy()

            EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np, None)

            epe3ds.update(EPE3D)
            acc3d_stricts.update(acc3d_strict)
            acc3d_relaxs.update(acc3d_relax)
            outliers.update(outlier)

            EPE3D_full, _, _, _ = evaluate_3d(pred_sf, sf_np, None)
            epe3ds_full.update(EPE3D_full)


    res_str = (' * EPE3D_Full {epe3d_full_.avg:.4f}\t'
               'EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}'
               .format(epe3d_full_=epe3ds_full,
                       epe3d_=epe3ds,
                       acc3d_s=acc3d_stricts,
                       acc3d_r=acc3d_relaxs,
                       outlier_=outliers,
                       ))

    print(res_str)
    logger.info(res_str)


if __name__ == '__main__':
    main()




