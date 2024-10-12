"""
To train models on unlabeled KITTI raw data.

Modified based on BiFlow:
https://github.com/cwc1260/BiFlow/blob/new1/train_bid_pointconv.py
"""

import argparse
import sys 
import os 

sys.path.append('../')

import torch, numpy as np, glob, math, torch.utils.data, multiprocessing as mp
import torch.nn.functional as F
import time
import pickle 
import datetime
import logging

from tqdm import tqdm 
from models_bid_pointconv import PointConvBidirection
from models_bid_pointconv import multiScaleLoss, multiScaleLoss_mask
from pathlib import Path
from collections import defaultdict

import transforms
import datasets
import cmd_args 
from main_utils import *
from evaluation_utils import evaluate_3d
from pseudo_labels_utils.Conf_aware_Rigid_iter_utils import Conf_aware_Label_Gen_module

def main():

    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0, 1'

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/' + args.model_name + '-' + args.dataset + '-' + str(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))

    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = PointConvBidirection()

    train_dataset = datasets.__dict__[args.dataset](
        train=True,
        num_points=args.num_points,
        data_root = args.data_root,
        limit_35m=True,
        num_supervoxels=args.num_supervoxels,
        use_flip=args.use_flip
    )
    logger.info('train_dataset: ' + str(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

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

    flow_thr = args.flow_thr
    pos_thr = args.pos_thr

    label_gen_func = Conf_aware_Label_Gen_module(iter=4, flow_thr=flow_thr, pos_thr=pos_thr)
    '''GPU selection and multi-GPU'''

    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')

    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True 
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids = device_ids)
        label_gen_func.cuda(device_ids[0])
        label_gen_func = torch.nn.DataParallel(label_gen_func, device_ids = device_ids)
    else:
        model.cuda()
        label_gen_func.cuda()

    init_epoch = 0

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
                
    optimizer.param_groups[0]['initial_lr'] = args.learning_rate 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, last_epoch=init_epoch - 1)
    LEARNING_RATE_CLIP = 1e-5
    
    history = defaultdict(lambda: list())
    best_epe = 1000.0
    conf_aware = False

    for epoch in range(init_epoch, args.epochs):
        if epoch >= args.epoch_confidence:
            conf_aware = True

        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        print('Learning rate:%f'%lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        total_loss = 0
        total_seen = 0
        optimizer.zero_grad()
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            pos1, pos2, flow, _, voxel_list = data
            #move to cuda
            pos1 = pos1.cuda()
            pos2 = pos2.cuda()

            norm1 = torch.zeros_like(pos1)
            norm2 = torch.zeros_like(pos2)
            # flow = flow.cuda()
            voxel_list = voxel_list.cuda().int()


            model = model.train()
            if conf_aware:
                with torch.no_grad():
                    back_pred_flows, _, _, _, _ = model(pos2, pos1, norm2, norm1)

            pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)

            if not conf_aware:
                back_pred_flows = pred_flows.copy()

            ''' Registration'''
            pseudo_gt, label_validity_mask, pos_diff, flow_diff = \
                label_gen_func(pos1, pred_flows[0].permute(0, 2, 1).contiguous(),
                               back_pred_flows[0].permute(0, 2, 1).contiguous(),
                               voxel_list, pos2,
                               conf_aware)

            if conf_aware:
                validity_mask = label_validity_mask * (flow_diff < flow_thr).float() * (pos_diff < pos_thr).float()
            else:
                validity_mask = label_validity_mask

            loss = multiScaleLoss_mask(pred_flows, pseudo_gt, fps_pc1_idxs, validity_mask)

            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.cpu().data * args.batch_size
            total_seen += args.batch_size

        scheduler.step()

        train_loss = total_loss / total_seen
        str_out = 'EPOCH %d %s mean loss: %f'%(epoch, blue('train'), train_loss)
        print(str_out)
        logger.info(str_out)

        eval_epe3d, eval_acc3d = eval_sceneflow(model.eval(), val_loader)
        str_out = 'EPOCH %d %s mean epe3d: %f  mean acc3d: %f'%(epoch, blue('eval'), eval_epe3d, eval_acc3d)
        print(str_out)
        logger.info(str_out)


        if eval_epe3d < best_epe:
            best_epe = eval_epe3d
            if args.multi_gpu is not None:
                torch.save(model.module.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
            else:
                torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, best_epe))
            logger.info('Save model ...')
            print('Save model ...')
        print('Best epe loss is: %.5f'%(best_epe))
        logger.info('Best epe loss is: %.5f'%(best_epe))


def eval_sceneflow(model, loader):

    metrics = defaultdict(lambda:list())
    for _ in range(5):
        # for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        for batch_id, data in enumerate(loader):
            pos1, pos2, flow, _, voxel_list = data
            # move to cuda
            pos1 = pos1.cuda()
            pos2 = pos2.cuda()

            pos1_center = torch.mean(pos1, 1, keepdim=True)

            norm1 = torch.zeros_like(pos1)
            norm2 = torch.zeros_like(pos2)
            flow = flow.cuda()

            with torch.no_grad():
                pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)

                eval_loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)

                # epe3d = torch.norm(pred_flows[0].permute(0, 2, 1) - flow, dim = 2).mean()
                epe_3d, acc_3d, acc_3d_2, _ = evaluate_3d(pred_flows[0].permute(0, 2, 1).detach().cpu().numpy(),
                                                          flow.detach().cpu().numpy())

            metrics['epe_3d'].append(epe_3d)
            metrics['acc_3d'].append(acc_3d)

    mean_epe3d = np.mean(metrics['epe_3d'])
    mean_acc3d = np.mean(metrics['acc_3d'])

    return mean_epe3d, mean_acc3d

if __name__ == '__main__':
    main()




