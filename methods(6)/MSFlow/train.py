import os
import time
import datetime
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from datasets import MVTecDataset, VisADataset,get_image_names
from models.extractors import build_extractor
from models.flow_models import build_msflow_model
from post_process import post_process
from utils import Score_Observer, t2np, positionalencoding2d, save_weights, load_weights
from evaluations import eval_det_loc


def model_forward(c, extractor, parallel_flows, fusion_flow, image):
    h_list = extractor(image)
    # for idx, h in enumerate(h_list):
    #     print(f"特征图 {idx} 的形状: {h.shape}")
    if c.pool_type == 'avg':
        pool_layer = nn.AvgPool2d(3, 2, 1)
    elif c.pool_type == 'max':
        pool_layer = nn.MaxPool2d(3, 2, 1)
    else:
        pool_layer = nn.Identity()

    z_list = []
    parallel_jac_list = []
    for idx, (h, parallel_flow, c_cond) in enumerate(zip(h_list, parallel_flows, c.c_conds)):
        y = pool_layer(h)
        B, _, H, W = y.shape
        cond = positionalencoding2d(c_cond, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
        z, jac = parallel_flow(y, [cond, ])
        z_list.append(z)
        parallel_jac_list.append(jac)

    z_list, fuse_jac = fusion_flow(z_list)
    jac = fuse_jac + sum(parallel_jac_list)

    # 检查每个张量的形状
    shapes = [z.shape for z in z_list]

    #print("张量的形状:", shapes)
    #保存z_list到txt就是预测分数
    # 将 z_list 中的每个张量转为 numpy 数组
    #z_array = np.array([z.cpu().detach().numpy() for z in z_list])

    # 使用 numpy 保存到 txt 文件中
    #np.savetxt('y_score.txt', z_array, fmt='%d')  # %d 表示整数格式

    return z_list, jac


def train_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow, params, optimizer, warmup_scheduler,
                     decay_scheduler, scaler=None):
    parallel_flows = [parallel_flow.train() for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.train()

    for sub_epoch in range(c.sub_epochs):
        epoch_loss = 0.
        image_count = 0
        for idx, (image, _, _) in enumerate(loader):
            optimizer.zero_grad()
            image = image.to(c.device)
            if scaler:
                with autocast():
                    z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)
                    loss = 0.
                    for z in z_list:
                        loss += 0.5 * torch.sum(z ** 2, (1, 2, 3))
                    loss = loss - jac
                    loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 2)
                scaler.step(optimizer)
                scaler.update()
            else:
                z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)
                loss = 0.
                for z in z_list:
                    loss += 0.5 * torch.sum(z ** 2, (1, 2, 3))
                loss = loss - jac
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 2)
                optimizer.step()
            epoch_loss += t2np(loss)
            image_count += image.shape[0]
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if warmup_scheduler:
            warmup_scheduler.step()
        if decay_scheduler:
            decay_scheduler.step()

        mean_epoch_loss = epoch_loss / image_count
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
              'Epoch {:d}.{:d} train loss: {:.3e}\tlr={:.2e}'.format(
                  epoch, sub_epoch, mean_epoch_loss, lr))


def inference_meta_epoch(c, epoch, loader, extractor, parallel_flows, fusion_flow):
    parallel_flows = [parallel_flow.eval() for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.eval()
    epoch_loss = 0.
    image_count = 0
    gt_label_list = list()
    gt_mask_list = list()
    outputs_list = [list() for _ in parallel_flows]
    size_list = []
    start = time.time()
    print(f'加载的数据集大小: {len(loader.dataset)}')

    with torch.no_grad():
        for idx, (image, label, mask) in enumerate(loader):
            #print(f'批次 {idx} size: {image.shape[0]}')  # 打印当前批次的大小
            if image.shape[0] == 0:
                print("Warning: 警告")
            image = image.to(c.device)
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))

            z_list, jac = model_forward(c, extractor, parallel_flows, fusion_flow, image)

            loss = 0.
            for lvl, z in enumerate(z_list):
                if idx == 0:
                    size_list.append(list(z.shape[-2:]))
                logp = - 0.5 * torch.mean(z ** 2, 1)
                outputs_list[lvl].append(logp)
                loss += 0.5 * torch.sum(z ** 2, (1, 2, 3))

            loss = loss - jac
            loss = loss.mean()
            epoch_loss += t2np(loss)
            image_count += image.shape[0]
        #print("垃圾image_count:",image_count)
        mean_epoch_loss = epoch_loss / image_count
        fps = len(loader.dataset) / (time.time() - start)
        print(datetime.datetime.now().strftime("[%Y-%m-%d-%H:%M:%S]"),
              'Epoch {:d}   test loss: {:.3e}\tFPS: {:.1f}'.format(
                  epoch, mean_epoch_loss, fps))

    return gt_label_list, gt_mask_list, outputs_list, size_list

def train(c):
    if c.wandb_enable:
        wandb.finish()
        wandb_run = wandb.init(
            project='65001-msflow',
            group=c.version_name,
            name=c.class_name)

    Dataset = MVTecDataset if c.dataset == 'mvtec' else VisADataset

    train_dataset = Dataset(c, is_train=True)
    #print(f'训练集大小: {len(train_dataset)}')  # 打印训练集大小
    test_dataset = Dataset(c, is_train=False)
    #print(f'测试集大小: {len(test_dataset)}')  # 打印测试集大小
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True,
                                               num_workers=c.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False,
                                              num_workers=c.workers, pin_memory=True)
    #print(f'数据集大小: {len(test_dataset)}')
    extractor, output_channels = build_extractor(c)
    extractor = extractor.to(c.device).eval()
    parallel_flows, fusion_flow = build_msflow_model(c, output_channels)
    parallel_flows = [parallel_flow.to(c.device) for parallel_flow in parallel_flows]
    fusion_flow = fusion_flow.to(c.device)
    # if c.wandb_enable:
    #     for idx, parallel_flow in enumerate(parallel_flows):
    #         wandb.watch(parallel_flow, log='all', log_freq=100, idx=idx)
    #     wandb.watch(fusion_flow, log='all', log_freq=100, idx=len(parallel_flows))
    params = list(fusion_flow.parameters())
    for parallel_flow in parallel_flows:
        params += list(parallel_flow.parameters())

    optimizer = torch.optim.Adam(params, lr=c.lr)
    if c.amp_enable:
        scaler = GradScaler()

    det_auroc_obs = Score_Observer('Det.AUROC', c.meta_epochs)
    loc_auroc_obs = Score_Observer('Loc.AUROC', c.meta_epochs)
    loc_pro_obs = Score_Observer('Loc.PRO', c.meta_epochs)

    start_epoch = 0
    if c.mode == 'test':
        start_epoch = load_weights(parallel_flows, fusion_flow, c.eval_ckpt)
        epoch = start_epoch + 1
        gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor,
                                                                                    parallel_flows, fusion_flow)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)

        for path in test_dataset.x:
            dir = path
        #file_names = [os.path.basename(path) for path in test_dataset.x]
        file_names = [path.split("MVTec-AD_test/")[-1] for path in test_dataset.x]
        # 保存anomaly_score到文本文件
        #print("dataset",test_dataset.x)
        #保存每张图片的名称和对应的异常分数到文本文件
        with open("anomaly_scores_with_names.txt", 'a') as f:
            #f.write("Image Name, Anomaly Score\n")  # 写入标题
            # 使用字符串分割
            # parts = dir.split("/")
            # category = parts[-4]  # 获取倒数第三个元素
            # f.write(f"{category}\n")
            for name, score in zip(file_names , anomaly_score):
                f.write(f"{name}, {score:.10f}\n")  # 所以分数以普通小数格式写入
        '''
        with open("anomaly_score.txt", 'w') as f:
            # 写入 anomaly_score
            f.write("Anomaly Score:\n")
            #np.savetxt(f, anomaly_score, header="Anomaly Score Values", comments="")
            # 将异常评分保存为普通小数
            np.savetxt(f, anomaly_score, header="Anomaly Score Values", comments="", fmt='%.10f')
        '''
        #   修改 best_det_auroc, best_loc_auroc, best_loc_pro = eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.pro_eval)
        best_det_auroc = eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list, anomaly_score_map_add, anomaly_score_map_mul, c.pro_eval)

        return

    if c.resume:
        last_epoch = load_weights(parallel_flows, fusion_flow, os.path.join(c.ckpt_dir, 'last.pt'), optimizer)
        start_epoch = last_epoch + 1
        print('Resume from epoch {}'.format(start_epoch))

    if c.lr_warmup and start_epoch < c.lr_warmup_epochs:
        if start_epoch == 0:
            start_factor = c.lr_warmup_from
            end_factor = 1.0
        else:
            start_factor = 1.0
            end_factor = c.lr / optimizer.state_dict()['param_groups'][0]['lr']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor,
                                                             end_factor=end_factor, total_iters=(
                                                                                                        c.lr_warmup_epochs - start_epoch) * c.sub_epochs)
    else:
        warmup_scheduler = None

    mile_stones = [milestone - start_epoch for milestone in c.lr_decay_milestones if milestone > start_epoch]

    if mile_stones:
        decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, mile_stones, c.lr_decay_gamma)
    else:
        decay_scheduler = None

    for epoch in range(start_epoch, c.meta_epochs):
        print()
        train_meta_epoch(c, epoch, train_loader, extractor, parallel_flows, fusion_flow, params, optimizer,
                         warmup_scheduler, decay_scheduler, scaler if c.amp_enable else None)

        gt_label_list, gt_mask_list, outputs_list, size_list = inference_meta_epoch(c, epoch, test_loader, extractor,
                                                                                    parallel_flows, fusion_flow)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(c, size_list, outputs_list)

        if c.pro_eval and (epoch > 0 and epoch % c.pro_eval_interval == 0):
            pro_eval = True
        else:
            pro_eval = False

        det_auroc, loc_auroc, loc_pro_auc, \
            best_det_auroc, best_loc_auroc, best_loc_pro = \
            eval_det_loc(det_auroc_obs, loc_auroc_obs, loc_pro_obs, epoch, gt_label_list, anomaly_score, gt_mask_list,
                         anomaly_score_map_add, anomaly_score_map_mul, pro_eval)

        if c.wandb_enable:
            wandb_run.log(
                {
                    'Det.AUROC': det_auroc,
                    'Loc.AUROC': loc_auroc,
                    'Loc.PRO': loc_pro_auc
                },
                step=epoch
            )

        save_weights(epoch, parallel_flows, fusion_flow, 'last', c.ckpt_dir, optimizer)
        if best_det_auroc and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_det', c.ckpt_dir)
        if best_loc_auroc and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_auroc', c.ckpt_dir)
        if best_loc_pro and c.mode == 'train':
            save_weights(epoch, parallel_flows, fusion_flow, 'best_loc_pro', c.ckpt_dir)
