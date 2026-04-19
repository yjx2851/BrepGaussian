#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
import random
import math

def triplet_loss_for_segmentation(
    mask_image: torch.Tensor, 
    gt_mask: torch.Tensor, 
    gt_edge: torch.Tensor = None,
    num_anchors_per_mask: int = 64,  # 每个有效Mask采样的锚点数量
    margin: float = 0.3,              # 三元组间隔（需根据任务调参，建议0.2~0.5）
    num_neg_candidates: int = 10      # 为每个锚点采样的负样本候选数（选最优负样本）
) -> torch.Tensor:
    """
    核心：在单个视角内部，基于局部标签构建锚点-正样本-负样本，优化面片特征区分度。
    
    Args:
        mask_image (Tensor): 形状(N, H, W)，每个像素的N维特征向量（与surface_loss一致）
        gt_mask (Tensor): 形状(1, H, W)，当前视角的局部分割标签（仅该视角内有效）
        gt_edge (Tensor): 形状(1, H, W)或(H, W)，边缘掩码（过滤边缘像素，与surface_loss一致）
        num_anchors_per_mask: 每个有效Mask采样的锚点数量（控制计算量）
        margin: 三元组间隔，需调参（目标：d_ap + margin < d_an）
        num_neg_candidates: 为每个锚点采样的负样本候选数（选距离最近的负样本，增强约束）
    
    Returns:
        Tensor: 标量Loss，可直接用于反向传播
    """
    # Part 1. 预处理：与surface_loss完全对齐 --------------------------
    features = mask_image   # (N, H, W)：像素级特征
    labels_map = gt_mask    # (1, H, W)：当前视角的局部标签
    N, H, W = features.shape

    # Step 1:过滤边缘像素：将边缘像素的标签设为0（背景），避免边缘噪声影响
    if gt_edge is not None:
        gt_edge_2d = gt_edge.squeeze(0) if gt_edge.dim() == 3 else gt_edge  # (H, W)
        non_edge_mask = (gt_edge_2d == 0)  # 非边缘区域掩码
        labels_2d = labels_map.squeeze(0) * non_edge_mask  # (H, W)：过滤后的标签图
    else:
        labels_2d = labels_map.squeeze(0)  # (H, W)：无边缘过滤时直接挤压

    # Step 2:特征归一化：用cosine相似性计算距离（与surface_loss一致，保证特征尺度统一）
    features_normalized = F.normalize(features, dim=0)  # (N, H, W)：沿特征维度归一化

    # Step 3:提取有效标签：排除背景（0），且所属Mask像素数在128~10000之间（继承surface_loss过滤逻辑）
    unique_labels = torch.unique(labels_2d)
    # unique_labels = unique_labels[unique_labels != 0]  # 背景的效果 排除背景标签
    valid_labels = []
    label_to_pixels = {}  # 存储每个有效标签对应的像素坐标（(y, x)）
    for label in unique_labels:
        label_mask = (labels_2d == label)
        num_pixels = label_mask.sum().item()
        
        # 过滤过小/过大的Mask（避免噪声或背景类）
        if 100 <= num_pixels <= 10000 and not label_mask[H-1, W-1]:  # 排除图像角落的Mask（可能是噪声）
            valid_labels.append(label)
            # 存储该标签的所有非边缘像素坐标（(y, x)格式，便于后续采样）
            y_coords, x_coords = torch.where(label_mask)
            label_to_pixels[label] = torch.stack([y_coords, x_coords], dim=1).to(features.device)

    # 若无有效标签（如全背景），返回0损失（避免梯度错误）
    if len(valid_labels) < 2:         # 至少需要2个有效标签才能构建负样本（锚点+正+负）
        return features.mean() * 0.0  # 保持张量设备和 dtype 一致


    # Part 2:构建批量三元组：锚点-正样本-负样本 --------------------------
    triplets = []  # 存储批量三元组：[(anchor_feat, pos_feat, neg_feat), ...]
    for label in valid_labels:
        # 获取当前标签的所有像素坐标
        pixels = label_to_pixels[label]  # (M, 2)，M为该标签的像素数
        if pixels.size(0) < 2:           # 至少需要2个像素（锚点+正样本）
            continue

        # 为当前标签采样锚点（控制数量，避免计算量过大）
        num_anchors = min(num_anchors_per_mask, pixels.size(0))
        anchor_indices = torch.randperm(pixels.size(0), device=pixels.device)[:num_anchors]
        anchor_pixels = pixels[anchor_indices]  # (num_anchors, 2)：锚点的(y, x)坐标

        # 为每个锚点采样正样本和负样本
        for (y_a, x_a) in anchor_pixels:
            # -------------------------- 采样正样本：同一标签的其他像素 --------------------------
            # 排除锚点自身，从当前标签的其他像素中随机选1个
            anchor_coord = torch.tensor([[y_a, x_a]], device=pixels.device)  # 构造形状为(1, 2)的锚点坐标
            row_mask = ~torch.all(pixels == anchor_coord, dim=1)  # 按行比较，生成“是否不等于锚点”的掩码
            pos_candidates = pixels[row_mask]  # 筛选出不等于锚点的坐标行
            if pos_candidates.numel() == 0:
                continue  # 无有效正样本，跳过当前锚点
            pos_candidates = pos_candidates.view(-1, 2)  # 重塑为(N, 2)，此时元素数一定是2的倍数
            
            pos_idx = torch.randint(0, pos_candidates.size(0), (1,), device=pixels.device).item()
            y_p, x_p = pos_candidates[pos_idx]

            # -------------------------- 采样负样本：其他标签的像素 --------------------------
            # 从所有其他有效标签中采样候选负样本（增加约束难度：选距离锚点最近的负样本）
            neg_candidates = []
            for neg_label in valid_labels:
                if neg_label == label:
                    continue  # 排除当前标签（负样本必须来自不同标签）
                neg_pixels = label_to_pixels[neg_label]
                # 为每个负标签采样少量候选（避免遍历所有负像素）
                num_neg_sample = min(5, neg_pixels.size(0))  # 每个负标签采样5个候选
                neg_idx = torch.randperm(neg_pixels.size(0), device=pixels.device)[:num_neg_sample]
                neg_candidates.append(neg_pixels[neg_idx])
            if not neg_candidates:
                continue  # 无可用负样本（极端情况，跳过该锚点）
            neg_candidates = torch.cat(neg_candidates, dim=0)  # (K, 2)，K为负样本候选总数

            # 计算锚点与所有负样本候选的cosine距离，选距离最近的负样本（ hardest negative mining）
            # 锚点特征：(N,)
            anchor_feat = features_normalized[:, y_a, x_a]
            # 所有负样本候选特征：(K, N)
            neg_feats = features_normalized[:, neg_candidates[:, 0], neg_candidates[:, 1]].T
            # 计算cosine距离（1 - cosine相似性，距离越小越相似）
            cos_sim = torch.matmul(neg_feats, anchor_feat)  # (K,)：cosine相似性
            neg_distances = 1 - cos_sim  # (K,)：cosine距离
            # 选距离最近的负样本（最难分的负样本，增强约束效果）
            hardest_neg_idx = torch.argmin(neg_distances).item()
            y_n, x_n = neg_candidates[hardest_neg_idx]

            # -------------------------- 收集三元组特征 --------------------------
            anchor_feat = features_normalized[:, y_a, x_a]  # (N,)
            pos_feat = features_normalized[:, y_p, x_p]    # (N,)
            neg_feat = features_normalized[:, y_n, x_n]    # (N,)
            triplets.append(torch.stack([anchor_feat, pos_feat, neg_feat], dim=0))  # (3, N)


    # 若未构建出有效三元组，返回0损失
    if not triplets:
        return features.mean() * 0.0
    # 批量三元组特征：(T, 3, N)，T为三元组数量
    triplet_batch = torch.stack(triplets, dim=0)
    anchors = triplet_batch[:, 0, :]  # (T, N)：所有锚点特征
    positives = triplet_batch[:, 1, :]  # (T, N)：所有正样本特征
    negatives = triplet_batch[:, 2, :]  # (T, N)：所有负样本特征


    # -------------------------- 3. 计算三元组Loss --------------------------
    # 计算cosine距离（1 - 点积，因特征已归一化）
    d_ap = 1 - torch.sum(anchors * positives, dim=1)  # (T,)：锚点-正样本距离
    d_an = 1 - torch.sum(anchors * negatives, dim=1)  # (T,)：锚点-负样本距离

    # 三元组Loss核心公式：max(0, d_ap - d_an + margin)
    # 仅当d_ap >= d_an - margin时，产生损失（需优化：拉近锚点-正，推远锚点-负）
    raw_loss = d_ap - d_an + margin
    valid_loss = F.relu(raw_loss)  # 过滤无损失的三元组（raw_loss < 0时为0）

    # 平均损失（对所有有效三元组取平均）
    total_loss = valid_loss.mean()
    return total_loss

def triplet_loss_for_segmentation_matrix(
    mask_image: torch.Tensor, 
    gt_mask: torch.Tensor, 
    gt_edge: torch.Tensor = None,
    num_anchors_per_mask: int = 64,
    margin: float = 0.3,
    num_neg_candidates: int = 10
) -> torch.Tensor:
    """
    矩阵化版本的三元组损失函数，与原始循环版本逻辑完全一致但速度更快。
    """
    # Part 1. 预处理：与原始版本完全一致
    features = mask_image
    labels_map = gt_mask
    N, H, W = features.shape

    # Step 1: 过滤边缘像素
    if gt_edge is not None:
        gt_edge_2d = gt_edge.squeeze(0) if gt_edge.dim() == 3 else gt_edge
        non_edge_mask = (gt_edge_2d == 0)
        labels_2d = labels_map.squeeze(0) * non_edge_mask
    else:
        labels_2d = labels_map.squeeze(0)

    # Step 2: 特征归一化
    features_normalized = F.normalize(features, dim=0)

    # Step 3: 提取有效标签
    unique_labels = torch.unique(labels_2d)
    valid_labels = []
    label_to_pixels = {}
    
    for label in unique_labels:
        label_mask = (labels_2d == label)
        num_pixels = label_mask.sum().item()
        
        if 100 <= num_pixels <= 10000 and not label_mask[H-1, W-1]:
            valid_labels.append(label)
            y_coords, x_coords = torch.where(label_mask)
            label_to_pixels[label] = torch.stack([y_coords, x_coords], dim=1).to(features.device)

    if len(valid_labels) < 2:
        return features.mean() * 0.0

    # Part 2: 矩阵化构建批量三元组
    all_anchor_features = []
    all_pos_features = []
    all_neg_features = []
    
    # 为每个有效标签处理
    for label in valid_labels:
        pixels = label_to_pixels[label]  # (M, 2)
        if pixels.size(0) < 2:
            continue
            
        M = pixels.size(0)
        num_anchors = min(num_anchors_per_mask, M)
        
        # Step 1: 采样锚点
        if M == num_anchors:
            anchor_indices = torch.arange(M, device=pixels.device)
        else:
            perm = torch.randperm(M, device=pixels.device)
            anchor_indices = perm[:num_anchors]
        
        anchor_pixels = pixels[anchor_indices]  # (num_anchors, 2)
        
        # Step 2: 采样正样本
        pos_pixels_list = []
        valid_anchor_indices = []
        
        for i, anchor_pixel in enumerate(anchor_pixels):
            # 找到当前标签中不等于锚点的像素
            diff_mask = ~((pixels[:, 0] == anchor_pixel[0]) & (pixels[:, 1] == anchor_pixel[1]))
            candidate_pixels = pixels[diff_mask]
            
            if len(candidate_pixels) > 0:
                # 随机选择一个正样本
                idx = torch.randint(0, len(candidate_pixels), (1,), device=pixels.device)
                pos_pixels_list.append(candidate_pixels[idx])
                valid_anchor_indices.append(i)
        
        if not pos_pixels_list:  # 没有找到有效正样本
            continue
            
        # 只保留有有效正样本的锚点
        valid_anchor_indices = torch.tensor(valid_anchor_indices, device=pixels.device)
        anchor_pixels = anchor_pixels[valid_anchor_indices]
        
        # 堆叠正样本像素
        if len(pos_pixels_list) > 1:
            pos_pixels = torch.cat(pos_pixels_list, dim=0)
        else:
            pos_pixels = pos_pixels_list[0].unsqueeze(0)
        
        if pos_pixels.dim() == 1:
            pos_pixels = pos_pixels.unsqueeze(0)
        
        # Step 3: 采样负样本候选
        neg_candidates_list = []
        for neg_label in valid_labels:
            if neg_label == label:
                continue
            neg_pixels = label_to_pixels[neg_label]
            # 从每个负标签中采样固定数量候选
            num_neg_from_label = min(5, len(neg_pixels))
            if num_neg_from_label > 0:
                neg_indices = torch.randperm(len(neg_pixels), device=pixels.device)[:num_neg_from_label]
                neg_candidates_list.append(neg_pixels[neg_indices])
        
        if not neg_candidates_list:
            continue
            
        neg_candidates = torch.cat(neg_candidates_list, dim=0)  # (K, 2)
        
        # 计算锚点特征
        anchor_y = anchor_pixels[:, 0]
        anchor_x = anchor_pixels[:, 1]
        anchor_feats = features_normalized[:, anchor_y, anchor_x].T  # (num_valid_anchors, N)
        
        # 计算正样本特征
        pos_y = pos_pixels[:, 0]
        pos_x = pos_pixels[:, 1]
        pos_feats = features_normalized[:, pos_y, pos_x].T  # (num_valid_anchors, N)
        
        # 计算负样本候选特征
        neg_y = neg_candidates[:, 0]
        neg_x = neg_candidates[:, 1]
        neg_candidate_feats = features_normalized[:, neg_y, neg_x].T  # (K, N)
        
        # Step 4: Hard negative mining (矩阵化)
        # 计算锚点与所有负样本候选的cosine相似度
        similarity = torch.matmul(anchor_feats, neg_candidate_feats.T)  # (num_valid_anchors, K)
        
        # 找到每个锚点的最难负样本（距离最近的 = 相似度最高的）
        hardest_neg_indices = torch.argmax(similarity, dim=1)  # (num_valid_anchors,)
        
        # 提取最难负样本特征
        neg_feats = neg_candidate_feats[hardest_neg_indices]  # (num_valid_anchors, N)
        
        # 收集三元组
        all_anchor_features.append(anchor_feats)
        all_pos_features.append(pos_feats)
        all_neg_features.append(neg_feats)
    
    # 如果没有构建出有效三元组，返回0损失
    if not all_anchor_features:
        return features.mean() * 0.0
    
    # 合并所有三元组
    anchors = torch.cat(all_anchor_features, dim=0)  # (T, N)
    positives = torch.cat(all_pos_features, dim=0)   # (T, N)
    negatives = torch.cat(all_neg_features, dim=0)   # (T, N)
    
    # Part 3: 计算三元组Loss
    # 计算cosine距离
    d_ap = 1 - torch.sum(anchors * positives, dim=1)  # (T,)
    d_an = 1 - torch.sum(anchors * negatives, dim=1)  # (T,)
    
    # 三元组Loss
    raw_loss = d_ap - d_an + margin
    valid_loss = F.relu(raw_loss)
    
    total_loss = valid_loss.mean()
    return total_loss

import torch
import torch.nn.functional as F
import math

def triplet_loss_for_line(
    mask_image: torch.Tensor, 
    gt_mask: torch.Tensor, 
    gt_edge: torch.Tensor = None,
    num_anchors_per_mask: int = 32,  # 减少锚点数量（线mask像素少）
    margin: float = 0.2,             # 缩小margin（线特征区分度可能更低）
    num_neg_candidates: int = 4      # 减少负样本候选（降低计算量）
) -> torch.Tensor:
    """
    适配线mask的三元组损失优化版本：
    1. 降低像素数量阈值，适应线mask稀疏性
    2. 减少采样数量，降低计算开销
    3. 向量化操作优化，提升GPU利用率
    """
    features = mask_image   # (N, H, W)
    labels_map = gt_mask    # (1, H, W)
    N, H, W = features.shape

    # 过滤边缘像素
    if gt_edge is not None:
        gt_edge_2d = gt_edge.squeeze(0) if gt_edge.dim() == 3 else gt_edge
        non_edge_mask = (gt_edge_2d == 0)
        labels_2d = labels_map.squeeze(0) * non_edge_mask
    else:
        labels_2d = labels_map.squeeze(0)

    # 特征归一化
    features_normalized = F.normalize(features, dim=0)

    # 提取有效标签（核心优化：降低线mask的像素数量阈值）
    unique_labels = torch.unique(labels_2d)
    valid_labels = []
    label_to_pixels = {}
    for label in unique_labels:
        if label == 0:
            continue  # 强制排除背景
        label_mask = (labels_2d == label)
        num_pixels = label_mask.sum().item()
        
        # 线mask适配：最小像素从100→20，最大放宽到20000
        if 100 <= num_pixels <= 20000 and not label_mask[H-1, W-1]:
            valid_labels.append(label)
            y_coords, x_coords = torch.where(label_mask)
            label_to_pixels[label] = torch.stack([y_coords, x_coords], dim=1).to(features.device)

    if len(valid_labels) < 2:
        return torch.tensor(0.0, device=features.device)

    # 批量采样锚点、正样本、负样本（向量化优化）
    anchors_list = []
    positives_list = []
    negatives_list = []

    # 预收集所有负标签像素（避免嵌套循环）
    all_neg_pixels = []
    for label in valid_labels:
        all_neg_pixels.append(label_to_pixels[label])
    all_neg_pixels = torch.cat(all_neg_pixels, dim=0)  # (total_neg_pixels, 2)

    for label in valid_labels:
        pixels = label_to_pixels[label]
        if pixels.size(0) < 2:
            continue

        # 采样锚点（数量减半）
        num_anchors = min(num_anchors_per_mask, pixels.size(0))
        anchor_indices = torch.randperm(pixels.size(0), device=pixels.device)[:num_anchors]
        anchor_pixels = pixels[anchor_indices]

        # 正样本采样：同一标签内随机选择（向量化）
        pos_indices = torch.randint(0, pixels.size(0), (num_anchors,), device=pixels.device)
        # 避免与锚点相同（简单处理：若相同则+1取模）
        pos_indices = torch.where(pos_indices == anchor_indices, 
                                 (pos_indices + 1) % pixels.size(0), 
                                 pos_indices)
        pos_pixels = pixels[pos_indices]

        # 负样本采样：其他标签中随机选择（简化hardest negative）
        # 过滤当前标签像素
        is_current_label = torch.any(all_neg_pixels.unsqueeze(1) == pixels.unsqueeze(0), dim=2).any(dim=1)
        valid_neg_pixels = all_neg_pixels[~is_current_label]
        if valid_neg_pixels.size(0) == 0:
            continue

        # 随机采样负样本（替代距离计算排序，大幅提速）
        neg_indices = torch.randint(0, valid_neg_pixels.size(0), (num_anchors,), device=pixels.device)
        neg_pixels = valid_neg_pixels[neg_indices]

        # 收集特征（向量化提取）
        anchors = features_normalized[:, anchor_pixels[:, 0], anchor_pixels[:, 1]].T  # (num_anchors, N)
        positives = features_normalized[:, pos_pixels[:, 0], pos_pixels[:, 1]].T      # (num_anchors, N)
        negatives = features_normalized[:, neg_pixels[:, 0], neg_pixels[:, 1]].T      # (num_anchors, N)

        anchors_list.append(anchors)
        positives_list.append(positives)
        negatives_list.append(negatives)

    if not anchors_list:
        return torch.tensor(0.0, device=features.device)

    # 计算三元组损失
    anchors = torch.cat(anchors_list, dim=0)
    positives = torch.cat(positives_list, dim=0)
    negatives = torch.cat(negatives_list, dim=0)

    d_ap = 1 - torch.sum(anchors * positives, dim=1)  # cosine距离
    d_an = 1 - torch.sum(anchors * negatives, dim=1)
    raw_loss = d_ap - d_an + margin
    total_loss = F.relu(raw_loss).mean()

    return total_loss

def get_loss_weights(iteration, total_iter):
    """
    动态计算surface_loss和triplet_loss的权重
    基础期（0~30%）：surface=1.0, triplet=0.0
    过渡期（30%~70%）：surface从1.0降到0.3，triplet从0升到0.7
    强化期（70%~100%）：surface=0.3, triplet=0.7
    """
    phase1_end = 0.3 * total_iter  # 基础期结束迭代点
    phase2_end = 0.7 * total_iter  # 过渡期结束迭代点
    
    if iteration <= phase1_end:
        return 1.0, 0.0
    elif iteration <= phase2_end:
        # 过渡期：用余弦退火平滑过渡（比线性过渡更稳定）
        ratio = (iteration - phase1_end) / (phase2_end - phase1_end)  # 0~1
        surface_weight = 1.0 - 0.3 * (1 - math.cos(ratio * math.pi))  # 1.0→0.3
        triplet_weight = 0.7 * (1 - math.cos(ratio * math.pi))        # 0→0.7
        return surface_weight, triplet_weight
    else:
        return 0.3, 0.7


def expand_mask_indices(mask, expand_radius=3):
    """
    扩大mask的采样区域，包含周围临近的像素（优化版本）
    
    Args:
        mask: Tensor, (H, W) 二值mask
        expand_radius: int, 扩大的像素范围（半径）
    
    Returns:
        expanded_indices: Tensor, (num_expanded_pixels, 2) 扩大后的像素坐标
    """
    if mask.dim() == 3:
        mask = mask.squeeze(0)  # (H, W)
    
    H, W = mask.shape
    device = mask.device
    
    # 获取原始mask的像素坐标
    original_indices = mask.nonzero(as_tuple=False)  # (num_original, 2)
    if original_indices.size(0) == 0:
        return original_indices
    
    # 使用向量化操作进行膨胀
    # 创建偏移量网格
    dy = torch.arange(-expand_radius, expand_radius + 1, device=device)
    dx = torch.arange(-expand_radius, expand_radius + 1, device=device)
    dy_grid, dx_grid = torch.meshgrid(dy, dx, indexing='ij')
    offsets = torch.stack([dy_grid.flatten(), dx_grid.flatten()], dim=1)  # (num_offsets, 2)
    
    # 为每个原始像素添加所有偏移量
    # original_indices: (num_original, 2), offsets: (num_offsets, 2)
    # 使用广播：original_indices[:, None, :] + offsets[None, :, :]
    expanded_coords = original_indices[:, None, :] + offsets[None, :, :]  # (num_original, num_offsets, 2)
    expanded_coords = expanded_coords.reshape(-1, 2)  # (num_original * num_offsets, 2)
    
    # 过滤边界外的坐标
    valid_mask = (expanded_coords[:, 0] >= 0) & (expanded_coords[:, 0] < H) & \
                 (expanded_coords[:, 1] >= 0) & (expanded_coords[:, 1] < W)
    expanded_coords = expanded_coords[valid_mask]
    
    # 去重（使用torch.unique）
    expanded_coords = torch.unique(expanded_coords, dim=0)
    
    return expanded_coords

def line_constrative_loss_matrix(mask_image, gt_mask, gt_edge=None, num_samples_per_mask: int = 1024, 
                                inter_weight: float = 0.1, expand_radius: int = 5):
    """
    Matrix-optimized version of line_constrative_loss.
    Uses batch operations to eliminate most loops.
    """
    features = mask_image  # (N, H, W)
    labels_map = gt_mask.squeeze(0)  # (H, W)
    N, H, W = features.shape
    
    # Edge processing (if provided)
    if gt_edge is not None:
        if gt_edge.dim() == 3:
            gt_edge_2d = gt_edge.squeeze(0)
        else:
            gt_edge_2d = gt_edge
        non_edge_mask = (gt_edge_2d == 0)
        labels_map = labels_map * non_edge_mask
    
    # Normalize features
    features_normalized = F.normalize(features, dim=0)  # (N, H, W)
    
    # Get unique labels, exclude background 0
    unique_labels = torch.unique(labels_map)
    if (unique_labels == 0).any():
        unique_labels = unique_labels[unique_labels != 0]
    
    if unique_labels.numel() == 0:
        return features.mean() * 0.0
    
    # ==================== PART 1: Intra-mask consistency loss ====================
    # Prepare for batch sampling
    valid_masks = []
    sampled_features_list = []
    sampled_counts = []
    
    for label_value in unique_labels:
        mask = (labels_map == label_value)
        num_pixels = int(mask.sum())
        
        # Apply filters (same as original)
        if num_pixels <= 128 or num_pixels > 10000:
            continue
        if mask[H-1, W-1]:
            continue
        
        # Get indices and sample
        mask_indices = mask.nonzero(as_tuple=False)
        if mask_indices.size(0) == 0:
            continue
        
        sample_size = min(num_samples_per_mask, mask_indices.size(0))
        perm = torch.randperm(mask_indices.size(0), device=mask_indices.device)[:sample_size]
        sampled_indices = mask_indices[perm]  # (S, 2)
        
        # Extract features
        sampled_features = features_normalized[:, sampled_indices[:, 0], sampled_indices[:, 1]].T  # (S, N)
        
        valid_masks.append(label_value)
        sampled_features_list.append(sampled_features)
        sampled_counts.append(sampled_features.shape[0])
    
    if not valid_masks:
        return features.mean() * 0.0
    
    # ==================== Matrix computation for intra-mask loss ====================
    intra_losses = []
    for i, features_i in enumerate(sampled_features_list):
        S = features_i.shape[0]
        if S <= 1:
            continue
        
        # Compute similarity matrix
        sim_matrix = features_i @ features_i.T  # (S, S)
        
        # Create mask for upper triangle (excluding diagonal)
        rows, cols = torch.triu_indices(S, S, offset=1, device=features_i.device)
        
        # Compute cosine distances (1 - similarity)
        pairwise_dists = 1.0 - sim_matrix[rows, cols]
        
        # Average for this mask
        intra_losses.append(pairwise_dists.mean())
    
    if not intra_losses:
        return features.mean() * 0.0
    
    intra_loss = torch.stack(intra_losses).mean()
    
    # ==================== PART 2: Inter-mask separation loss ====================
    # Get all non-zero pixels
    all_nonzero_mask = (labels_map > 0)
    all_nonzero_indices = torch.where(all_nonzero_mask)
    
    if all_nonzero_indices[0].numel() == 0:
        return intra_loss
    
    # Sample random points from all non-zero pixels
    all_coords = torch.stack(all_nonzero_indices, dim=1)  # (num_nonzero, 2)
    num_samples = min(num_samples_per_mask * 2, all_coords.shape[0])
    
    if all_coords.shape[0] > num_samples:
        sampled_idx = torch.randperm(all_coords.shape[0], device=all_coords.device)[:num_samples]
        sampled_coords = all_coords[sampled_idx]
    else:
        sampled_coords = all_coords
        num_samples = all_coords.shape[0]
    
    # Extract features and labels
    sampled_features = features_normalized[:, sampled_coords[:, 0], sampled_coords[:, 1]].T  # (num_samples, N)
    sampled_labels = labels_map[sampled_coords[:, 0], sampled_coords[:, 1]]  # (num_samples,)
    
    # ==================== Matrix computation for inter-mask loss ====================
    inter_losses = []
    
    # Compute full similarity matrix
    sim_matrix_all = sampled_features @ sampled_features.T  # (num_samples, num_samples)
    
    # Get unique labels in sampled points
    unique_sampled_labels = torch.unique(sampled_labels)
    unique_sampled_labels = unique_sampled_labels[unique_sampled_labels != 0]
    
    if len(unique_sampled_labels) < 2:
        return intra_loss
    
    # Create label matrix for mask operations
    label_matrix = sampled_labels.unsqueeze(0) == sampled_labels.unsqueeze(1)  # (num_samples, num_samples)
    
    # Process each pair of labels
    for i in range(len(unique_sampled_labels)):
        label_i = unique_sampled_labels[i]
        mask_i = (sampled_labels == label_i)
        
        if not mask_i.any():
            continue
            
        for j in range(i+1, len(unique_sampled_labels)):
            label_j = unique_sampled_labels[j]
            mask_j = (sampled_labels == label_j)
            
            if not mask_j.any():
                continue
            
            # Extract cross-similarity between label_i and label_j
            # This creates a boolean mask for cross-similarities
            cross_mask = mask_i.unsqueeze(1) & mask_j.unsqueeze(0)
            
            # Get all cross-similarities
            cross_sims = sim_matrix_all[cross_mask]
            
            if cross_sims.numel() == 0:
                continue
            
            # Compute inter-mask loss: (1 + average_similarity)
            # Equivalent to (2 - (1 - average_similarity))
            avg_cross_sim = cross_sims.mean()
            inter_loss = 1.0 + avg_cross_sim  # Range [0, 2], higher = more similar = worse
            inter_losses.append(inter_loss)
    
    # Combine losses
    if inter_losses:
        inter_loss = torch.stack(inter_losses).mean()
        total_loss = intra_loss + inter_weight * inter_loss
    else:
        total_loss = intra_loss
    
    return total_loss

# constrative loss
def line_constrative_loss(mask_image, gt_mask, gt_edge = None, num_samples_per_mask: int = 1024, inter_weight: float = 0.1, expand_radius: int = 5):
    """
    Compute intra-mask consistency loss and inter-mask separation loss for N-dimensional features.
    Uses cosine similarity to match the PCD export method.

    For each instance label in gt_mask (excluding 0 if present):
    1. Compute intra-mask consistency: minimize cosine distance within each mask region
    2. Compute inter-mask separation: maximize cosine distance between different mask regions
    Lower loss encourages similar features within each mask region and different features between regions.

    Args:
        mask_image (Tensor): shape (N, H, W) - N-dimensional feature vectors per pixel
        gt_mask (Tensor): shape (1, H, W) with integer labels for different masks
        gt_edge (Tensor): shape (1, H, W) or (H, W), optional edge mask
        num_samples_per_mask (int): number of samples per mask for computation
        inter_weight (float): weight for inter-mask separation loss
        expand_radius (int): radius to expand sampling region around mask pixels

    Returns:
        Tensor: scalar loss
    """

    features = mask_image  # Already (N, H, W)
    labels_map = gt_mask  # Already (1, H, W)


    # Squeeze to (H, W) for easier processing
    labels_2d = labels_map.squeeze(0)  # (1, H, W) -> (H, W)
    N, H, W = features.shape

    # optional: no need for edge processing    
    # Remove edge pixels from gt_mask: set edge pixels to 0
    if gt_edge is not None:
        # gt_edge should be (1, H, W) or (H, W), convert to (H, W)
        if gt_edge.dim() == 3:
            gt_edge_2d = gt_edge.squeeze(0)
        else:
            gt_edge_2d = gt_edge
        # Create mask for non-edge pixels (where gt_edge is 0)
        non_edge_mask = (gt_edge_2d == 0)
        # Set edge pixels to 0 in the mask (keep non-edge pixels, zero out edge pixels)
        labels_2d = labels_2d * non_edge_mask

    # Normalize features for cosine similarity computation
    features_normalized = torch.nn.functional.normalize(features, dim=0)  # (N, H, W)

    # Get unique instance labels, ignore background label 0 if present
    unique_labels = torch.unique(labels_2d)
    if (unique_labels == 0).any():
        unique_labels = unique_labels[unique_labels != 0]

    # If no valid labels, return zero (preserve graph dtype/device)
    if unique_labels.numel() == 0:
        return features.mean() * 0.0

    # For each mask: sample pixels and minimize the sum of pairwise cosine similarities
    mask_losses = []
    sampled_features_per_mask = []  # store sampled normalized features per valid mask
    for label_value in unique_labels:
        mask = (labels_2d == label_value)  # (H, W)
        num_pixels = int(mask.sum())
        
        # Optional filters like before
        if num_pixels <= 128 or num_pixels > 10000:
            continue
        if mask[H-1, W-1]:
            continue

        # indices within mask: (num_pixels, 2)
        mask_indices = mask.nonzero(as_tuple=False)
        if mask_indices.size(0) == 0:
            continue

        # Randomly sample up to num_samples_per_mask pixels
        sample_size = min(num_samples_per_mask, mask_indices.size(0))
        # Use randperm on the same device
        perm = torch.randperm(mask_indices.size(0), device=mask_indices.device)[:sample_size]
        sampled = mask_indices[perm]  # (S, 2)

        # Gather normalized features and compute pairwise cosine similarities
        sampled_features = features_normalized[:, sampled[:, 0], sampled[:, 1]].T  # (S, N)
        # Since vectors are normalized, dot product equals cosine similarity
        sim_matrix = sampled_features @ sampled_features.T  # (S, S)
        # Exclude diagonal
        S = sim_matrix.shape[0]
        if S <= 1:
            continue
        # Keep for inter-mask separation term
        sampled_features_per_mask.append(sampled_features)
        triu_mask = torch.triu(torch.ones((S, S), dtype=torch.bool, device=sim_matrix.device), diagonal=1)
        # Use cosine distance = 1 - cosine similarity for each pair
        pairwise_dists = (1.0 - sim_matrix)[triu_mask]
        # Average pairwise cosine distance for this mask
        mask_loss = pairwise_dists.mean()
        mask_losses.append(mask_loss)

    if len(mask_losses) == 0:
        return features.mean() * 0.0

    # Average across masks for intra-mask term
    intra_loss = torch.stack(mask_losses).mean()

    # Inter-mask separation loss: sample random points from all non-zero pixels
    inter_mask_losses = []
    
    # Get all non-zero pixel coordinates from the entire image
    all_nonzero_mask = (labels_2d > 0)  # All non-zero pixels in the image
    all_nonzero_indices = torch.where(all_nonzero_mask)
    all_nonzero_coords = torch.stack(all_nonzero_indices, dim=1)  # (num_nonzero, 2)
    
    if len(all_nonzero_coords) > 0:
        # Sample random points from all non-zero pixels
        num_samples = min(num_samples_per_mask * 2, len(all_nonzero_coords))  # Sample more points for better coverage
        if len(all_nonzero_coords) > num_samples:
            sampled_idx = torch.randperm(len(all_nonzero_coords))[:num_samples]
            sampled_coords = all_nonzero_coords[sampled_idx]
        else:
            sampled_coords = all_nonzero_coords
            
        # Get features for all sampled points
        # 特征和标签
        sampled_features = features_normalized[:, sampled_coords[:, 0], sampled_coords[:, 1]].T  # (num_samples, N)
        sampled_labels = labels_2d[sampled_coords[:, 0], sampled_coords[:, 1]]  # (num_samples,)
        
        # Compute pairwise similarities between all sampled points
        sims = sampled_features @ sampled_features.T  # (num_samples, num_samples)
        
        # Create masks for different label pairs
        for i, label_i in enumerate(unique_labels):
            for j, label_j in enumerate(unique_labels):
                if i >= j:  # Avoid duplicate pairs and self-pairs
                    continue
                    
                # Find points belonging to label_i and label_j
                # 找到索引
                mask_i = (sampled_labels == label_i)
                mask_j = (sampled_labels == label_j)
                
                if mask_i.sum() > 0 and mask_j.sum() > 0:
                    # Get features for each label
                    features_i = sampled_features[mask_i]  # (num_i, N)
                    features_j = sampled_features[mask_j]  # (num_j, N)
                    
                    # Compute cross-similarity between different labels
                    cross_sims = features_i @ features_j.T  # (num_i, num_j) - cosine similarity [-1, 1]
                    # Convert to cosine distance [0, 2] where 0 = identical, 2 = opposite
                    cos_distances = 1 - cross_sims  # (num_i, num_j) - cosine distance [0, 2]
                    # We want to maximize distance between different masks, so minimize (2 - distance)
                    # This gives us a loss that increases as similarity increases
                    inter_loss = (2 - cos_distances).mean()  # Range [0, 2], higher = more similar = worse
                    inter_mask_losses.append(inter_loss)

    # Combine intra-mask consistency and inter-mask separation losses
    if len(inter_mask_losses) > 0:
        inter_loss = torch.stack(inter_mask_losses).mean()
        total_loss = intra_loss + inter_weight * inter_loss
    else:
        total_loss = intra_loss

    return total_loss

## mean loss
# def surface_loss(mask_image, gt_mask, debug_id=None):
#     """
#     Compute intra-mask consistency loss and inter-mask separation loss for N-dimensional features.
#     Uses cosine similarity to match the PCD export method.

#     For each instance label in gt_mask (excluding 0 if present):
#     1. Compute intra-mask consistency: minimize cosine distance within each mask region
#     2. Compute inter-mask separation: maximize cosine distance between different mask regions
#     Lower loss encourages similar features within each mask region and different features between regions.

#     Args:
#         mask_image (Tensor): shape (N, H, W) - N-dimensional feature vectors per pixel
#         gt_mask (Tensor): shape (1, H, W) with integer labels for different masks

#     Returns:
#         Tensor: scalar loss
#     """
#     # Ensure mask_image is (N, H, W)
#     if mask_image.dim() == 2:
#         # (H, W) -> (1, H, W)
#         features = mask_image.unsqueeze(0)
#     elif mask_image.dim() == 3:
#         features = mask_image  # Already (N, H, W)
#     else:
#         raise ValueError(f"Expected mask_image shape (N, H, W), got {mask_image.shape}")

#     # Ensure gt_mask is (1, H, W)
#     if gt_mask.dim() == 2:
#         labels_map = gt_mask.unsqueeze(0)  # (H, W) -> (1, H, W)
#     elif gt_mask.dim() == 3 and gt_mask.size(0) == 1:
#         labels_map = gt_mask  # Already (1, H, W)
#     else:
#         raise ValueError(f"Expected gt_mask shape (1, H, W), got {gt_mask.shape}")

#     # Ensure same device
#     labels_map = labels_map.to(device=features.device)

#     # Safety check: ensure spatial dims match
#     if features.size(-2) != labels_map.size(-2) or features.size(-1) != labels_map.size(-1):
#         raise ValueError("mask_image and gt_mask spatial dimensions must match")

#     # Squeeze to (H, W) for easier processing
#     labels_2d = labels_map.squeeze(0)  # (1, H, W) -> (H, W)
#     N, H, W = features.shape

#     # Normalize features for cosine similarity computation
#     features_normalized = torch.nn.functional.normalize(features, dim=0)  # (N, H, W)

#     # Get unique instance labels, ignore background label 0 if present
#     unique_labels = torch.unique(labels_2d)
#     if (unique_labels == 0).any():
#         unique_labels = unique_labels[unique_labels != 0]

#     # If no valid labels, return zero (preserve graph dtype/device)
#     if unique_labels.numel() == 0:
#         return features.mean() * 0.0

#     # Collect valid masks and their mean feature vectors
#     valid_masks = []
#     mask_means = []
    
#     for label_value in unique_labels:
#         mask = (labels_2d == label_value)  # (H, W)
#         num_pixels = mask.sum()
        
#         # Filter out masks that are too small or too large
#         if num_pixels <= 100 or num_pixels > 10000:
#             continue
        
#         # Check if mask contains the bottom-right corner (unreasonable mask)
#         bottom_right_corner = mask[H-1, W-1]  # bottom-right corner
#         if bottom_right_corner:
#             continue

#         # Get feature vectors within mask: shape (num_pixels, N)
#         mask_indices = mask.nonzero(as_tuple=False)  # (num_pixels, 2)
#         if mask_indices.size(0) == 0:
#             continue
            
#         # Extract features for this mask: (num_pixels, N)
#         mask_features = features_normalized[:, mask_indices[:, 0], mask_indices[:, 1]].T  # (num_pixels, N)
#         mean_features = mask_features.mean(dim=0)  # (N,)
#         mean_features = torch.nn.functional.normalize(mean_features, dim=0)  # Normalize mean
        
#         valid_masks.append(mask)
#         mask_means.append(mean_features)

#     if len(valid_masks) == 0:
#         return features.mean() * 0.0

#     # 1. Intra-mask consistency loss (minimize cosine distance within each mask)
#     intra_losses = []
#     for i, mask in enumerate(valid_masks):
#         mask_indices = mask.nonzero(as_tuple=False)
#         if mask_indices.size(0) == 0:
#             continue
            
#         # Extract features for this mask: (num_pixels, N)
#         mask_features = features_normalized[:, mask_indices[:, 0], mask_indices[:, 1]].T  # (num_pixels, N)
#         mean_features = mask_means[i]  # (N,)
        
#         # Compute cosine similarity with mean for each pixel
#         # Cosine similarity = dot product for normalized vectors
#         similarities = torch.mm(mask_features, mean_features.unsqueeze(1)).squeeze(1)  # (num_pixels,)
        
#         # Convert to cosine distance: 1 - cosine_similarity
#         cosine_distances = 1.0 - similarities
#         intra_loss = cosine_distances.mean()  # Average cosine distance from mean
#         intra_losses.append(intra_loss)
    
#     if len(intra_losses) == 0:
#         return features.mean() * 0.0
        
#     intra_loss_total = torch.stack(intra_losses).mean()

#     # 2. Inter-mask separation loss (maximize cosine distance between different masks)
#     if len(mask_means) > 1:
#         mask_means_tensor = torch.stack(mask_means)  # (num_masks, N)
        
#         # Compute pairwise cosine similarities between mask means
#         # Shape: (num_masks, num_masks)
#         pairwise_similarities = torch.mm(mask_means_tensor, mask_means_tensor.T)
        
#         # Remove diagonal (self-comparison) and get upper triangle
#         mask_upper = torch.triu(torch.ones_like(pairwise_similarities), diagonal=1).bool()
#         valid_similarities = pairwise_similarities[mask_upper]
        
#         if valid_similarities.numel() > 0:
#             # Inter-mask loss: minimize the maximum similarity (maximize minimum distance)
#             # This encourages maximum separation between different masks
#             max_similarity = valid_similarities.max()
#             inter_loss = max_similarity  # Penalty for high similarity between different masks
#         else:
#             inter_loss = features.new_zeros(())
#     else:
#         inter_loss = features.new_zeros(())

#     # Combine intra-mask consistency and inter-mask separation
#     # Weight the losses appropriately
#     total_loss = intra_loss_total + 0.1 * inter_loss
    
#     return total_loss

# def surface_loss(mask_image, gt_edge, distance_decay_factor=0.1, min_distance=1.0):
#     """
#     Compute surface loss considering distance to boundary edges.
    
#     For each non-zero pixel in mask_image, compute distance to nearest edge pixel.
#     The loss encourages pixels farther from edges to have darker colors (lower values).
    
#     Args:
#         mask_image (Tensor): shape (N, H, W) - feature vectors per pixel
#         gt_edge (Tensor): shape (1, H, W) or (H, W) - binary edge map (1=edge, 0=non-edge)
#         distance_decay_factor (float): controls how much distance affects the loss
#         min_distance (float): minimum distance to avoid division by zero
    
#     Returns:
#         Tensor: scalar loss
#     """
#     # Ensure gt_edge is 2D
#     if gt_edge.dim() == 3:
#         gt_edge_2d = gt_edge.squeeze(0)
#     else:
#         gt_edge_2d = gt_edge
    
#     N, H, W = mask_image.shape
#     device = mask_image.device
    
#     # Check if there are any edges
#     if gt_edge_2d.sum() == 0:
#         # No edges found, return zero loss
#         return torch.tensor(0.0, device=device, requires_grad=True)
    
#     # Use distance transform for efficiency
#     # Convert to numpy for scipy distance transform
#     edge_np = gt_edge_2d.cpu().numpy().astype(np.uint8)
    
#     # Compute distance transform (distance to nearest edge)
#     from scipy.ndimage import distance_transform_edt
#     distance_map = distance_transform_edt(1 - edge_np)  # Distance to nearest edge
    
#     # Convert back to tensor
#     min_distances = torch.from_numpy(distance_map).float().to(device)
    
#     # Add minimum distance to avoid division by zero
#     min_distances = torch.clamp(min_distances, min=min_distance)
    
#     # Compute distance-based weights (farther = higher weight for darker color)
#     # Use distance with decay factor: farther pixels should have lower values (darker)
#     distance_weights = min_distances * distance_decay_factor  # (H, W) - farther pixels have higher weights
    
#     # Scale up the weights to increase loss magnitude
#     max_distance = min_distances.max()
#     distance_weights = distance_weights / (max_distance * distance_decay_factor + 1.0)  # Normalize to [0, 1]
    
#     # Compute loss for each feature channel
#     total_loss = 0.0
#     for i in range(N):
#         feature_channel = mask_image[i]  # (H, W)
        
#         # Only consider non-zero pixels
#         non_zero_mask = feature_channel != 0
        
#         if non_zero_mask.sum() > 0:
#             # For non-zero pixels, encourage darker colors (lower values) for farther pixels
#             # Loss = feature_value * distance_weight (higher weight for farther pixels)
#             # We want to minimize this, so farther pixels with high feature values get higher loss
#             channel_loss = (feature_channel * distance_weights * non_zero_mask.float()).sum()
#             total_loss += channel_loss
    
#     # Scale up the total loss
#     total_loss = total_loss * 10.0  # Scale factor to increase loss magnitude
    
#     # Normalize by number of non-zero pixels
#     non_zero_pixels = (mask_image != 0).any(dim=0).float().sum()
#     if non_zero_pixels > 0:
#         total_loss = total_loss / non_zero_pixels
    
#     return total_loss



def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

