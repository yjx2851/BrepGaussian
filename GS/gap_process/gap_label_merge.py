import numpy as np
import torch
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
import argparse
import os
from pathlib import Path

# --------------------------------------------------
# Step 1. 读取 PCD 文件
# --------------------------------------------------
def read_pcd_with_labels(pcd_path):
    """读取带 label 的 PCD 文件，返回点坐标与 label"""
    with open(pcd_path, 'r') as f:
        lines = f.readlines()

    # 找到数据开始的位置
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("DATA"):
            start = i + 1
            break

    data = np.loadtxt(lines[start:])
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    label = data[:, 3].astype(int)
    edge = data[:, 4]

    points = np.vstack([x, y, z]).T
    return points, label, edge, lines[:start]

# --------------------------------------------------
# Step 2. Chamfer Distance 计算
# --------------------------------------------------
def chamfer_distance(pc1, pc2):
    """计算两组点云的 Chamfer Distance"""
    pc1 = torch.tensor(pc1, dtype=torch.float32)
    pc2 = torch.tensor(pc2, dtype=torch.float32)

    dist = torch.cdist(pc1.unsqueeze(0), pc2.unsqueeze(0)).squeeze(0)
    cd = torch.mean(torch.min(dist, dim=1)[0]) + torch.mean(torch.min(dist, dim=0)[0])
    return cd.item()

# --------------------------------------------------
# Step 3. 主流程
# --------------------------------------------------
def merge_labels_by_cd(points, labels, max_points=300, threshold=0.005, min_points=150):
    """按 CD 距离聚类合并标签"""
    # 按 label 分组
    label_groups = defaultdict(list)
    for p, l in zip(points, labels):
        label_groups[l].append(p)
    label_groups = {k: np.array(v) for k, v in label_groups.items()}

    # 过滤掉点数少于 min_points 的标签
    valid_labels = {k: v for k, v in label_groups.items() if len(v) >= min_points}
    invalid_labels = set(label_groups.keys()) - set(valid_labels.keys())
    
    # if len(invalid_labels) > 0:
        # print(f"[INFO] 过滤掉 {len(invalid_labels)} 个点数少于 {min_points} 的标签: {sorted(invalid_labels)}")
    
    label_groups = valid_labels
    label_keys = list(label_groups.keys())
    n = len(label_keys)

    # 如果没有有效标签，直接返回
    if n == 0:
        valid_mask = np.zeros(len(labels), dtype=bool)
        new_labels = np.zeros(len(labels), dtype=int)  # 不会被使用，因为valid_mask全为False
        return new_labels, {}, valid_mask

    # 如果只有一个有效标签，不需要聚类
    if n == 1:
        new_label_map = {label_keys[0]: 0}
        valid_mask = np.array([l == label_keys[0] for l in labels])
        # 只对有效标签赋值，无效标签的点会被valid_mask过滤掉，不会保存
        new_labels = np.zeros(len(labels), dtype=int)
        for i, l in enumerate(labels):
            if l == label_keys[0]:
                new_labels[i] = 0
        print(f"[INFO] 只有一个有效标签，无需聚类")
        return new_labels, new_label_map, valid_mask

    # 对每个标签的点云下采样
    for k in label_keys:
        if len(label_groups[k]) > max_points:
            idx = np.random.choice(len(label_groups[k]), max_points, replace=False)
            label_groups[k] = label_groups[k][idx]

    # 计算 CD 距离矩阵
    cd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cd = chamfer_distance(label_groups[label_keys[i]], label_groups[label_keys[j]])
            cd_matrix[i, j] = cd_matrix[j, i] = cd

    print("[INFO] CD 距离矩阵计算完成")

    # 使用层次聚类
    Z = linkage(cd_matrix, method='average')
    clusters = fcluster(Z, t=threshold, criterion='distance')

    print(f"[INFO] 聚类完成，共 {len(set(clusters))} 个新标签")

    # 建立旧标签 -> 新标签映射（只包含有效标签）
    new_label_map = {old: new for old, new in zip(label_keys, clusters)}

    # 替换 labels，并创建掩码标记有效点（属于有效标签的点）
    valid_mask = np.array([l in new_label_map for l in labels])
    # 只对有效标签赋值，无效标签的点会被valid_mask过滤掉，不会保存
    new_labels = np.zeros(len(labels), dtype=int)
    for i, l in enumerate(labels):
        if l in new_label_map:
            new_labels[i] = new_label_map[l]
    
    return new_labels, new_label_map, valid_mask

# --------------------------------------------------
# Step 4. 保存新的 PCD 文件
# --------------------------------------------------
def save_pcd_with_new_labels(save_path, header_lines, points, new_labels, edge):
    """保存新的带合并标签的 PCD 文件"""
    with open(save_path, 'w') as f:
        for line in header_lines:
            if line.startswith("POINTS"):
                f.write(f"POINTS {len(points)}\n")
            elif line.startswith("WIDTH"):
                f.write(f"WIDTH {len(points)}\n")
            else:
                f.write(line)
        # 写入数据
        for p, l, e in zip(points, new_labels, edge):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(l)} {e}\n")
    print(f"[INFO] 新文件已保存至: {save_path}")


# --------------------------------------------------
# Step 5. 单个 PCD 处理（读 -> 合并 -> 过滤 -> 保存）
# --------------------------------------------------
def merge_labels(
    input_pcd_path: str,
    output_pcd_path: str,
    threshold: float = 0.5,
    max_points: int = 300,
    min_points: int = 150,
):
    """
    处理单个 PCD 文件：读取、按 CD 合并标签、过滤无效点、保存。
    返回 (是否成功保存, 旧标签->新标签映射)。
    """
    points, labels, edge, header = read_pcd_with_labels(input_pcd_path)
    new_labels, mapping, valid_mask = merge_labels_by_cd(
        points, labels, max_points, threshold, min_points
    )

    filtered_points = points[valid_mask]
    filtered_labels = new_labels[valid_mask]
    filtered_edge = edge[valid_mask]

    original_count = len(points)
    filtered_count = len(filtered_points)
    removed_count = original_count - filtered_count

    # if removed_count > 0:
    #     print(f"[INFO] 过滤掉 {removed_count} 个无效点（剩余 {filtered_count} 个点）")

    if filtered_count == 0:
        # print(f"[WARNING] 过滤后没有有效点，跳过保存")
        return False, mapping

    out_dir = os.path.dirname(output_pcd_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_pcd_with_new_labels(output_pcd_path, header, filtered_points, filtered_labels, filtered_edge)

    # print(f"[INFO] 旧标签 -> 新标签 映射：")
    # for k, v in sorted(mapping.items()):
    #     print(f"  {k} → {v}")

    return True, mapping


# --------------------------------------------------
# Step 6. 命令行入口（单文件）
# --------------------------------------------------
if __name__ == "__main__":
    threshold = 0.5
    min_points = 150
    parser = argparse.ArgumentParser(description="使用 Chamfer Distance 合并点云标签（单文件）")
    parser.add_argument("--input_pcd", type=str, required=True, help="输入 PCD 文件路径")
    parser.add_argument("--output_pcd", type=str, required=True, help="输出 PCD 文件路径")
    parser.add_argument("--threshold", type=float, default=threshold, help="CD 聚类距离阈值")
    parser.add_argument("--max_points", type=int, default=300, help="每个标签的下采样点数")
    parser.add_argument("--min_points", type=int, default=min_points, help="标签的最小点数，少于该点数的标签将被过滤")

    args = parser.parse_args()

    merge_labels(
        input_pcd_path=args.input_pcd,
        output_pcd_path=args.output_pcd,
        threshold=args.threshold,
        max_points=args.max_points,
        min_points=args.min_points,
    )
