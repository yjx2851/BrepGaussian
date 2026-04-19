from typing import Optional, List, Dict, Tuple
import os

def find_label_index(header_lines: List[str]) -> Optional[int]:
    fields_line = next((line for line in header_lines if line.startswith("FIELDS")), None)
    if not fields_line:
        return None
    parts = fields_line.strip().split()
    # parts[0] == "FIELDS"
    fields = parts[1:]
    try:
        return fields.index("label")
    except ValueError:
        return None


def count_unique_labels(pcd_path: str, min_points: int = 1) -> int:
    with open(pcd_path, "r", encoding="utf-8", errors="ignore") as f:
        header: List[str] = []
        data_started = False
        label_idx: Optional[int] = None
        label_counts: Dict[str, int] = {}

        for line in f:
            if not data_started:
                header.append(line)
                if line.startswith("DATA"):
                    # Determine label index once header complete
                    label_idx = find_label_index(header)
                    if label_idx is None:
                        raise RuntimeError("label field not found in PCD FIELDS header")
                    data_started = True
                continue

            # Data lines
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if label_idx < len(parts):
                label = parts[label_idx]
                label_counts[label] = label_counts.get(label, 0) + 1
    # Count labels meeting the threshold
    return sum(1 for c in label_counts.values() if c >= min_points)


def _collect_label_counts_and_header(pcd_path: str) -> Tuple[Dict[str, int], List[str], int]:
    """Return (label_counts, header_lines, label_idx)."""
    label_counts: Dict[str, int] = {}
    header: List[str] = []
    label_idx: Optional[int] = None
    data_started = False
    with open(pcd_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not data_started:
                header.append(line)
                if line.startswith("DATA"):
                    label_idx = find_label_index(header)
                    if label_idx is None:
                        raise RuntimeError("label field not found in PCD FIELDS header")
                    data_started = True
                continue
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if label_idx < len(parts):
                label = parts[label_idx]
                label_counts[label] = label_counts.get(label, 0) + 1
    assert label_idx is not None
    return label_counts, header, label_idx


def filter_and_save_with_remapped_labels(
    pcd_path: str,
    out_path: str,
    min_points: int = 100,
    strict_greater: bool = True,
) -> int:
    """
    Filter out labels with counts <= min_points (if strict_greater) or < min_points otherwise,
    remap remaining labels to 1..N, save to out_path, and return kept point count.
    """
    label_counts, header_lines, label_idx = _collect_label_counts_and_header(pcd_path)

    # Choose labels to keep
    if strict_greater:
        keep_labels = {label for label, c in label_counts.items() if c > min_points}
    else:
        keep_labels = {label for label, c in label_counts.items() if c >= min_points}

    # Build discrete mapping old_label -> new consecutive id starting from 1
    sorted_labels = sorted(keep_labels, key=lambda x: (len(x), x))
    label_to_new_id: Dict[str, str] = {old: str(i + 1) for i, old in enumerate(sorted_labels)}

    # Indices of header lines to update
    def _replace_single_value_line(prefix: str, new_value: int, line: str) -> str:
        # Replace the first number after the prefix, preserving other tokens
        parts = line.strip().split()
        if parts and parts[0] == prefix:
            parts[1] = str(new_value)
            return " ".join(parts) + "\n"
        return line

    updated_header: List[str] = []
    data_found = False
    for line in header_lines:
        if line.startswith("DATA"):
            data_found = True
        updated_header.append(line)
    if not data_found:
        raise RuntimeError("Malformed PCD header: DATA line not found")

    # Second pass: filter and remap points
    kept_lines: List[str] = []
    with open(pcd_path, "r", encoding="utf-8", errors="ignore") as f:
        data_started = False
        for line in f:
            if not data_started:
                if line.startswith("DATA"):
                    data_started = True
                continue
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if label_idx < len(parts) and parts[label_idx] in label_to_new_id:
                parts[label_idx] = label_to_new_id[parts[label_idx]]
                kept_lines.append(" ".join(parts))

    kept_points = len(kept_lines)

    # Update WIDTH and POINTS in header using a simple rewrite
    final_header: List[str] = []
    for line in updated_header:
        if line.startswith("WIDTH"):
            final_header.append(_replace_single_value_line("WIDTH", kept_points, line))
        elif line.startswith("POINTS"):
            final_header.append(_replace_single_value_line("POINTS", kept_points, line))
        else:
            final_header.append(line)

    # Write out new PCD
    with open(out_path, "w", encoding="utf-8") as out:
        for line in final_header:
            out.write(line)
        for dl in kept_lines:
            out.write(dl + "\n")

    return kept_points


def _read_points_with_labels(pcd_path: str) -> Tuple[List[str], List[List[str]], int, int]:
    header: List[str] = []
    data_started = False
    label_idx: Optional[int] = None
    xyz_indices: List[int] = []
    rows: List[List[str]] = []
    with open(pcd_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not data_started:
                header.append(line)
                if line.startswith("FIELDS"):
                    fields = line.strip().split()[1:]
                    xyz_indices = [fields.index("x"), fields.index("y"), fields.index("z")]
                    try:
                        label_idx = fields.index("label")
                    except ValueError:
                        raise RuntimeError("label field not found in PCD FIELDS header")
                if line.startswith("DATA"):
                    data_started = True
                continue
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rows.append(s.split())
    assert label_idx is not None
    return header, rows, label_idx, xyz_indices[0]


def _euclidean_distance_sq(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def _cluster_indices_by_radius(coords: List[Tuple[float, float, float]], eps: float) -> List[List[int]]:
    n = len(coords)
    visited = [False] * n
    clusters: List[List[int]] = []
    eps_sq = eps * eps
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        queue = [i]
        cluster = [i]
        while queue:
            j = queue.pop()
            pj = coords[j]
            for k in range(n):
                if visited[k]:
                    continue
                if _euclidean_distance_sq(pj, coords[k]) <= eps_sq:
                    visited[k] = True
                    queue.append(k)
                    cluster.append(k)
        clusters.append(cluster)
    return clusters


def cluster_per_label_and_save(
    pcd_path: str,
    out_path: str,
    eps: float,
) -> int:
    header, rows, label_idx, x_idx = _read_points_with_labels(pcd_path)

    # Collect xyz and labels
    coords: List[Tuple[float, float, float]] = []
    labels: List[int] = []
    # Determine y,z indices by deriving from x index and known FIELDS order
    fields_line = next(line for line in header if line.startswith("FIELDS"))
    fields = fields_line.strip().split()[1:]
    x_i = fields.index("x")
    y_i = fields.index("y")
    z_i = fields.index("z")
    for r in rows:
        coords.append((float(r[x_i]), float(r[y_i]), float(r[z_i])))
        labels.append(int(float(r[label_idx])))

    # Group indices by original label
    from collections import defaultdict

    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        label_to_indices[lab].append(idx)

    max_label = max(labels) if labels else 0
    next_label = max_label + 1

    # Cluster per label and reassign additional clusters
    for lab, inds in label_to_indices.items():
        if len(inds) <= 1:
            continue
        sub_coords = [coords[i] for i in inds]
        clusters = _cluster_indices_by_radius(sub_coords, eps)
        if len(clusters) <= 1:
            continue
        # Keep the largest cluster as original label; others get new labels
        clusters_sorted = sorted(clusters, key=len, reverse=True)
        for cluster_idx, cluster in enumerate(clusters_sorted):
            if cluster_idx == 0:
                continue
            new_lab = next_label
            next_label += 1
            for local_i in cluster:
                global_i = inds[local_i]
                labels[global_i] = new_lab

    # Write updated labels back into rows
    for i, lab in enumerate(labels):
        rows[i][label_idx] = str(lab)

    # Update header POINTS/WIDTH (unchanged count) and write file
    count_points = len(rows)

    def _replace_single_value_line(prefix: str, new_value: int, line: str) -> str:
        parts = line.strip().split()
        if parts and parts[0] == prefix:
            parts[1] = str(new_value)
            return " ".join(parts) + "\n"
        return line

    final_header: List[str] = []
    for line in header:
        if line.startswith("WIDTH"):
            final_header.append(_replace_single_value_line("WIDTH", count_points, line))
        elif line.startswith("POINTS"):
            final_header.append(_replace_single_value_line("POINTS", count_points, line))
        else:
            final_header.append(line)

    with open(out_path, "w", encoding="utf-8") as out:
        for line in final_header:
            out.write(line)
        for r in rows:
            out.write(" ".join(r) + "\n")

    return next_label - 1


def cluster_then_filter_then_remap_save(
    pcd_path: str,
    out_path: str,
    eps: float,
    min_points: int = 50,
    strict_greater: bool = True,
) -> int:
    """
    1) 对每个原始标签进行聚类（基于 eps 半径），每个聚类当作一个“新标签”。
    2) 以聚类为单位过滤：仅保留点数满足阈值的聚类（> 或 >=）。
    3) 对保留下来的聚类重编号为 1..N，并写出新的 PCD。
    返回写出的点数。
    """
    header, rows, label_idx, _ = _read_points_with_labels(pcd_path)

    # 解析字段索引
    fields_line = next(line for line in header if line.startswith("FIELDS"))
    fields = fields_line.strip().split()[1:]
    x_i = fields.index("x")
    y_i = fields.index("y")
    z_i = fields.index("z")

    # 提取坐标与原始标签
    coords: List[Tuple[float, float, float]] = []
    original_labels: List[int] = []
    for r in rows:
        coords.append((float(r[x_i]), float(r[y_i]), float(r[z_i])))
        original_labels.append(int(float(r[label_idx])))

    # 按原始标签分组索引
    from collections import defaultdict

    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(original_labels):
        label_to_indices[lab].append(idx)

    # 对每个标签内做聚类，并为每个聚类分配临时新标签 id（仅用于统计与筛选，后续会重编号）
    point_temp_label: List[int] = [-1] * len(rows)
    temp_label_next = 1
    for _, inds in label_to_indices.items():
        if not inds:
            continue
        sub_coords = [coords[i] for i in inds]
        clusters = _cluster_indices_by_radius(sub_coords, eps)
        for cluster in clusters:
            temp_label = temp_label_next
            temp_label_next += 1
            for local_i in cluster:
                global_i = inds[local_i]
                point_temp_label[global_i] = temp_label

    # 统计每个临时聚类标签的点数
    temp_counts: Dict[int, int] = defaultdict(int)
    for lab in point_temp_label:
        if lab != -1:
            temp_counts[lab] += 1

    # 过滤：仅保留点数达阈值的聚类
    if strict_greater:
        keep_labels = {lab for lab, c in temp_counts.items() if c > min_points}
    else:
        keep_labels = {lab for lab, c in temp_counts.items() if c >= min_points}

    # 重编号：将保留的聚类标签映射为 1..N
    kept_points_rows: List[List[str]] = []
    kept_labels_sorted = sorted(keep_labels)
    temp_to_final: Dict[int, str] = {lab: str(i + 1) for i, lab in enumerate(kept_labels_sorted)}
    for i, r in enumerate(rows):
        tl = point_temp_label[i]
        if tl in temp_to_final:
            r2 = list(r)
            r2[label_idx] = temp_to_final[tl]
            kept_points_rows.append(r2)

    kept_points = len(kept_points_rows)

    # 更新头 WIDTH/POINTS
    def _replace_single_value_line(prefix: str, new_value: int, line: str) -> str:
        parts = line.strip().split()
        if parts and parts[0] == prefix:
            parts[1] = str(new_value)
            return " ".join(parts) + "\n"
        return line

    final_header: List[str] = []
    for line in header:
        if line.startswith("WIDTH"):
            final_header.append(_replace_single_value_line("WIDTH", kept_points, line))
        elif line.startswith("POINTS"):
            final_header.append(_replace_single_value_line("POINTS", kept_points, line))
        else:
            final_header.append(line)

    # 写出新 PCD
    with open(out_path, "w", encoding="utf-8") as out:
        for line in final_header:
            out.write(line)
        for r in kept_points_rows:
            out.write(" ".join(r) + "\n")

    return kept_points


def main() -> None:
    eps = 0.08
    min_points = 150
    batch = 2
    os.makedirs(f"./save_ab_nodensify_filtered_{eps}_{min_points}", exist_ok=True)
    for t in os.listdir(f"./save_densify"):
        # pcd_path = os.path.join(f"./save{batch}_densify", t)
        pcd_path = os.path.join(f"./save_origin", t)
        out_path = os.path.join(f"./save_ab_nodensify_filtered_{eps}_{min_points}", t)
        cluster_then_filter_then_remap_save(
            pcd_path=pcd_path,
            out_path=out_path,
            eps=eps,
            min_points=min_points,
            strict_greater=True,
        )
    # # New pipeline: cluster per original label -> filter by cluster size -> remap -> save
    # pcd_path = r"d:\Desktop\for_cvpr_2026\GS_Param\GAP\point_cloud.pcd"
    # out_path = r"d:\Desktop\for_cvpr_2026\GS_Param\GAP\point_cloud_clustered_filtered.pcd"

    # cluster_then_filter_then_remap_save(
    #     pcd_path=pcd_path,
    #     out_path=out_path,
    #     eps=0.05,            # 调整此值可控制聚类“宽/紧”
    #     min_points=50,      # 聚类阈值（按簇点数）
    #     strict_greater=True, # True: >100；False: >=100
    # )

    # print(out_path)


if __name__ == "__main__":
    main()


