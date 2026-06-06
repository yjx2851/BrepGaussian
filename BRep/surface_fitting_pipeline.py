import trimesh
import numpy as np
import open3d as o3d
import sys
import os
import colorsys
from typing import Optional, List

sys.path.append(os.path.dirname(__file__))
from ransac_surface_fitting import (
    read_pcd_by_label, 
    Plane, 
    CylindricalSurface,
    fit_surface_to_points,
    are_surfaces_similar_with_debug,
    get_plane_cylinder_intersection
)
from json_to_surface_patches import process_surface


def read_pcd_edge_points(pcd_path: str) -> np.ndarray:

    edge_points = []
    
    with open(pcd_path, 'r') as f:
        lines = f.readlines()
        
        field_idx = -1
        for i, line in enumerate(lines):
            if line.startswith('FIELDS'):
                field_idx = i
                break
        
        if field_idx == -1:
            raise ValueError("无法找到FIELDS行")
        
        fields = lines[field_idx].strip().split()[1:]
        
        if 'edge' not in fields:
            print("  警告: PCD文件中没有找到edge字段，返回空数组")
            return np.array([])
        
        edge_idx = fields.index('edge')
        x_idx = fields.index('x')
        y_idx = fields.index('y')
        z_idx = fields.index('z')
        
        data_start_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == 'DATA ascii':
                data_start_idx = i + 1
                break
        
        for line in lines[data_start_idx:]:
            if not line.strip():
                continue
                
            parts = line.strip().split()
            if len(parts) < len(fields):
                continue
            
            try:
                edge_val = float(parts[edge_idx])

                x = float(parts[x_idx])
                y = float(parts[y_idx])
                z = float(parts[z_idx])
                edge_points.append([x, y, z])
            except (ValueError, IndexError):
                continue
    
    return np.array(edge_points) if len(edge_points) > 0 else np.array([])


def adjust_plane_normal_direction(plane_normal, plane_point, points, d_value=None, threshold_distance=0.01):

    if len(points) == 0:
        return plane_normal, d_value if d_value is not None else -np.dot(plane_normal, plane_point), False, {}
    
    if d_value is None:
        d_value = -np.dot(plane_normal, plane_point)
    
    # calculate the signed distance from all points to the plane
    distances = np.dot(points, plane_normal) + d_value
    
    # count the points in the positive and negative directions of the normal vector
    positive_count = np.sum(distances > threshold_distance)  # 在法向量正方向（外侧）
    negative_count = np.sum(distances < -threshold_distance)  # 在法向量负方向（内侧）
    near_zero_count = len(distances) - positive_count - negative_count
    
    min_ratio = 1.2  # the inner point cloud should be at least 1.2 times more than the outer point cloud
    min_diff = max(10, len(points) * 0.1)  # the inner point cloud should be at least 10 points more than the outer point cloud, or 10% of the total number of points
    
    is_correct = (negative_count > positive_count * min_ratio) or (negative_count > positive_count + min_diff)
    
    debug_info = {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'near_zero_count': near_zero_count,
        'total_points': len(points),
        'ratio': negative_count / (positive_count + 1e-10)
    }
    
    if not is_correct:
        # normal direction is wrong, need to flip to the outer side
        adjusted_normal = -plane_normal
        adjusted_d = -d_value
        flipped = True
    else:
        # normal direction is correct
        adjusted_normal = plane_normal
        adjusted_d = d_value
        flipped = False
    
    return adjusted_normal, adjusted_d, flipped, debug_info


def adjust_cylinder_radial_direction(cylinder_center, cylinder_axis, radius, points, threshold_distance=0.01):
    """
    adjust the radial direction of the cylindrical surface, ensure the normal direction points to the outer side
    """
    if len(points) == 0:
        return False, {}
    
    # ensure the axis direction vector is normalized
    axis_normalized = cylinder_axis / (np.linalg.norm(cylinder_axis) + 1e-10)
    
    # calculate the radial distance from all points to the axis
    vecs_to_points = points - cylinder_center
    vecs_projected = vecs_to_points - np.outer(np.dot(vecs_to_points, axis_normalized), axis_normalized)
    radial_distances = np.linalg.norm(vecs_projected, axis=1)
    
    # count the inner and outer points
    inner_count = np.sum(radial_distances < (radius - threshold_distance))
    outer_count = np.sum(radial_distances > (radius + threshold_distance))
    near_radius_count = len(radial_distances) - inner_count - outer_count
    
    min_ratio = 1.2
    min_diff = max(10, len(points) * 0.1)
    
    is_correct = (inner_count > outer_count * min_ratio) or (inner_count > outer_count + min_diff)
    
    debug_info = {
        'inner_count': inner_count,
        'outer_count': outer_count,
        'near_radius_count': near_radius_count,
        'ratio': outer_count / (inner_count + 1e-10),
        'avg_radial_distance': np.mean(radial_distances),
        'radius': radius
    }
    
    needs_flip = not is_correct
    
    return needs_flip, debug_info


def fit_surfaces_by_labels(label_points: dict, cylinder_threshold: float = 0.01, 
                           plane_threshold: float = 0.01, min_points: int = 50):
    """
    fit the plane/cylindrical surface by the labels
    """
    results = []
    for label, pts in label_points.items():
        if label == -1 or len(pts) < min_points:
            continue
        try:
            res = fit_surface_to_points(
                pts, 
                plane_threshold=plane_threshold, 
                cylinder_threshold=cylinder_threshold, 
                min_points=min_points
            )
            if res is not None:
                res['label'] = label
                results.append(res)
                print(f"  标签 {label}: 拟合成功，类型={res.get('type', 'unknown')}, "
                      f"内点数={res.get('inlier_count', 0)}")
        except Exception as e:
            print(f"  标签 {label} 拟合失败: {e}")
    
    return results


def fit_planes_from_unlabeled(unlabeled_points: np.ndarray, plane_threshold: float = 0.01, 
                              min_points: int = 200, max_iterations: int = 5):
    """
    1.2 对剩下的label为-1的点进行RANSAC聚合拟合平面
    
    Args:
        unlabeled_points: 未标注的点云 [N, 3]
        plane_threshold: 平面拟合阈值
        min_points: 最少点数
        max_iterations: 最多拟合几个平面
        
    Returns:
        list[Dict]: 拟合的平面结果列表
    """
    if unlabeled_points is None or len(unlabeled_points) < min_points:
        return []
    
    results = []
    remaining_points = unlabeled_points.copy()
    
    for iteration in range(max_iterations):
        if len(remaining_points) < min_points:
            break
        
        try:
            res = fit_surface_to_points(
                remaining_points,
                plane_threshold=plane_threshold,
                cylinder_threshold=0.02,  # 未标注点主要拟合平面
                min_points=min_points
            )
            
            if res is None or res.get('type') != 'plane':
                break
            
            # 提取内点
            surface = res.get('surface', {})
            normal = np.array(surface.get('normal', []))
            point = np.array(surface.get('point', []))
            
            if len(normal) == 3 and len(point) == 3:
                d_value = -np.dot(normal, point)
                distances = np.abs(np.dot(remaining_points, normal) + d_value)
                inlier_mask = distances < plane_threshold
                inlier_points = remaining_points[inlier_mask]
                
                if len(inlier_points) >= min_points:
                    res['label'] = -1
                    results.append(res)
                    print(f"  未标注点平面 {iteration+1}: 内点数={len(inlier_points)}")
                    
                    # 移除内点，继续拟合剩余点
                    remaining_points = remaining_points[~inlier_mask]
                else:
                    break
            else:
                break
                
        except Exception as e:
            print(f"  未标注点平面拟合失败 (迭代 {iteration+1}): {e}")
            break
    
    return results


def dedupe_similar_surfaces(results: list) -> list:
    """
    1.3 根据面的位置和朝向剔除重复面
    
    Args:
        results: 拟合结果列表
        
    Returns:
        list[Dict]: 去重后的结果列表
    """
    if not results:
        return []
    
    kept = []
    for r in results:
        is_dup = False
        for k in kept:
            similar, reason = are_surfaces_similar_with_debug(r, k)
            if similar:
                print(f"  剔除重复面: {r.get('label', 'unknown')} 与 {k.get('label', 'unknown')} 相似 ({reason})")
                is_dup = True
                break
        if not is_dup:
            kept.append(r)
    
    return kept


def determine_surface_orientations(results: list, label_points: dict):
    """
    2. 根据点云分布判断面朝向，面法向指向外侧
    
    判断面两侧点数，距离面越近权重越高，可能需要初步框定面的范围
    柱面根据点距圆柱轴的距离划分内外
    
    Args:
        results: 拟合结果列表
        label_points: 按标签分组的点云字典
        
    Returns:
        list[Dict]: 调整朝向后的结果列表
    """
    if not results:
        return results
    
    # 收集所有点云
    all_pts = None
    if label_points:
        try:
            all_pts_list = []
            for pts in label_points.values():
                all_pts_list.append(pts)
            if len(all_pts_list) > 0:
                all_pts = np.vstack(all_pts_list)
        except Exception:
            all_pts = None
    
    if all_pts is None or len(all_pts) == 0:
        print("  警告: 没有点云用于判断朝向，跳过朝向调整")
        return results
    
    print(f"  使用 {len(all_pts)} 个点判断面朝向...")
    
    for i, r in enumerate(results):
        s = r.get('surface', {})
        st = s.get('type', r.get('type', ''))
        
        if st == 'plane':
            # 平面：调整法向量方向
            normal = np.array(s.get('normal', []), dtype=float)
            point = np.array(s.get('point', []), dtype=float)
            
            if len(normal) == 3 and len(point) == 3:
                d_value = -np.dot(normal, point)
                distances = np.abs(np.dot(all_pts, normal) + d_value)
                # 选择距离平面较近的点（阈值内的点）
                threshold = 0.02
                plane_points = all_pts[distances < threshold]
                
                if len(plane_points) > 0:
                    adj_n, adj_d, flipped, debug_info = adjust_plane_normal_direction(
                        normal, point, plane_points, d_value=d_value
                    )
                    if flipped:
                        s['normal'] = adj_n.tolist()
                        print(f"    面 {i+1} (平面): 法向量已翻转 (内侧点={debug_info['negative_count']}, "
                              f"外侧点={debug_info['positive_count']}, 比例={debug_info['ratio']:.2f})")
        
        elif st == 'cylindrical_surface':
            # 柱面：根据点距圆柱轴的距离划分内外
            top = np.array(s.get('top_center', []), dtype=float)
            bot = np.array(s.get('bottom_center', []), dtype=float)
            radius = float(s.get('radius', 0))
            
            if len(top) == 3 and len(bot) == 3 and radius > 0:
                axis = top - bot
                if np.linalg.norm(axis) > 1e-6:
                    axis_n = axis / (np.linalg.norm(axis) + 1e-10)
                    center = (top + bot) / 2
                    
                    # 计算点到轴线的径向距离
                    vecs = all_pts - center
                    vecs_proj = vecs - np.outer(np.dot(vecs, axis_n), axis_n)
                    rads = np.linalg.norm(vecs_proj, axis=1)
                    
                    # 选择距离柱面较近的点
                    threshold = 0.02
                    cyl_points = all_pts[np.abs(rads - radius) < threshold]
                    
                    if len(cyl_points) > 0:
                        needs_flip, debug_info = adjust_cylinder_radial_direction(
                            center, axis_n, radius, cyl_points
                        )
                        s['needs_flip_radial'] = bool(needs_flip)
                        print(f"    面 {i+1} (柱面): 内侧点={debug_info['inner_count']}, "
                              f"外侧点={debug_info['outer_count']}, 需要翻转={needs_flip}")
    
    return results


def plane_plane_intersection_line(plane1: dict, plane2: dict):
    """
    3.1 计算两个平面的交线（直线）
    
    Args:
        plane1: 第一个平面结果字典
        plane2: 第二个平面结果字典
        
    Returns:
        dict: 交线信息 {'type': 'line', 'point': [3], 'direction': [3], 'planes': (plane1, plane2)}
              如果平面平行则返回 None
    """
    s1 = plane1.get('surface', {})
    s2 = plane2.get('surface', {})
    
    n1 = np.array(s1.get('normal', []), dtype=float)
    p1 = np.array(s1.get('point', []), dtype=float)
    n2 = np.array(s2.get('normal', []), dtype=float)
    p2 = np.array(s2.get('point', []), dtype=float)
    
    if len(n1) != 3 or len(n2) != 3 or len(p1) != 3 or len(p2) != 3:
        return None
    
    # 计算交线方向（两个法向量的叉积）
    d = np.cross(n1, n2)
    if np.linalg.norm(d) < 1e-8:
        # 平面平行或重合
        return None
    
    d = d / (np.linalg.norm(d) + 1e-10)
    
    # 求解直线上的一个点
    # 平面方程：n1·x = n1·p1, n2·x = n2·p2
    # 使用最小二乘法求解
    A = np.vstack([n1, n2, d])
    b = np.array([np.dot(n1, p1), np.dot(n2, p2), 0.0])
    
    try:
        x0, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        # 如果求解失败，使用两个平面点的中点作为近似
        x0 = (p1 + p2) / 2
    
    return {
        'type': 'line',
        'point': x0,
        'direction': d,
        'planes': (plane1, plane2)
    }


def compute_intersections(results: list):
    """
    3.1 处理面面交线
    
    计算平面交直线、平面与柱面交二次曲线
    
    Args:
        results: 拟合结果列表
        
    Returns:
        list[dict]: 交线/交曲线列表
    """
    intersections = []
    
    # 分离平面和柱面
    planes = [(idx, r) for idx, r in enumerate(results) if r.get('surface', {}).get('type', r.get('type', '')) == 'plane']
    cylinders = [r for r in results if r.get('surface', {}).get('type', r.get('type', '')) == 'cylindrical_surface']
    
    print(f"  计算交点: {len(planes)} 个平面, {len(cylinders)} 个柱面")
    
    # 平面-平面交线
    for ii in range(len(planes)):
        for jj in range(ii + 1, len(planes)):
            pi_idx, pi = planes[ii]
            pj_idx, pj = planes[jj]
            inter = plane_plane_intersection_line(pi, pj)
            if inter is not None:
                inter['plane_indices'] = (int(pi_idx), int(pj_idx))
                intersections.append(inter)
    
    # 平面-柱面交线（圆）
    # 需要找到柱面在 results 中的索引
    cylinder_indices = {}
    for idx, r in enumerate(results):
        if r.get('surface', {}).get('type', r.get('type', '')) == 'cylindrical_surface':
            # 使用 id 或 label 来匹配柱面
            cylinder_key = (r.get('label'), id(r))
            cylinder_indices[cylinder_key] = idx
    
    for pl_idx, pl in planes:
        for cy in cylinders:
            # 找到柱面在 results 中的索引
            cy_key = (cy.get('label'), id(cy))
            cy_idx = cylinder_indices.get(cy_key, None)
            if cy_idx is None:
                # 如果找不到，尝试通过遍历 results 找到匹配的索引
                for idx, r in enumerate(results):
                    if r is cy or (r.get('label') == cy.get('label') and 
                                   r.get('surface', {}).get('type', r.get('type', '')) == 'cylindrical_surface'):
                        cy_idx = idx
                        break
            
            inter_info = get_plane_cylinder_intersection(pl, cy)
            if inter_info is not None:
                intersections.append({
                    'type': 'circle',
                    'center': inter_info['center'],
                    'normal': inter_info['normal'],
                    'radius': inter_info['radius'],
                    'plane': pl,
                    'cylinder': cy,
                    'surface_indices': (int(pl_idx), int(cy_idx)) if cy_idx is not None else None
                })
    
    return intersections


def segment_intersections_with_edge(intersections: list, edge_points: np.ndarray, 
                                   distance_thresh: float = 0.02):
    """
    3.2 根据附近edge点云（edge标签为1的点）位置判断每条交线是否合法，并确定范围
    
    方案一：先处理所有线线交点，分段判断每段线是否应该存在，如果线段附近有edge点云，则存在。
    
    Args:
        intersections: 交线列表
        edge_points: 边缘点云 [N, 3]（label=1的点）
        distance_thresh: 距离阈值
        
    Returns:
        list[dict]: 分段后的交线列表，每条线包含范围信息
    """
    if intersections is None or len(intersections) == 0:
        return []
    
    if edge_points is None or len(edge_points) == 0:
        print("  警告: 没有边缘点云，无法裁剪交线范围")
        return intersections
    
    print(f"  使用 {len(edge_points)} 个边缘点裁剪交线范围...")
    
    segmented = []
    
    for inter in intersections:
        if inter['type'] == 'line':
            # 直线：将边缘点投影到直线，确定线段范围
            p0 = inter['point']
            d = inter['direction'] / (np.linalg.norm(inter['direction']) + 1e-10)
            
            vecs = edge_points - p0
            # 投影参数 t
            t = np.dot(vecs, d)
            # 垂直距离
            perp = vecs - np.outer(t, d)
            perp_distances = np.linalg.norm(perp, axis=1)
            
            # 只保留距离直线小于阈值的点
            mask = perp_distances < distance_thresh
            
            if np.any(mask):
                t_selected = t[mask]
                t_min = float(np.min(t_selected))
                t_max = float(np.max(t_selected))
                
                # 计算线段端点
                p_start = p0 + t_min * d
                p_end = p0 + t_max * d
                
                seg_item = {
                    **inter,
                    't_min': t_min,
                    't_max': t_max,
                    'start_point': p_start,
                    'end_point': p_end,
                    'edge_point_count': int(np.sum(mask))
                }
                if 'plane_indices' in inter:
                    seg_item['plane_indices'] = inter['plane_indices']
                segmented.append(seg_item)
                print(f"    线段: t范围=[{t_min:.4f}, {t_max:.4f}], 边缘点数={np.sum(mask)}")
            else:
                print(f"    线段: 附近无边缘点，跳过")
        
        elif inter['type'] == 'circle':
            # 圆：检查边缘点是否在圆附近
            center = inter['center']
            radius = inter['radius']
            normal = inter['normal'] / (np.linalg.norm(inter['normal']) + 1e-10)
            
            # 将点投影到圆所在平面
            vecs = edge_points - center
            # 点到平面的距离
            dist_to_plane = np.abs(np.dot(vecs, normal))
            # 在平面内的径向距离
            vecs_in_plane = vecs - np.outer(np.dot(vecs, normal), normal)
            rad_distances = np.abs(np.linalg.norm(vecs_in_plane, axis=1) - radius)
            
            # 检查是否有边缘点在圆附近
            mask = (dist_to_plane < distance_thresh) & (rad_distances < distance_thresh)
            
            if np.any(mask):
                segmented.append({
                    **inter,
                    'edge_point_count': int(np.sum(mask))
                })
                print(f"    圆: 半径={radius:.4f}, 边缘点数={np.sum(mask)}")
            else:
                print(f"    圆: 附近无边缘点，跳过")
    
    return segmented


def _closest_points_between_lines(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray):
    """
    计算两条三维直线的最近点（无限长直线），返回各自参数 t0, t1 以及对应点 q0, q1。
    直线形式：L0(t)=p0+t*d0, L1(s)=p1+s*d1
    """
    d0 = d0 / (np.linalg.norm(d0) + 1e-12)
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    w0 = p0 - p1
    a = np.dot(d0, d0)
    b = np.dot(d0, d1)
    c = np.dot(d1, d1)
    d = np.dot(d0, w0)
    e = np.dot(d1, w0)
    denom = a*c - b*b
    if abs(denom) < 1e-12:
        # 平行或近似平行，取任意投影
        t0 = -d / (a + 1e-12)
        t1 = 0.0
    else:
        t0 = (b*e - c*d) / denom
        t1 = (a*e - b*d) / denom
    q0 = p0 + t0 * d0
    q1 = p1 + t1 * d1
    return t0, q0, t1, q1


def snap_line_endpoints_by_line_intersections(segmented: list,
                                             dist_tol: float = 0.01,
                                             endpoint_tol: float = 0.02):
    """
    通过线-线几何相交/最短连线，得到准确的公共端点，并将相近端点吸附到一致坐标。
    仅处理 type=='line' 的对象：
      - 对每对线，求最近点 q0, q1；若 |q0-q1| < dist_tol 且 t0,t1 处于各自线段范围内（含 endpoint_tol 余量），
        则认为两线共享一个节点，将该节点记录下来。
      - 聚类所有节点（半径 dist_tol），用簇中心作为统一端点坐标。
      - 将每条线的 start/end 若在某个节点半径内，则吸附到该节点。
    """
    # 收集线条
    lines = [l for l in segmented if l.get('type') == 'line']
    if len(lines) < 2:
        return segmented

    # 预计算参数
    line_params = []
    for l in lines:
        p0 = np.asarray(l['point']) if 'point' in l else np.asarray(l['start_point'])
        d = np.asarray(l['direction'])
        d = d / (np.linalg.norm(d) + 1e-12)
        tmin = l.get('t_min', None)
        tmax = l.get('t_max', None)
        if tmin is None or tmax is None:
            # 从端点反推 t 范围
            sp = np.asarray(l['start_point'])
            ep = np.asarray(l['end_point'])
            # 以 p0 为参考，近似求 t
            tmin = np.dot(sp - p0, d)
            tmax = np.dot(ep - p0, d)
            if tmin > tmax:
                tmin, tmax = tmax, tmin
        line_params.append((p0, d, float(tmin), float(tmax)))

    # 计算候选节点（两线相交）
    candidate_nodes = []
    for i in range(len(lines)):
        p0, d0, t0min, t0max = line_params[i]
        for j in range(i+1, len(lines)):
            p1, d1, t1min, t1max = line_params[j]
            t0, q0, t1, q1 = _closest_points_between_lines(p0, d0, p1, d1)
            if np.linalg.norm(q0 - q1) <= dist_tol:
                # 参数是否在各自范围内（加余量）
                if (t0min - endpoint_tol) <= t0 <= (t0max + endpoint_tol) and \
                   (t1min - endpoint_tol) <= t1 <= (t1max + endpoint_tol):
                    node = 0.5 * (q0 + q1)
                    candidate_nodes.append(node)

    if len(candidate_nodes) == 0:
        return segmented

    candidate_nodes = np.asarray(candidate_nodes)

    # 简易聚类：逐个合并到已有簇（半径 dist_tol）
    clusters = []
    for node in candidate_nodes:
        matched = False
        for c in clusters:
            if np.linalg.norm(node - c['center']) <= dist_tol:
                c['points'].append(node)
                c['center'] = np.mean(c['points'], axis=0)
                matched = True
                break
        if not matched:
            clusters.append({'points': [node], 'center': node.copy()})

    snap_centers = [c['center'] for c in clusters]

    # 将每条线段端点吸附到最近的节点（若在 dist_tol 内）
    def snap_point(pt: np.ndarray) -> np.ndarray:
        best = pt
        best_dist = float('inf')
        for c in snap_centers:
            dist = np.linalg.norm(pt - c)
            if dist < best_dist:
                best_dist = dist
                best = c
        return best if best_dist <= dist_tol else pt

    for l in lines:
        sp = np.asarray(l['start_point'])
        ep = np.asarray(l['end_point'])
        sp_new = snap_point(sp)
        ep_new = snap_point(ep)
        l['start_point'] = sp_new
        l['end_point'] = ep_new

    return segmented


def create_line_cylinder(start_point: np.ndarray, end_point: np.ndarray, 
                        radius: float = 0.005, resolution: int = 8):
    """
    创建细圆柱用于可视化线段
    
    Args:
        start_point: 起点 [3]
        end_point: 终点 [3]
        radius: 圆柱半径
        resolution: 圆周分辨率
        
    Returns:
        o3d.geometry.TriangleMesh: 圆柱网格
    """
    direction = end_point - start_point
    height = np.linalg.norm(direction)
    
    if height < 1e-6:
        return None
    
    # 创建圆柱（沿Z轴）
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=height,
        resolution=resolution
    )
    
    # 计算旋转矩阵，将Z轴旋转到direction方向
    z_axis = np.array([0, 0, 1])
    direction_normalized = direction / height
    
    if not np.allclose(direction_normalized, z_axis):
        # 计算旋转轴和角度
        rotation_axis = np.cross(z_axis, direction_normalized)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1, 1))
            
            # 创建旋转矩阵
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
                rotation_axis * rotation_angle
            )
            cylinder.rotate(rotation_matrix, center=[0, 0, 0])
    
    # 平移到线段中点（Open3D 圆柱默认中心在原点）
    center = (start_point + end_point) / 2
    cylinder.translate(center)
    
    return cylinder


def create_sphere(center: np.ndarray, radius: float = 0.01, resolution: int = 20):
    """
    创建小球体用于可视化端点
    
    Args:
        center: 球心 [3]
        radius: 球半径
        resolution: 球的分辨率
        
    Returns:
        o3d.geometry.TriangleMesh: 球体网格
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius,
        resolution=resolution
    )
    sphere.translate(center)
    return sphere


def visualize_intersections(intersections: list, label_points: dict = None, 
                           edge_points: np.ndarray = None,
                           line_radius: float = 0.005, sphere_radius: float = 0.01):
    """
    可视化交线：线段用细圆柱，端点用小球体
    
    Args:
        intersections: 交线列表
        label_points: 点云字典（可选，用于显示点云）
        edge_points: 边缘点数组 [N, 3]（edge=1的点）
        line_radius: 线段圆柱半径
        sphere_radius: 端点球体半径
    """
    geometries = []
    
    # 添加点云（如果提供）
    if label_points is not None:
        for label, points in label_points.items():
            if points is not None and len(points) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                
                # 根据标签设置不同颜色
                if label == -1:  # 未标注点
                    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
                else:
                    # 其他标签用不同颜色
                    color = np.array([0.0, 0.8, 1.0])  # 青色
                    pcd.paint_uniform_color(color)
                
                geometries.append(pcd)
    
    # 单独添加边缘点（edge=1）
    if edge_points is not None and len(edge_points) > 0:
        edge_pcd = o3d.geometry.PointCloud()
        edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
        edge_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        geometries.append(edge_pcd)
    
    # 可视化交线
    line_count = 0
    circle_count = 0
    
    for inter in intersections:
        if inter.get('type') == 'line':
            # 线段：用细圆柱可视化
            start_point = inter.get('start_point')
            end_point = inter.get('end_point')
            
            if start_point is not None and end_point is not None:
                start_point = np.array(start_point)
                end_point = np.array(end_point)
                
                # 创建圆柱
                cylinder = create_line_cylinder(start_point, end_point, radius=line_radius)
                if cylinder is not None:
                    cylinder.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
                    geometries.append(cylinder)
                
                # 创建端点球体
                sphere_start = create_sphere(start_point, radius=sphere_radius)
                sphere_start.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色
                geometries.append(sphere_start)
                
                sphere_end = create_sphere(end_point, radius=sphere_radius)
                sphere_end.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色
                geometries.append(sphere_end)
                
                line_count += 1
        
        elif inter.get('type') == 'circle':
            # 圆：用多个小圆柱段近似可视化
            center = np.array(inter.get('center'))
            radius = float(inter.get('radius'))
            normal = np.array(inter.get('normal'))
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            
            # 构建圆所在平面的坐标系
            if abs(normal[2]) < 0.9:
                u = np.array([0, 0, 1])
            else:
                u = np.array([1, 0, 0])
            u = u - np.dot(u, normal) * normal
            u = u / (np.linalg.norm(u) + 1e-10)
            v = np.cross(normal, u)
            v = v / (np.linalg.norm(v) + 1e-10)
            
            # 用多个线段近似圆
            num_segments = 32
            for i in range(num_segments):
                angle1 = i * 2.0 * np.pi / num_segments
                angle2 = (i + 1) * 2.0 * np.pi / num_segments
                
                p1 = center + radius * (np.cos(angle1) * u + np.sin(angle1) * v)
                p2 = center + radius * (np.cos(angle2) * u + np.sin(angle2) * v)
                
                cylinder = create_line_cylinder(p1, p2, radius=line_radius)
                if cylinder is not None:
                    cylinder.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
                    geometries.append(cylinder)
            
            # 圆心用球体标记
            sphere_center = create_sphere(center, radius=sphere_radius)
            sphere_center.paint_uniform_color([1.0, 0.0, 1.0])  # 洋红色
            geometries.append(sphere_center)
            
            circle_count += 1
    
    if len(geometries) == 0:
        print("  警告: 没有可可视化的几何体")
        return
    
    print(f"\n  可视化: {line_count} 条线段, {circle_count} 个圆")
    print("  颜色说明:")
    print("    - 绿色细圆柱: 平面交线（线段）")
    print("    - 黄色球体: 线段端点")
    print("    - 蓝色细圆柱: 平面-柱面交线（圆）")
    print("    - 洋红色球体: 圆心")
    print("    - 红色点云: 边缘点 (edge=1)")
    print("    - 灰色点云: 未标注点 (label=-1)")
    print("    - 青色点云: 其他标签点")
    print("\n  按 'Q' 或关闭窗口继续...")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="交线可视化",
        width=1024,
        height=768,
        point_show_normal=False
    )


def visualize_final_geometry(corners: list, lines: list, curves: list = None,
                            line_radius: float = 0.005, sphere_radius: float = 0.01):
    """
    可视化最终结果：
      - 角点表 corners（坐标）用小球体
      - 线段表 lines（端点索引）用细圆柱连接对应角点
    """
    geometries = []
    corners_np = [np.asarray(c, dtype=float) for c in corners]
    
    # 画角点
    for c in corners_np:
        sp = create_sphere(c, radius=sphere_radius)
        sp.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色
        geometries.append(sp)
    
    # 画线段
    for line in lines:
        i0 = int(line.get('start', -1))
        i1 = int(line.get('end', -1))
        if 0 <= i0 < len(corners_np) and 0 <= i1 < len(corners_np) and i0 != i1:
            p0 = corners_np[i0]
            p1 = corners_np[i1]
            cyl = create_line_cylinder(p0, p1, radius=line_radius)
            if cyl is not None:
                cyl.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
                geometries.append(cyl)

    # 画曲线（圆/圆弧）
    curve_count = 0
    if curves:
        for c in curves:
            center = np.asarray(c.get('center', []), dtype=float)
            normal = np.asarray(c.get('normal', []), dtype=float)
            radius = float(c.get('radius', 0.0))
            if center.shape == (3,) and normal.shape == (3,) and radius > 0:
                n = normal / (np.linalg.norm(normal) + 1e-12)
                # 构建圆平面基
                u = np.array([0, 0, 1], dtype=float) if abs(n[2]) < 0.9 else np.array([1, 0, 0], dtype=float)
                u = u - np.dot(u, n) * n
                u = u / (np.linalg.norm(u) + 1e-12)
                v = np.cross(n, u)
                v = v / (np.linalg.norm(v) + 1e-12)
                # 近似圆
                num_segments = 48
                for i in range(num_segments):
                    a1 = i * 2.0 * np.pi / num_segments
                    a2 = (i + 1) * 2.0 * np.pi / num_segments
                    p1 = center + radius * (np.cos(a1) * u + np.sin(a1) * v)
                    p2 = center + radius * (np.cos(a2) * u + np.sin(a2) * v)
                    cyl = create_line_cylinder(p1, p2, radius=line_radius)
                    if cyl is not None:
                        cyl.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
                        geometries.append(cyl)
                curve_count += 1
    
    if len(geometries) == 0:
        print("  警告: 最终结果没有可可视化的几何体")
        return
    
    print(f"\n  最终几何可视化: 角点={len(corners)}, 线段={len(lines)}, 曲线={(len(curves) if curves else 0)}")
    print("  颜色说明:")
    print("    - 绿色细圆柱: 最终线段（按角点索引连接）")
    print("    - 黄色球体: 最终角点")
    print("\n  按 'Q' 或关闭窗口继续...")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="最终结果可视化",
        width=1024,
        height=768,
        point_show_normal=False
    )


def get_face_light_color(face_idx: int) -> np.ndarray:
    """为每个面片生成互不相同的浅色（低饱和度、高明度）。"""
    hue = (face_idx * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.30, 0.96)
    return np.array([r, g, b], dtype=float)


def build_surface_meshes(surfaces: list, lines: list, curves: list,
                         corners: list) -> List[Optional[o3d.geometry.TriangleMesh]]:
    """为每个解析面生成三角网格（平面/柱面）。"""
    meshes: List[Optional[o3d.geometry.TriangleMesh]] = []
    for i, surface in enumerate(surfaces):
        mesh = process_surface(surface, lines, curves, corners, output_dir="", surface_idx=i, verbose=False)
        meshes.append(mesh)
    return meshes


def export_final_obj(corners: list, lines: list, output_dir: str,
                     line_radius: float = 0.005, sphere_radius: float = 0.01,
                     filename: str = "pipeline_results.obj",
                     curves: Optional[list] = None,
                     surface_meshes: Optional[List[Optional[o3d.geometry.TriangleMesh]]] = None):
    """
    导出最终结果为一个 OBJ 模型：
      - 面: 浅色三角面片（各面颜色不同）
      - 线段: 蓝色细圆柱
      - 角点: 红色小球
    OBJ 保存在与 JSON 相同目录。
    """
    corners_np = [np.asarray(c, dtype=float) for c in corners]
    all_vertices = []
    all_triangles = []
    all_colors = []
    vert_offset = 0

    # 面片（各面不同浅色）
    if surface_meshes:
        for idx, mesh in enumerate(surface_meshes):
            if mesh is None:
                continue
            color = get_face_light_color(idx)
            mesh.paint_uniform_color(color)
            v = np.asarray(mesh.vertices)
            f = np.asarray(mesh.triangles)
            c = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.tile(color.reshape(1, 3), (len(v), 1))
            all_vertices.append(v)
            all_triangles.append(f + vert_offset)
            all_colors.append(c)
            vert_offset += len(v)

    # 线段（蓝色）
    for line in lines:
        i0 = int(line.get('start', -1))
        i1 = int(line.get('end', -1))
        if 0 <= i0 < len(corners_np) and 0 <= i1 < len(corners_np) and i0 != i1:
            p0 = corners_np[i0]
            p1 = corners_np[i1]
            cyl = create_line_cylinder(p0, p1, radius=line_radius)
            if cyl is None:
                continue
            cyl.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
            v = np.asarray(cyl.vertices)
            f = np.asarray(cyl.triangles)
            c = np.asarray(cyl.vertex_colors) if cyl.has_vertex_colors() else np.tile(np.array([[0.0, 0.0, 1.0]]), (len(v), 1))
            all_vertices.append(v)
            all_triangles.append(f + vert_offset)
            all_colors.append(c)
            vert_offset += len(v)

    # 角点（红色）
    for cpt in corners_np:
        sp = create_sphere(cpt, radius=sphere_radius)
        sp.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        v = np.asarray(sp.vertices)
        f = np.asarray(sp.triangles)
        c = np.asarray(sp.vertex_colors) if sp.has_vertex_colors() else np.tile(np.array([[1.0, 0.0, 0.0]]), (len(v), 1))
        all_vertices.append(v)
        all_triangles.append(f + vert_offset)
        all_colors.append(c)
        vert_offset += len(v)

    # 曲线（圆/圆弧）- 用与直线相同粗细的圆柱片段近似（蓝色）
    if curves:
        for c in curves:
            center = np.asarray(c.get('center', []), dtype=float)
            normal = np.asarray(c.get('normal', []), dtype=float)
            radius = float(c.get('radius', 0.0))
            if center.shape == (3,) and normal.shape == (3,) and radius > 0:
                n = normal / (np.linalg.norm(normal) + 1e-12)
                u = np.array([0, 0, 1], dtype=float) if abs(n[2]) < 0.9 else np.array([1, 0, 0], dtype=float)
                u = u - np.dot(u, n) * n
                u = u / (np.linalg.norm(u) + 1e-12)
                v = np.cross(n, u)
                v = v / (np.linalg.norm(v) + 1e-12)
                num_segments = 48
                for i in range(num_segments):
                    a1 = i * 2.0 * np.pi / num_segments
                    a2 = (i + 1) * 2.0 * np.pi / num_segments
                    p1 = center + radius * (np.cos(a1) * u + np.sin(a1) * v)
                    p2 = center + radius * (np.cos(a2) * u + np.sin(a2) * v)
                    cyl = create_line_cylinder(p1, p2, radius=line_radius)
                    if cyl is None:
                        continue
                    cyl.paint_uniform_color([0.0, 0.0, 1.0])
                    v_m = np.asarray(cyl.vertices)
                    f_m = np.asarray(cyl.triangles)
                    c_m = np.asarray(cyl.vertex_colors) if cyl.has_vertex_colors() else np.tile(np.array([[0.0, 0.0, 1.0]]), (len(v_m), 1))
                    all_vertices.append(v_m)
                    all_triangles.append(f_m + vert_offset)
                    all_colors.append(c_m)
                    vert_offset += len(v_m)

    if len(all_vertices) == 0:
        print("  警告: 无法导出 OBJ（没有可用的几何体）")
        return

    V = np.vstack(all_vertices)
    F = np.vstack(all_triangles)
    C = np.vstack(all_colors) if len(all_colors) > 0 else None

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))
    mesh.compute_vertex_normals()
    if C is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(C)
    # OBJ 路径
    os.makedirs(output_dir, exist_ok=True)
    obj_path = os.path.join(output_dir, filename)
    # 清除三角形法线（OBJ不需要三角面法线）
    try:
        mesh.triangle_normals = o3d.utility.Vector3dVector()
    except Exception:
        pass
    o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_normals=False)


def export_parts_objs(corners: list, lines: list, output_dir: str,
                      line_radius: float = 0.005, sphere_radius: float = 0.01,
                      curves: Optional[list] = None,
                      surface_meshes: Optional[List[Optional[o3d.geometry.TriangleMesh]]] = None,
                      subdir: str = "parts"):
    """
    导出分部件 OBJ：
      - parts/face_XXXXX.obj：每个面单独保存
      - parts/lines.obj：蓝色细圆柱（包含直线与曲线）
      - parts/points.obj：红色小球
    """
    corners_np = [np.asarray(c, dtype=float) for c in corners]
    parts_dir = os.path.join(output_dir, subdir)
    os.makedirs(parts_dir, exist_ok=True)

    # 每个面单独导出
    if surface_meshes:
        face_count = 0
        for idx, mesh in enumerate(surface_meshes):
            if mesh is None:
                continue
            color = get_face_light_color(idx)
            mesh.paint_uniform_color(color)
            face_path = os.path.join(parts_dir, f"face_{idx:05d}.obj")
            try:
                mesh_copy = o3d.geometry.TriangleMesh(mesh)
                mesh_copy.triangle_normals = o3d.utility.Vector3dVector()
            except Exception:
                mesh_copy = mesh
            o3d.io.write_triangle_mesh(face_path, mesh_copy, write_vertex_normals=False)
            face_count += 1

    # 构建 lines mesh
    all_v_l, all_f_l, all_c_l = [], [], []
    voff = 0
    # 直线
    for line in lines:
        i0 = int(line.get('start', -1))
        i1 = int(line.get('end', -1))
        if 0 <= i0 < len(corners_np) and 0 <= i1 < len(corners_np) and i0 != i1:
            p0 = corners_np[i0]
            p1 = corners_np[i1]
            cyl = create_line_cylinder(p0, p1, radius=line_radius)
            if cyl is None:
                continue
            cyl.paint_uniform_color([0.0, 0.0, 1.0])
            v = np.asarray(cyl.vertices); f = np.asarray(cyl.triangles)
            c = np.asarray(cyl.vertex_colors) if cyl.has_vertex_colors() else np.tile(np.array([[0.0,0.0,1.0]]),(len(v),1))
            all_v_l.append(v); all_f_l.append(f + voff); all_c_l.append(c); voff += len(v)
    # 曲线
    if curves:
        for c in curves:
            center = np.asarray(c.get('center', []), dtype=float)
            normal = np.asarray(c.get('normal', []), dtype=float)
            radius = float(c.get('radius', 0.0))
            if center.shape==(3,) and normal.shape==(3,) and radius>0:
                n = normal / (np.linalg.norm(normal) + 1e-12)
                u = np.array([0,0,1], dtype=float) if abs(n[2])<0.9 else np.array([1,0,0], dtype=float)
                u = u - np.dot(u, n)*n; u = u/(np.linalg.norm(u)+1e-12)
                v = np.cross(n,u); v = v/(np.linalg.norm(v)+1e-12)
                num_segments = 48
                for i in range(num_segments):
                    a1 = i*2.0*np.pi/num_segments; a2 = (i+1)*2.0*np.pi/num_segments
                    p1 = center + radius*(np.cos(a1)*u + np.sin(a1)*v)
                    p2 = center + radius*(np.cos(a2)*u + np.sin(a2)*v)
                    cyl = create_line_cylinder(p1, p2, radius=line_radius)
                    if cyl is None:
                        continue
                    cyl.paint_uniform_color([0.0,0.0,1.0])
                    v_m = np.asarray(cyl.vertices); f_m = np.asarray(cyl.triangles)
                    c_m = np.asarray(cyl.vertex_colors) if cyl.has_vertex_colors() else np.tile(np.array([[0.0,0.0,1.0]]),(len(v_m),1))
                    all_v_l.append(v_m); all_f_l.append(f_m + voff); all_c_l.append(c_m); voff += len(v_m)

    if len(all_v_l) > 0:
        V = np.vstack(all_v_l); F = np.vstack(all_f_l); C = np.vstack(all_c_l)
        mesh_l = o3d.geometry.TriangleMesh(); mesh_l.vertices = o3d.utility.Vector3dVector(V)
        mesh_l.triangles = o3d.utility.Vector3iVector(F.astype(np.int32)); mesh_l.compute_vertex_normals()
        mesh_l.vertex_colors = o3d.utility.Vector3dVector(C)
        try:
            mesh_l.triangle_normals = o3d.utility.Vector3dVector()
        except Exception:
            pass
        o3d.io.write_triangle_mesh(os.path.join(parts_dir, "lines.obj"), mesh_l, write_vertex_normals=False)

    # 构建 points mesh
    all_v_p, all_f_p, all_c_p = [], [], []
    voff = 0
    for cpt in corners_np:
        sp = create_sphere(cpt, radius=sphere_radius)
        sp.paint_uniform_color([1.0, 0.0, 0.0])
        v = np.asarray(sp.vertices); f = np.asarray(sp.triangles)
        c = np.asarray(sp.vertex_colors) if sp.has_vertex_colors() else np.tile(np.array([[1.0,0.0,0.0]]),(len(v),1))
        all_v_p.append(v); all_f_p.append(f + voff); all_c_p.append(c); voff += len(v)

    if len(all_v_p) > 0:
        Vp = np.vstack(all_v_p); Fp = np.vstack(all_f_p); Cp = np.vstack(all_c_p)
        mesh_p = o3d.geometry.TriangleMesh(); mesh_p.vertices = o3d.utility.Vector3dVector(Vp)
        mesh_p.triangles = o3d.utility.Vector3iVector(Fp.astype(np.int32)); mesh_p.compute_vertex_normals()
        mesh_p.vertex_colors = o3d.utility.Vector3dVector(Cp)
        try:
            mesh_p.triangle_normals = o3d.utility.Vector3dVector()
        except Exception:
            pass
        o3d.io.write_triangle_mesh(os.path.join(parts_dir, "points.obj"), mesh_p, write_vertex_normals=False)


def export_structured_obj(corners: list, lines: list, output_dir: str,
                          curves: Optional[list] = None,
                          surface_meshes: Optional[List[Optional[o3d.geometry.TriangleMesh]]] = None,
                          line_radius: float = 0.005, sphere_radius: float = 0.01,
                          filename: str = "pipeline_results_structured.obj",
                          mtl_name: str = "pipeline_results_structured.mtl"):
    """
    导出一个结构化的 OBJ：
      - 单文件，包含多个 object（o name）：每个面、每条线、每个点一个独立 object
      - 使用 MTL 材质区分：面=各面不同浅色，线/曲线=蓝色，点=红色
    便于在 Blender 中按对象编辑。
    """
    corners_np = [np.asarray(c, dtype=float) for c in corners]
    obj_lines = []
    v_offset = 0
    # 写入 MTL 参照
    obj_lines.append(f"mtllib {mtl_name}\n")

    def write_mesh_as_object(name: str, mesh: o3d.geometry.TriangleMesh, usemtl: str):
        nonlocal v_offset
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        obj_lines.append(f"o {name}\n")
        obj_lines.append(f"usemtl {usemtl}\n")
        for p in v:
            obj_lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        # faces are 1-indexed in OBJ
        for tri in f:
            a, b, c = tri + 1 + v_offset
            obj_lines.append(f"f {a} {b} {c}\n")
        v_offset += len(v)

    # 面对象（各面不同浅色材质）
    face_materials = []
    if surface_meshes:
        for idx, mesh in enumerate(surface_meshes):
            if mesh is None:
                continue
            mtl_name_face = f"face_{idx:05d}"
            color = get_face_light_color(idx)
            write_mesh_as_object(f"face_{idx:05d}", mesh, mtl_name_face)
            face_materials.append((mtl_name_face, color))

    # 线段对象
    for idx, line in enumerate(lines):
        i0 = int(line.get('start', -1))
        i1 = int(line.get('end', -1))
        if 0 <= i0 < len(corners_np) and 0 <= i1 < len(corners_np) and i0 != i1:
            p0 = corners_np[i0]; p1 = corners_np[i1]
            cyl = create_line_cylinder(p0, p1, radius=line_radius)
            if cyl is None:
                continue
            write_mesh_as_object(f"line_{idx:05d}", cyl, "line_blue")

    # 曲线对象（圆近似）
    if curves:
        curve_counter = 0
        for c in curves:
            center = np.asarray(c.get('center', []), dtype=float)
            normal = np.asarray(c.get('normal', []), dtype=float)
            radius = float(c.get('radius', 0.0))
            if center.shape==(3,) and normal.shape==(3,) and radius>0:
                n = normal / (np.linalg.norm(normal) + 1e-12)
                u = np.array([0,0,1], dtype=float) if abs(n[2])<0.9 else np.array([1,0,0], dtype=float)
                u = u - np.dot(u, n)*n; u = u/(np.linalg.norm(u)+1e-12)
                v = np.cross(n,u); v = v/(np.linalg.norm(v)+1e-12)
                num_segments = 48
                for i in range(num_segments):
                    a1 = i*2.0*np.pi/num_segments; a2 = (i+1)*2.0*np.pi/num_segments
                    p1 = center + radius*(np.cos(a1)*u + np.sin(a1)*v)
                    p2 = center + radius*(np.cos(a2)*u + np.sin(a2)*v)
                    cyl = create_line_cylinder(p1, p2, radius=line_radius)
                    if cyl is None:
                        continue
                    write_mesh_as_object(f"curve_{curve_counter:05d}_{i:03d}", cyl, "line_blue")
                curve_counter += 1

    # 点对象
    for idx, cpt in enumerate(corners_np):
        sp = create_sphere(cpt, radius=sphere_radius)
        write_mesh_as_object(f"point_{idx:05d}", sp, "point_red")

    # 输出 OBJ
    os.makedirs(output_dir, exist_ok=True)
    obj_path = os.path.join(output_dir, filename)
    with open(obj_path, "w", encoding="utf-8") as f:
        f.writelines(obj_lines)

    # 输出 MTL
    mtl_path = os.path.join(output_dir, mtl_name)
    mtl_lines = []
    for mtl_name_face, color in face_materials:
        mtl_lines.extend([
            f"newmtl {mtl_name_face}\n",
            f"Kd {color[0]:.4f} {color[1]:.4f} {color[2]:.4f}\n",
            "Ka 0.0 0.0 0.0\n",
            "Ks 0.0 0.0 0.0\n",
            "d 1.0\n",
            "illum 1\n\n",
        ])
    mtl_lines.extend([
        "newmtl line_blue\n",
        "Kd 0.0 0.0 1.0\n",
        "Ka 0.0 0.0 0.0\n",
        "Ks 0.0 0.0 0.0\n",
        "d 1.0\n",
        "illum 1\n\n",
        "newmtl point_red\n",
        "Kd 1.0 0.0 0.0\n",
        "Ka 0.0 0.0 0.0\n",
        "Ks 0.0 0.0 0.0\n",
        "d 1.0\n",
        "illum 1\n"
    ])
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.writelines(mtl_lines)


def debug_check_final_alignment(segmented: list, corners: list, lines: list):
    """
    诊断：检查每条最终线段端点与对应角点的偏差，帮助发现索引/吸附问题。
    """
    corners_np = [np.asarray(c, dtype=float) for c in corners]
    total = 0
    bad = 0
    max_dev = 0.0
    devs = []
    # 构建快速映射：用端点坐标到最近角点的距离（仅用于诊断输出）
    for line in lines:
        i0 = int(line.get('start', -1))
        i1 = int(line.get('end', -1))
        if not (0 <= i0 < len(corners_np) and 0 <= i1 < len(corners_np)):
            continue
        total += 1
        corner_s = corners_np[i0]
        corner_e = corners_np[i1]
        # 在 segmented 中找到最接近该线的原端点（诊断，不严格一一对应）
        # 取与 corner_s 最近的原端点距离
        best_s = 1e9
        best_e = 1e9
        for inter in segmented:
            if inter.get('type') != 'line':
                continue
            if 'start_point' not in inter or 'end_point' not in inter:
                continue
            sp = np.asarray(inter['start_point']); ep = np.asarray(inter['end_point'])
            best_s = min(best_s, float(np.linalg.norm(sp - corner_s)))
            best_e = min(best_e, float(np.linalg.norm(ep - corner_e)))
        dev = max(best_s, best_e)
        devs.append(dev)
        max_dev = max(max_dev, dev)
        if dev > 0.02:
            bad += 1
    if total > 0:
        devs = np.array(devs) if len(devs) else np.array([0.0])
        print(f"\n[诊断] 最终线段端点与角点对齐偏差：")
        print(f"  线段数={total}, 偏差>2cm 的条数={bad}")
        print(f"  偏差统计: mean={devs.mean():.5f}, median={np.median(devs):.5f}, max={max_dev:.5f}")


def run_pipeline(pcd_path: str, output_dir: str = "pipeline_output",
                 plane_threshold: float = 0.01, cylinder_threshold: float = 0.01, 
                 min_points: int = 50, visualize: bool = True):
    """
    运行完整的新流程
    
    Args:
        pcd_path: PCD文件路径
        output_dir: 输出目录
        plane_threshold: 平面拟合阈值
        cylinder_threshold: 柱面拟合阈值
        min_points: 最少点数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("新流程：按标签拟合平面/柱面并进行面拼接")
    print("="*60)
    
    # 读取点云
    print("\n[步骤0] 读取点云...")
    label_points = read_pcd_by_label(pcd_path)
    edge_points = read_pcd_edge_points(pcd_path)  # 从edge字段读取边缘点
    unlabeled_points = label_points.get(-1, None)
    
    print(f"  标签数量: {len(label_points)}")
    print(f"  边缘点 (edge=1): {len(edge_points)}")
    print(f"  未标注点 (label=-1): {len(unlabeled_points) if unlabeled_points is not None else 0}")
    
    # 0.5 归一化：计算bbox最长轴长度并归一化所有点云
    print("\n[步骤0.5] 归一化点云...")
    all_points_list = []
    if edge_points is not None and len(edge_points) > 0:
        all_points_list.append(edge_points)
    for pts in label_points.values():
        if pts is not None and len(pts) > 0:
            all_points_list.append(pts)
    
    if len(all_points_list) > 0:
        all_points_combined = np.vstack(all_points_list)
        bbox_min = np.min(all_points_combined, axis=0)
        bbox_max = np.max(all_points_combined, axis=0)
        bbox_size = bbox_max - bbox_min
        max_axis_length = float(np.max(bbox_size))
        
        if max_axis_length > 1e-10:
            scale_factor = 1.0 / max_axis_length
            print(f"  BBox尺寸: [{bbox_size[0]:.4f}, {bbox_size[1]:.4f}, {bbox_size[2]:.4f}]")
            print(f"  最长轴长度: {max_axis_length:.4f}")
            print(f"  归一化缩放因子: {scale_factor:.6f}")
            
            # 归一化所有点云
            for label in label_points:
                if label_points[label] is not None and len(label_points[label]) > 0:
                    label_points[label] = label_points[label] * scale_factor
            
            if edge_points is not None and len(edge_points) > 0:
                edge_points = edge_points * scale_factor
            
            if unlabeled_points is not None and len(unlabeled_points) > 0:
                unlabeled_points = unlabeled_points * scale_factor
            
            print(f"  点云已归一化到最长轴长度为1")
        else:
            scale_factor = 1.0
            print(f"  警告: BBox尺寸过小，跳过归一化")
    else:
        scale_factor = 1.0
        print(f"  警告: 没有点云数据，跳过归一化")
    
    # 1. 面拟合
    print("\n[步骤1] 面拟合")
    print("-" * 60)
    
    # 1.1 根据标签拟合
    print("\n1.1 根据标签拟合平面/柱面...")
    labeled_results = fit_surfaces_by_labels(
        label_points,
        cylinder_threshold=cylinder_threshold,
        plane_threshold=plane_threshold,
        min_points=min_points
    )
    print(f"  拟合结果: {len(labeled_results)} 个面")
    
    # 1.2 未标注点（label=-1）直接忽略，不进行拟合
    print("\n1.2 未标注点（label=-1）处理...")
    print(f"  未标注点数量: {len(unlabeled_points) if unlabeled_points is not None else 0}，已忽略（不进行拟合）")
    
    # 直接使用有标签的拟合结果
    all_results = labeled_results
    print(f"\n  总计: {len(all_results)} 个面")
    
    # 1.3 去重
    print("\n1.3 根据位置和朝向剔除重复面...")
    all_results = dedupe_similar_surfaces(all_results)
    print(f"  去重后: {len(all_results)} 个面")
    
    # 2. 判断面朝向
    print("\n[步骤2] 根据点云分布判断面朝向...")
    print("-" * 60)
    all_results = determine_surface_orientations(all_results, label_points)
    
    # 3. 面拼接
    print("\n[步骤3] 面拼接")
    print("-" * 60)
    
    # 3.1 计算交线
    print("\n3.1 计算面面交线...")
    intersections = compute_intersections(all_results)
    print(f"  交线总数: {len(intersections)}")
    
    # 3.2 用边缘点裁剪交线范围
    print("\n3.2 根据边缘点云判断交线合法性并确定范围...")
    edge_distance_thresh = 0.02  # 统一的edge投影距离阈值
    segmented_intersections = segment_intersections_with_edge(
        intersections,
        edge_points,
        distance_thresh=edge_distance_thresh
    )

    # 3.25 三面角点：若三平面两两有交线，则用三平面精确求交点，统一作为共享角点，并用于端点
    # 增加合法性判断：三条交线需对 edge 点云拟合良好
    def compute_triple_plane_corners(all_results_local, seg_list,
                                     edge_pts=None,
                                     line_dist_thresh=0.02,
                                     window_t=0.05,
                                     min_support_per_line=3):
        # 收集已有线与其平面对索引
        line_map = {}
        for seg in seg_list:
            if seg.get('type') != 'line':
                continue
            if 'plane_indices' not in seg:
                continue
            a, b = seg['plane_indices']
            line_map.setdefault(frozenset((a, b)), []).append(seg)
        # 查找三元组
        triples = set()
        planes_num = len(all_results_local)
        # 建立存在边的集合
        edges = set(line_map.keys())
        # 粗略生成可能的三元组：从边集合推导
        edge_list = list(edges)
        for i in range(len(edge_list)):
            for j in range(i+1, len(edge_list)):
                ea = edge_list[i]
                eb = edge_list[j]
                # 三元组必须共享一个平面或组合成三个不同平面
                pa = set(ea)
                pb = set(eb)
                comb = pa.union(pb)
                if len(comb) == 3:
                    a,b,c = tuple(comb)
                    # 第三条边需存在
                    if frozenset((a,b)) in edges and frozenset((a,c)) in edges and frozenset((b,c)) in edges:
                        triples.add(tuple(sorted((a,b,c))))
        # 计算三平面交点
        triple_points = []
        for (ia, ib, ic) in triples:
            try:
                sa = all_results_local[ia].get('surface', {})
                sb = all_results_local[ib].get('surface', {})
                sc = all_results_local[ic].get('surface', {})
                na = np.asarray(sa.get('normal', []), dtype=float)
                pa = np.asarray(sa.get('point', []), dtype=float)
                nb = np.asarray(sb.get('normal', []), dtype=float)
                pb = np.asarray(sb.get('point', []), dtype=float)
                nc = np.asarray(sc.get('normal', []), dtype=float)
                pc = np.asarray(sc.get('point', []), dtype=float)
                if na.shape!=(3,) or nb.shape!=(3,) or nc.shape!=(3,) or pa.shape!=(3,) or pb.shape!=(3,) or pc.shape!=(3,):
                    continue
                A = np.vstack([na, nb, nc])
                bvec = np.array([np.dot(na, pa), np.dot(nb, pb), np.dot(nc, pc)], dtype=float)
                # 解 Ax = b
                try:
                    x = np.linalg.solve(A, bvec)
                except Exception:
                    x = np.linalg.lstsq(A, bvec, rcond=None)[0]

                # 合法性验证（放宽）：三面交点需靠近 edge 点云
                valid = True
                if edge_pts is not None and len(edge_pts) > 0:
                    # 仅要求该交点与 edge 点最近距离不超过阈值（更宽松）
                    diff = edge_pts - x.reshape(1, 3)
                    dmin = float(np.min(np.linalg.norm(diff, axis=1)))
                    point_tol = 1.5 * line_dist_thresh
                    if dmin > point_tol:
                        valid = False
                if not valid:
                    continue
                triple_points.append(((ia,ib,ic), x))
            except Exception:
                continue
        # 将 triple_points 投入每条线，作为候选端点
        line_nodes = {key: [] for key in line_map.keys()}
        for (ia,ib,ic), pt in triple_points:
            pairs = [frozenset((ia,ib)), frozenset((ia,ic)), frozenset((ib,ic))]
            for pr in pairs:
                if pr in line_nodes:
                    line_nodes[pr].append(pt)
        # 用 triple 节点更新线段端点（若 >=2 节点）
        for key, segs in line_map.items():
            nodes = line_nodes.get(key, [])
            if len(nodes) >= 2:
                # 沿该线方向取最小/最大 t
                # 拿第一条 seg 的 point/direction 作为参考
                seg0 = segs[0]
                p0 = np.asarray(seg0['point'], dtype=float)
                d = np.asarray(seg0['direction'], dtype=float)
                d = d / (np.linalg.norm(d) + 1e-12)
                tvals = [float(np.dot(np.asarray(pt)-p0, d)) for pt in nodes]
                i_min = int(np.argmin(tvals))
                i_max = int(np.argmax(tvals))
                sp = np.asarray(nodes[i_min])
                ep = np.asarray(nodes[i_max])
                # 更新该 pair 的所有分段线为共同端点
                for seg in segs:
                    seg['start_point'] = sp
                    seg['end_point'] = ep
                    seg['t_min'] = float(min(tvals))
                    seg['t_max'] = float(max(tvals))
        return seg_list

    segmented_intersections = compute_triple_plane_corners(
        all_results,
        segmented_intersections,
        edge_pts=edge_points if edge_points is not None and len(edge_points) > 0 else None,
        line_dist_thresh=edge_distance_thresh,
        window_t=3.0*edge_distance_thresh,
        min_support_per_line=max(3, int(0.005 * (len(edge_points) if edge_points is not None else 0)))
    )
    
    # 3.3 通过线-线相交吸附端点，统一共享端点坐标
    print("\n3.3 线-线相交端点吸附...")
    segmented_intersections = snap_line_endpoints_by_line_intersections(
        segmented_intersections,
        dist_tol=max(0.5*line_radius if 'line_radius' in locals() else 0.005, 0.003),
        endpoint_tol=0.02
    )
    
    line_count = sum(1 for s in segmented_intersections if s.get('type') == 'line')
    circle_count = sum(1 for s in segmented_intersections if s.get('type') == 'circle')
    print(f"\n  保留的交线: {len(segmented_intersections)} 条 (线段: {line_count}, 圆: {circle_count})")
    
    # 保存结果
    print("\n[步骤4] 保存结果...")
    print("-" * 60)
    
    # 4.1 全局角点去重与构建：候选节点=线段端点 + 线-线几何交点；聚类为角点；将端点吸附到角点并生成索引
    def _angle_small(v1: np.ndarray, v2: np.ndarray, deg_thresh: float = 5.0) -> bool:
        v1 = v1 / (np.linalg.norm(v1) + 1e-12)
        v2 = v2 / (np.linalg.norm(v2) + 1e-12)
        cosang = np.clip(np.dot(v1, v2), -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        return ang < deg_thresh

    def build_global_corners_and_index_lines(segmented: list,
                                             raw_inters: list,  # 保留参数以兼容，但不再使用
                                             corner_tol: float = 0.003,
                                             line_intersect_tol: float = 0.005,
                                             snap_tol: float = 0.005,
                                             parallel_angle_deg: float = 5.0):
        # 1) 收集候选节点：线段端点
        candidates = []
        line_objs = []
        for inter in segmented:
            if inter.get('type') == 'line' and 'start_point' in inter and 'end_point' in inter:
                sp = np.asarray(inter['start_point'])
                ep = np.asarray(inter['end_point'])
                candidates.append(sp)
                candidates.append(ep)
        # 2) 不再引入其它线-线交点，仅使用已分段交线的端点作为候选角点

        if len(candidates) == 0:
            return [], []
        # 转为数组
        candidates = np.asarray(candidates, dtype=float)
        # 3) 简单半径聚合
        corners = []  # centers
        clusters = []  # list of lists
        for pt in candidates:
            matched = False
            for ci, center in enumerate(corners):
                if np.linalg.norm(pt - center) <= corner_tol:
                    clusters[ci].append(pt)
                    corners[ci] = np.mean(clusters[ci], axis=0)
                    matched = True
                    break
            if not matched:
                corners.append(pt.copy())
                clusters.append([pt.copy()])

        # 4) 将每条线段端点吸附到最近角点（snap_tol内）并生成索引
        def find_corner_index(pt: np.ndarray) -> int:
            best = -1
            best_dist = float('inf')
            for i, c in enumerate(corners):
                d = np.linalg.norm(pt - c)
                if d < best_dist:
                    best_dist = d
                    best = i
            # 使用固定吸附半径（上一版实现）
            if best_dist <= snap_tol:
                return best
            # 若未在 snap 范围内，创建新角点
            corners.append(pt.copy())
            return len(corners) - 1

        indexed_lines = []
        seen_edges = set()
        for inter in segmented:
            if inter.get('type') == 'line' and 'start_point' in inter and 'end_point' in inter:
                sp = np.asarray(inter['start_point'])
                ep = np.asarray(inter['end_point'])
                i0 = find_corner_index(sp)
                i1 = find_corner_index(ep)
                if i0 == i1:
                    continue  # 零长度，跳过
                a, b = (i0, i1) if i0 < i1 else (i1, i0)
                key = (a, b)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                line_entry = {
                    'start': i0,
                    'end': i1,
                    'edge_point_count': int(inter.get('edge_point_count', 0))
                }
                # 保存面索引（如果存在）
                if 'plane_indices' in inter:
                    line_entry['surface_indices'] = list(inter['plane_indices'])
                indexed_lines.append(line_entry)

        return corners, indexed_lines

    # 3.x 自适应角点聚合/吸附半径：按edge阈值定标
    corner_tol_adapt = max(1e-5, 1.2 * edge_distance_thresh)
    snap_tol_adapt = max(1e-5, 1.2 * edge_distance_thresh)

    corners, indexed_lines = build_global_corners_and_index_lines(
        segmented_intersections,
        intersections,
        corner_tol=corner_tol_adapt,
        line_intersect_tol=0.005,
        snap_tol=snap_tol_adapt,
        parallel_angle_deg=5.0
    )

    # 3.9 还原尺度：将所有几何结果还原回原始尺度
    if scale_factor != 1.0:
        print("\n[步骤3.9] 还原几何结果到原始尺度...")
        inverse_scale = 1.0 / scale_factor
        print(f"  还原缩放因子: {inverse_scale:.6f}")
        
        # 还原面的参数
        for r in all_results:
            s = r.get('surface', {})
            surf_type = s.get('type', r.get('type', ''))
            
            if surf_type == 'plane':
                # 平面：还原点坐标（法向量不需要还原）
                if 'point' in s:
                    point = np.asarray(s['point'])
                    if point.shape == (3,):
                        s['point'] = (point * inverse_scale).tolist()
            
            elif surf_type == 'cylindrical_surface' or surf_type == 'cylinder':
                # 柱面：还原中心点和半径
                if 'top_center' in s:
                    top = np.asarray(s['top_center'])
                    if top.shape == (3,):
                        s['top_center'] = (top * inverse_scale).tolist()
                if 'bottom_center' in s:
                    bot = np.asarray(s['bottom_center'])
                    if bot.shape == (3,):
                        s['bottom_center'] = (bot * inverse_scale).tolist()
                if 'radius' in s:
                    s['radius'] = float(s['radius']) * inverse_scale
        
        # 还原交线
        for inter in segmented_intersections:
            if inter.get('type') == 'line':
                if 'point' in inter:
                    p = np.asarray(inter['point'])
                    if p.shape == (3,):
                        inter['point'] = (p * inverse_scale).tolist()
                if 'start_point' in inter:
                    sp = np.asarray(inter['start_point'])
                    if sp.shape == (3,):
                        inter['start_point'] = (sp * inverse_scale).tolist()
                if 'end_point' in inter:
                    ep = np.asarray(inter['end_point'])
                    if ep.shape == (3,):
                        inter['end_point'] = (ep * inverse_scale).tolist()
                # t_min 和 t_max 也需要还原
                if 't_min' in inter:
                    inter['t_min'] = float(inter['t_min']) * inverse_scale
                if 't_max' in inter:
                    inter['t_max'] = float(inter['t_max']) * inverse_scale
            
            elif inter.get('type') == 'circle':
                if 'center' in inter:
                    c = np.asarray(inter['center'])
                    if c.shape == (3,):
                        inter['center'] = (c * inverse_scale).tolist()
                if 'radius' in inter:
                    inter['radius'] = float(inter['radius']) * inverse_scale
        
        # 还原角点
        for i in range(len(corners)):
            corners[i] = np.asarray(corners[i]) * inverse_scale
        
        # 还原点云用于可视化（确保点云和几何结果在同一尺度）
        for label in label_points:
            if label_points[label] is not None and len(label_points[label]) > 0:
                label_points[label] = label_points[label] * inverse_scale
        
        if edge_points is not None and len(edge_points) > 0:
            edge_points = edge_points * inverse_scale
        
        if unlabeled_points is not None and len(unlabeled_points) > 0:
            unlabeled_points = unlabeled_points * inverse_scale
        
        print(f"  所有几何结果和点云已还原到原始尺度")
    
    # 保存拟合结果
    import json
    results_summary = {
        'surfaces': [],
        'intersections': [],
        'corners': [],
        'lines': [],
        'curves': []
    }
    
    for i, r in enumerate(all_results):
        s = r.get('surface', {})
        surf_type = s.get('type', r.get('type', 'unknown'))
        
        surface_entry = {
            'index': i,
            'type': surf_type,
            'label': r.get('label', 'unknown'),
            'inlier_count': r.get('inlier_count', 0),
            'error': r.get('error', 0.0)
        }
        
        # 根据类型保存相应参数
        if surf_type == 'plane':
            # 平面：保存法向量和点
            normal = s.get('normal', [])
            point = s.get('point', [])
            if normal and point:
                surface_entry['normal'] = np.asarray(normal).tolist() if not isinstance(normal, list) else normal
                surface_entry['point'] = np.asarray(point).tolist() if not isinstance(point, list) else point
                # 计算并保存 d 值（可选，方便后续使用）
                normal_arr = np.asarray(normal)
                point_arr = np.asarray(point)
                if normal_arr.shape == (3,) and point_arr.shape == (3,):
                    d_value = -np.dot(normal_arr, point_arr)
                    surface_entry['d'] = float(d_value)
        
        elif surf_type == 'cylindrical_surface' or surf_type == 'cylinder':
            # 柱面：保存轴向、位置、半径
            top_center = s.get('top_center', [])
            bottom_center = s.get('bottom_center', [])
            radius = s.get('radius', 0.0)
            
            if top_center and bottom_center and radius > 0:
                top_center = np.asarray(top_center).tolist() if not isinstance(top_center, list) else top_center
                bottom_center = np.asarray(bottom_center).tolist() if not isinstance(bottom_center, list) else bottom_center
                
                surface_entry['top_center'] = top_center
                surface_entry['bottom_center'] = bottom_center
                surface_entry['radius'] = float(radius)
                
                # 计算并保存轴向和中心点（方便后续使用）
                top_arr = np.asarray(top_center)
                bottom_arr = np.asarray(bottom_center)
                if top_arr.shape == (3,) and bottom_arr.shape == (3,):
                    axis = top_arr - bottom_arr
                    axis_len = np.linalg.norm(axis)
                    if axis_len > 1e-6:
                        axis_normalized = axis / axis_len
                        center = (top_arr + bottom_arr) / 2
                        surface_entry['axis'] = axis_normalized.tolist()
                        surface_entry['center'] = center.tolist()
                        surface_entry['height'] = float(axis_len)
        
        results_summary['surfaces'].append(surface_entry)
    
    for i, inter in enumerate(segmented_intersections):
        inter_summary = {
            'index': i,
            'type': inter.get('type', 'unknown')
        }
        if inter.get('type') == 'line':
            # 仍保留坐标（兼容），但推荐使用 corners + lines
            if 'start_point' in inter:
                inter_summary['start_point'] = np.asarray(inter['start_point']).tolist()
            if 'end_point' in inter:
                inter_summary['end_point'] = np.asarray(inter['end_point']).tolist()
            inter_summary['edge_point_count'] = inter.get('edge_point_count', 0)
            # 保存面索引
            if 'plane_indices' in inter:
                inter_summary['surface_indices'] = list(inter['plane_indices'])
        elif inter.get('type') == 'circle':
            inter_summary['center'] = np.asarray(inter.get('center', [])).tolist()
            inter_summary['radius'] = float(inter.get('radius', 0))
            inter_summary['edge_point_count'] = inter.get('edge_point_count', 0)
            # 保存面索引
            if 'surface_indices' in inter:
                inter_summary['surface_indices'] = list(inter['surface_indices'])
        results_summary['intersections'].append(inter_summary)

    # 写入角点与线索引
    results_summary['corners'] = [np.asarray(c).tolist() for c in corners]
    results_summary['lines'] = indexed_lines
    # 写入曲线（保留圆）
    curves_out = []
    for inter in segmented_intersections:
        if inter.get('type') == 'circle':
            curve_entry = {
                'type': 'circle',
                'center': np.asarray(inter.get('center', [])).tolist(),
                'normal': np.asarray(inter.get('normal', [])).tolist(),
                'radius': float(inter.get('radius', 0.0)),
                'edge_point_count': int(inter.get('edge_point_count', 0))
            }
            # 保存面索引
            if 'surface_indices' in inter:
                curve_entry['surface_indices'] = list(inter['surface_indices'])
            curves_out.append(curve_entry)
    results_summary['curves'] = curves_out
    
    results_file = os.path.join(output_dir, "pipeline_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"  结果已保存到: {results_file}")

    # 导出最终 OBJ（与 JSON 同目录）
    try:
        surface_meshes = build_surface_meshes(
            results_summary['surfaces'],
            results_summary['lines'],
            results_summary.get('curves', []),
            results_summary['corners'],
        )
        export_final_obj(corners, indexed_lines, output_dir,
                         line_radius=0.005, sphere_radius=0.01,
                         filename="pipeline_results.obj",
                         curves=results_summary.get('curves', []),
                         surface_meshes=surface_meshes)
        # 结构化导出
        export_parts_objs(corners, indexed_lines, output_dir,
                          line_radius=0.005, sphere_radius=0.01,
                          curves=results_summary.get('curves', []),
                          surface_meshes=surface_meshes,
                          subdir="parts")
        # 单文件结构化（每个面/点/线/曲线独立 object）
        export_structured_obj(
            corners,
            indexed_lines,
            output_dir,
            curves=results_summary.get('curves', []),
            surface_meshes=surface_meshes,
            line_radius=0.005,
            sphere_radius=0.01,
            filename="pipeline_results_structured.obj",
            mtl_name="pipeline_results_structured.mtl"
        )
        face_ok = sum(1 for m in surface_meshes if m is not None)
        print(f"  面片 OBJ: {face_ok}/{len(surface_meshes)} 个面已导出至 parts/face_*.obj")
        print(f"  合并 OBJ: pipeline_results.obj, pipeline_results_structured.obj")
    except Exception as e:
        print(f"  警告: 导出 OBJ 失败: {e}")
    
    # 可视化交线（拟合中间结果）
    if visualize and len(segmented_intersections) > 0:
        print("\n[步骤5] 可视化交线(拟合中间结果)...")
        print("-" * 60)
        visualize_intersections(segmented_intersections, label_points, edge_points)
    
    # 可视化最终结果（角点+线段索引）
    if visualize and len(corners) > 0 and len(indexed_lines) > 0:
        print("\n[步骤6] 可视化最终结果(角点/线段)...")
        print("-" * 60)
        # 诊断对齐情况
        debug_check_final_alignment(segmented_intersections, corners, indexed_lines)
        visualize_final_geometry(corners, indexed_lines, results_summary.get('curves', []))
    
    print("\n" + "="*60)
    print("流程完成！")
    print("="*60)
    
    return all_results, segmented_intersections


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='新流程：按标签拟合平面/柱面并进行面拼接')
    parser.add_argument('--pcd', type=str, default='point_cloud_filtered.pcd',
                       help='输入PCD文件路径')
    parser.add_argument('--output', type=str, default='pipeline_output',
                       help='输出目录')
    parser.add_argument('--plane_threshold', type=float, default=0.01,
                       help='平面拟合阈值')
    parser.add_argument('--cylinder_threshold', type=float, default=0.01,
                       help='柱面拟合阈值')
    parser.add_argument('--min_points', type=int, default=50,
                       help='最少点数')
    parser.add_argument('--no_visualize', action='store_true',
                       help='不显示可视化窗口')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pcd):
        print(f"错误: 文件不存在: {args.pcd}")
    else:
        run_pipeline(
            pcd_path=args.pcd,
            output_dir=args.output,
            plane_threshold=args.plane_threshold,
            cylinder_threshold=args.cylinder_threshold,
            min_points=args.min_points,
            visualize=not args.no_visualize
        )

