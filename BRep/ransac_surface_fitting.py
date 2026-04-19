import numpy as np
import json
import os
import copy
from typing import List, Tuple, Dict, Optional
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import least_squares
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import open3d as o3d


def read_pcd_by_label(pcd_path: str) -> Dict[int, np.ndarray]:
    """
    读取PCD文件并按label分组点云
    
    Args:
        pcd_path: PCD文件路径
        
    Returns:
        字典：{label: points_array}，其中points_array是[N, 3]的点数组
    """
    label_points = {}
    
    with open(pcd_path, 'r') as f:
        lines = f.readlines()
        
        # 找到FIELDS行
        field_idx = -1
        for i, line in enumerate(lines):
            if line.startswith('FIELDS'):
                field_idx = i
                break
        
        if field_idx == -1:
            raise ValueError("无法找到FIELDS行")
        
        # 解析字段
        fields = lines[field_idx].strip().split()[1:]
        
        # 检查是否有label字段
        if 'label' not in fields:
            raise ValueError("PCD文件中没有找到label字段")
        
        label_idx = fields.index('label')
        x_idx = fields.index('x')
        y_idx = fields.index('y')
        z_idx = fields.index('z')
        
        # 找到DATA行
        data_start_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == 'DATA ascii':
                data_start_idx = i + 1
                break
        
        # 读取数据并按label分组
        for line in lines[data_start_idx:]:
            if not line.strip():
                continue
                
            parts = line.strip().split()
            if len(parts) < len(fields):
                continue
            
            try:
                label = int(float(parts[label_idx]))
                x = float(parts[x_idx])
                y = float(parts[y_idx])
                z = float(parts[z_idx])
                
                if label not in label_points:
                    label_points[label] = []
                
                label_points[label].append([x, y, z])
            except (ValueError, IndexError):
                continue
    
    # 转换为numpy数组
    result = {}
    for label, points in label_points.items():
        if len(points) > 0:
            result[label] = np.array(points)
    
    return result


class Plane:
    """
    平面类
    平面方程: ax + by + cz + d = 0
    其中[a, b, c]是法向量，d是偏移量
    """
    
    def __init__(self, normal: np.ndarray, point: np.ndarray, boundary: Optional[np.ndarray] = None):
        """
        初始化平面
        
        Args:
            normal: 法向量 [3]
            point: 平面上的一个点 [3]
            boundary: 边界点 [N, 3]（平面上的凸包顶点）
        """
        self.normal = normal / np.linalg.norm(normal)  # 归一化法向量
        self.point = point
        self.d = -np.dot(self.normal, point)  # 计算d值
        self.boundary = boundary if boundary is not None else np.array([])
    
    def fit(self, points: np.ndarray, threshold: float = 0.01) -> Tuple[float, np.ndarray]:
        """
        使用RANSAC拟合平面
        
        Args:
            points: 点云 [N, 3]
            threshold: 内点距离阈值
            
        Returns:
            (拟合误差, 内点掩码)
        """
        n_points = len(points)
        if n_points < 3:
            return float('inf'), np.array([])
        
        best_normal = None
        best_point = None
        best_inliers = None
        best_count = 0
        
        max_iterations = min(1000, n_points * 10)
        
        for _ in range(max_iterations):
            # 随机选择3个点
            sample_indices = np.random.choice(n_points, size=3, replace=False)
            p1, p2, p3 = points[sample_indices[0]], points[sample_indices[1]], points[sample_indices[2]]
            
            # 计算平面法向量
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            
            if norm < 1e-6:
                continue
            
            normal = normal / norm
            
            # 计算所有点到平面的距离
            vec_to_points = points - p1
            distances = np.abs(np.dot(vec_to_points, normal))
            
            # 找到内点
            inliers = distances < threshold
            inlier_count = np.sum(inliers)
            
            if inlier_count > best_count:
                best_count = inlier_count
                best_normal = normal
                best_point = p1
                best_inliers = inliers
                
                # 提前终止
                if best_count / n_points > 0.9:
                    break
        
        if best_normal is None or best_count < 3:
            return float('inf'), np.array([])
        
        # 更新平面参数
        self.normal = best_normal
        self.point = best_point
        self.d = -np.dot(self.normal, self.point)
        
        # 计算拟合误差
        inlier_points = points[best_inliers]
        vec_to_inliers = inlier_points - self.point
        errors = np.abs(np.dot(vec_to_inliers, self.normal))
        fit_error = np.mean(errors)
        
        return fit_error, best_inliers
    
    def compute_boundary(self, points: np.ndarray) -> np.ndarray:
        """
        计算平面边界（凸包）
        
        Args:
            points: 内点 [N, 3]
            
        Returns:
            边界点 [M, 3]（凸包顶点）
        """
        if len(points) < 3:
            return np.array([])
        
        # 投影点到平面
        vec_to_points = points - self.point
        
        # 构建平面坐标系
        if abs(self.normal[2]) < 0.9:
            u = np.array([0, 0, 1])
        else:
            u = np.array([1, 0, 0])
        
        u = u - np.dot(u, self.normal) * self.normal
        u = u / (np.linalg.norm(u) + 1e-10)
        v = np.cross(self.normal, u)
        v = v / (np.linalg.norm(v) + 1e-10)
        
        # 计算2D坐标
        points_2d = np.array([
            np.dot(vec_to_points, u),
            np.dot(vec_to_points, v)
        ]).T
        
        # 计算2D凸包
        try:
            hull = ConvexHull(points_2d)
            # 确保hull.vertices是整数类型
            hull_vertex_indices = hull.vertices.astype(np.int32) if isinstance(hull.vertices, np.ndarray) else np.array(hull.vertices, dtype=np.int32)
            hull_vertices_2d = points_2d[hull_vertex_indices]
            
            # 转换回3D
            boundary_3d = []
            for vertex_2d in hull_vertices_2d:
                point_3d = self.point + vertex_2d[0] * u + vertex_2d[1] * v
                boundary_3d.append(point_3d)
            
            self.boundary = np.array(boundary_3d)
            return self.boundary
        except:
            # 如果凸包计算失败，返回空数组
            return np.array([])
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'type': 'plane',
            'normal': self.normal.tolist(),
            'point': self.point.tolist(),
            'd': float(self.d),
            'boundary': self.boundary.tolist() if len(self.boundary) > 0 else []
        }
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """计算点到平面的距离"""
        return abs(np.dot(point - self.point, self.normal))


def get_2d_circle_from_3_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    通过三个2D点计算圆的中心和半径
    """
    # 检查三点是否共线
    if abs((b[1] - a[1]) * (c[0] - a[0]) - (c[1] - a[1]) * (b[0] - a[0])) < 1e-10:
        return None, None
    
    # 计算两弦的中垂线交点
    # 弦1: a-b的中点
    mid1 = (a + b) / 2
    dir1 = b - a
    normal1 = np.array([-dir1[1], dir1[0]])  # 垂直方向
    normal1 = normal1 / (np.linalg.norm(normal1) + 1e-10)
    
    # 弦2: a-c的中点
    mid2 = (a + c) / 2
    dir2 = c - a
    normal2 = np.array([-dir2[1], dir2[0]])  # 垂直方向
    normal2 = normal2 / (np.linalg.norm(normal2) + 1e-10)
    
    # 求解两条直线的交点
    # mid1 + t1 * normal1 = mid2 + t2 * normal2
    A = np.column_stack([normal1, -normal2])
    b_vec = mid2 - mid1
    try:
        t = np.linalg.solve(A, b_vec)
        center = mid1 + t[0] * normal1
        radius = np.linalg.norm(a - center)
        return center, radius
    except:
        return None, None


def are_2d_points_collinear(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """检查三个2D点是否共线"""
    return abs((b[1] - a[1]) * (c[0] - a[0]) - (c[1] - a[1]) * (b[0] - a[0])) < 1e-10


def fit_circle_ransac_2d(points_2d: np.ndarray, threshold: float = 0.01, 
                         max_iterations: int = 50, min_inliers: int = 10) -> Tuple[Optional[np.ndarray], Optional[float], np.ndarray]:
    """
    使用RANSAC在2D平面上拟合圆
    
    Args:
        points_2d: 2D点 [N, 2]
        threshold: 内点阈值
        max_iterations: 最大迭代次数
        min_inliers: 最少内点数
        
    Returns:
        (圆心, 半径, 内点掩码)
    """
    n_points = len(points_2d)
    if n_points < 3:
        return None, None, np.array([])
    
    best_center = None
    best_radius = None
    best_inliers = None
    best_count = 0
    
    for _ in range(max_iterations):
        # 随机选择3个点
        sample_indices = np.random.choice(n_points, size=3, replace=False)
        a, b, c = points_2d[sample_indices[0]], points_2d[sample_indices[1]], points_2d[sample_indices[2]]
        
        # 检查是否共线
        if are_2d_points_collinear(a, b, c):
            continue
        
        # 计算圆
        center, radius = get_2d_circle_from_3_points(a, b, c)
        
        if center is None or radius is None or radius < 1e-6:
            continue
        
        # 计算所有点到圆的距离
        distances = np.linalg.norm(points_2d - center, axis=1)
        radius_errors = np.abs(distances - radius)
        
        # 找到内点
        inliers = radius_errors < threshold
        inlier_count = np.sum(inliers)
        
        if inlier_count > best_count:
            best_count = inlier_count
            best_center = center
            best_radius = radius
            best_inliers = inliers
            
            if best_count / n_points > 0.85:
                break
    
    if best_center is None or best_count < min_inliers:
        return None, None, np.array([])
    
    return best_center, best_radius, best_inliers


def get_cylinder_axises(points: np.ndarray) -> List[np.ndarray]:
    """
    计算可能的圆柱轴向（使用PCA和MOBB）
    
    Args:
        points: 点云 [N, 3]
        
    Returns:
        轴向列表，每个轴向是一个3x3的旋转矩阵（第三列是轴向）
    """
    axises = []
    
    # PCA轴向
    pca = PCA(n_components=3)
    pca.fit(points)
    pca_components = pca.components_  # [3, 3]
    axises.append(pca_components)
    
    # MOBB轴向
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    mobb = pcd.get_minimal_oriented_bounding_box()
    box_points = np.asarray(mobb.get_box_points())
    
    # 从box_points计算三个轴
    if len(box_points) >= 4:
        axis1 = box_points[0] - box_points[1]
        axis2 = box_points[0] - box_points[2]
        axis3 = box_points[0] - box_points[3]
        
        axis1 = axis1 / (np.linalg.norm(axis1) + 1e-10)
        axis2 = axis2 / (np.linalg.norm(axis2) + 1e-10)
        axis3 = axis3 / (np.linalg.norm(axis3) + 1e-10)
        
        # 构建旋转矩阵（第三个轴作为圆柱轴向）
        if abs(np.dot(axis1, axis2)) < 0.9:  # 确保不平行
            axis3_mobb = np.cross(axis1, axis2)
            axis3_mobb = axis3_mobb / (np.linalg.norm(axis3_mobb) + 1e-10)
            # 构造正交基
            axis1 = axis1 / (np.linalg.norm(axis1) + 1e-10)
            axis2 = np.cross(axis3_mobb, axis1)
            axis2 = axis2 / (np.linalg.norm(axis2) + 1e-10)
            mobb_components = np.array([axis1, axis2, axis3_mobb])
            axises.append(mobb_components)
    
    return axises


class CylindricalSurface:
    """
    柱面类
    使用top_center和bottom_center定义轴线，加上radius
    """
    
    def __init__(self, top_center: np.ndarray, bottom_center: np.ndarray, 
                 radius: float, boundary: Optional[np.ndarray] = None):
        """
        初始化柱面
        
        Args:
            top_center: 圆柱顶部中心点 [3]
            bottom_center: 圆柱底部中心点 [3]
            radius: 半径
            boundary: 边界点 [N, 3]
        """
        self.top_center = top_center
        self.bottom_center = bottom_center
        self.radius = radius
        self.boundary = boundary if boundary is not None else np.array([])
    
    def get_direction(self) -> np.ndarray:
        """获取轴线方向"""
        delta = self.top_center - self.bottom_center
        return delta / (np.linalg.norm(delta) + 1e-10)
    
    def get_center(self) -> np.ndarray:
        """获取轴线中点"""
        return (self.top_center + self.bottom_center) / 2
    
    def get_height(self) -> float:
        """获取圆柱高度"""
        return np.linalg.norm(self.top_center - self.bottom_center)
    
    def fit(self, points: np.ndarray, threshold: float = 0.01) -> Tuple[float, np.ndarray]:
        """
        拟合柱面（参考Cylinders目录的方法）
        
        Args:
            points: 点云 [N, 3]
            threshold: 拟合误差阈值
            
        Returns:
            (拟合误差, 内点掩码)
        """
        n_points = len(points)
        if n_points < 6:
            return float('inf'), np.array([])
        
        # 1. 中心化点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        mobb_center = pcd.get_minimal_oriented_bounding_box().get_center()
        central_points = points - mobb_center
        
        # 2. 计算可能的轴向
        try:
            axises = get_cylinder_axises(central_points)
        except:
            return float('inf'), np.array([])
        
        if len(axises) == 0:
            return float('inf'), np.array([])
        
        # 3. 对每个轴向进行拟合
        best_cylinder = None
        best_inliers = None
        best_inlier_count = 0
        
        for axis_matrix in axises:
            # 旋转矩阵的第三列是圆柱轴向
            axis_direction = axis_matrix[2]  # [3]
            
            # 将点云投影到垂直于轴线的平面（局部坐标系）
            # 投影后的点：[x, y, z] -> [x', y', z']，其中z'沿轴线方向
            rotated_points = np.dot(central_points, axis_matrix.T)  # [N, 3]
            
            # 提取2D圆面上的点（前两维）
            circle_points_2d = rotated_points[:, :2]  # [N, 2]
            z_coords = rotated_points[:, 2]  # [N]
            
            # 在2D平面上拟合圆
            circle_center_2d, circle_radius, circle_inliers = fit_circle_ransac_2d(
                circle_points_2d, 
                threshold=threshold,
                max_iterations=50,
                min_inliers=max(6, n_points // 10)
            )
            
            if circle_center_2d is None or circle_radius is None:
                continue
            
            # 检查点分布是否合理（内点是否分布在圆周的多个扇区）
            if not self._check_point_distribution(
                rotated_points, circle_center_2d, circle_radius, z_coords, threshold
            ):
                continue  # 如果分布不合理，跳过这个轴向
            
            # 计算内点数（使用圆柱距离）
            inlier_count = self._count_cylinder_inliers(
                rotated_points, circle_center_2d, circle_radius, z_coords, threshold
            )
            
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                
                # 构建局部坐标系的圆柱
                # 在局部坐标系中，圆柱沿z轴方向
                z_min, z_max = np.min(z_coords), np.max(z_coords)
                local_top = np.array([circle_center_2d[0], circle_center_2d[1], z_max])
                local_bottom = np.array([circle_center_2d[0], circle_center_2d[1], z_min])
                
                # 转换回世界坐标系
                world_top = np.dot(local_top, axis_matrix) + mobb_center
                world_bottom = np.dot(local_bottom, axis_matrix) + mobb_center
                
                best_cylinder = (world_top, world_bottom, circle_radius)
                
                # 计算内点掩码
                distances = np.linalg.norm(circle_points_2d - circle_center_2d, axis=1)
                radius_errors = np.abs(distances - circle_radius)
                z_min, z_max = np.min(z_coords), np.max(z_coords)
                z_in_range = (z_coords >= z_min - threshold) & (z_coords <= z_max + threshold)
                best_inliers = (radius_errors < threshold) & z_in_range
        
        if best_cylinder is None or best_inlier_count < 6:
            return float('inf'), np.array([])
        
        # 设置圆柱参数
        self.top_center, self.bottom_center, self.radius = best_cylinder
        
        # 确保best_inliers是布尔数组
        if best_inliers is None:
            return float('inf'), np.array([])
        
        # 确保best_inliers是布尔类型
        if not isinstance(best_inliers, np.ndarray) or best_inliers.dtype != bool:
            # 如果不是布尔数组，转换
            best_inliers = np.array(best_inliers, dtype=bool)
        
        # 计算拟合误差（综合考虑距离误差和分布均匀性）
        direction = self.get_direction()
        delta = points - self.bottom_center
        delta_cross = np.linalg.norm(np.cross(delta, direction), axis=1)
        radius_errors = np.abs(delta_cross - self.radius)
        
        # 基础拟合误差（距离误差）
        if np.sum(best_inliers) > 0:
            base_error = np.mean(radius_errors[best_inliers])
        else:
            base_error = float('inf')
        
        # 分布均匀性惩罚（检查内点在圆周和高度方向的分布）
        if np.sum(best_inliers) > 0:
            distribution_penalty = self._compute_distribution_penalty(
                points[best_inliers], threshold
            )
        else:
            distribution_penalty = 1.0
        
        # 综合误差 = 基础误差 + 分布惩罚
        fit_error = base_error + distribution_penalty
        
        return fit_error, best_inliers
    
    def _compute_distribution_penalty(self, inlier_points: np.ndarray, threshold: float) -> float:
        """
        计算分布均匀性惩罚
        
        Args:
            inlier_points: 内点 [M, 3]
            threshold: 阈值
            
        Returns:
            分布惩罚值（越大说明分布越不均匀）
        """
        if len(inlier_points) < 3:
            return 1.0  # 点数太少，惩罚最大
        
        direction = self.get_direction()
        
        # 构建垂直于轴线的坐标系
        if abs(direction[2]) < 0.9:
            u = np.array([0, 0, 1])
        else:
            u = np.array([1, 0, 0])
        
        u = u - np.dot(u, direction) * direction
        u = u / (np.linalg.norm(u) + 1e-10)
        v = np.cross(direction, u)
        v = v / (np.linalg.norm(v) + 1e-10)
        
        # 投影内点到柱面
        delta = inlier_points - self.bottom_center
        proj_lengths = np.dot(delta, direction)
        proj_points = self.bottom_center + proj_lengths[:, np.newaxis] * direction
        vec_from_axis = inlier_points - proj_points
        
        # 计算角度分布
        angles = np.arctan2(np.dot(vec_from_axis, v), np.dot(vec_from_axis, u))
        angles_deg = np.degrees(angles) % 360
        
        # 检查角度分布（分成12个扇区）
        partition_of_circle = 12
        sector_angle = 360 / partition_of_circle
        sector_indices = ((angles_deg - 1) // sector_angle).astype(np.int32) % partition_of_circle
        # 确保索引在有效范围内
        sector_indices = np.clip(sector_indices, 0, partition_of_circle - 1)
        unique_sectors, counts = np.unique(sector_indices, return_counts=True)
        # 确保unique_sectors是整数类型且在有效范围内
        unique_sectors = unique_sectors.astype(np.int32)
        unique_sectors = np.clip(unique_sectors, 0, partition_of_circle - 1)
        
        # 计算覆盖的扇区比例
        coverage_ratio = len(unique_sectors) / partition_of_circle
        
        # 检查高度分布（分成10段）
        height_partitions = 10
        height = self.get_height()
        if height < 1e-6:
            height_penalty = 1.0
        else:
            height_segments = ((proj_lengths / height) * height_partitions).astype(np.int32)
            height_segments = np.clip(height_segments, 0, height_partitions - 1)
            # 确保高度段索引是整数类型
            height_segments = height_segments.astype(np.int32)
            unique_heights = len(np.unique(height_segments))
            height_coverage = unique_heights / height_partitions
            height_penalty = max(0, 1.0 - height_coverage)
        
        # 角度分布惩罚（覆盖的扇区越少，惩罚越大）
        angle_penalty = max(0, 1.0 - coverage_ratio)
        
        # 综合惩罚
        total_penalty = (angle_penalty + height_penalty) * threshold * 2.0
        
        return total_penalty
    
    def _count_cylinder_inliers(self, rotated_points: np.ndarray, circle_center_2d: np.ndarray,
                                circle_radius: float, z_coords: np.ndarray, threshold: float) -> int:
        """计算圆柱内点数，并检查内点分布"""
        circle_points_2d = rotated_points[:, :2]
        distances = np.linalg.norm(circle_points_2d - circle_center_2d, axis=1)
        radius_errors = np.abs(distances - circle_radius)
        
        # 检查z坐标是否在范围内
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        z_in_range = (z_coords >= z_min - threshold) & (z_coords <= z_max + threshold)
        
        inliers = (radius_errors < threshold) & z_in_range
        return np.sum(inliers)
    
    def _check_point_distribution(self, rotated_points: np.ndarray, circle_center_2d: np.ndarray,
                                  circle_radius: float, z_coords: np.ndarray, threshold: float,
                                  partition_of_circle: int = 12, minimal_of_partition: int = 3,
                                  minimal_of_acc_partition: int = 3) -> bool:
        """
        检查内点在柱面上的分布（参考Cylinders的rectify方法）
        
        Args:
            rotated_points: 旋转后的点 [N, 3]
            circle_center_2d: 2D圆心
            circle_radius: 半径
            z_coords: z坐标 [N]
            threshold: 阈值
            partition_of_circle: 圆周分区数
            minimal_of_partition: 每个分区最少点数
            minimal_of_acc_partition: 有效分区最少数量
            
        Returns:
            是否通过分布检查
        """
        circle_points_2d = rotated_points[:, :2]
        distances = np.linalg.norm(circle_points_2d - circle_center_2d, axis=1)
        
        # 应用距离筛选，仅保留在 [radius - threshold, radius + threshold] 范围内的点
        valid_mask = (distances > (circle_radius - threshold)) & (distances < (circle_radius + threshold))
        
        if np.sum(valid_mask) < minimal_of_partition * minimal_of_acc_partition:
            return False
        
        # 计算角度
        delta_x = circle_points_2d[valid_mask, 0] - circle_center_2d[0]
        delta_y = circle_points_2d[valid_mask, 1] - circle_center_2d[1]
        angles = np.degrees(np.arctan2(delta_y, delta_x)) % 360
        
        # 分区
        sector_angle = 360 / partition_of_circle
        votes = np.zeros(partition_of_circle, dtype=np.int32)
        sector_indices = ((angles - 1) // sector_angle).astype(np.int32) % partition_of_circle
        
        # 确保sector_indices是整数类型且在有效范围内
        sector_indices = sector_indices.astype(np.int32)
        sector_indices = np.clip(sector_indices, 0, partition_of_circle - 1)
        
        # 统计每个分区的点数
        unique, counts = np.unique(sector_indices, return_counts=True)
        # 确保unique是整数类型且在有效范围内
        unique = unique.astype(np.int32)
        unique = np.clip(unique, 0, partition_of_circle - 1)
        
        # 确保votes数组足够大
        if len(votes) < partition_of_circle:
            votes = np.zeros(partition_of_circle, dtype=np.int32)
        
        # 安全地更新votes
        valid_mask = (unique >= 0) & (unique < partition_of_circle)
        if np.any(valid_mask):
            votes[unique[valid_mask]] += counts[valid_mask]
        
        # 计算有效扇区（至少包含minimal_of_partition个点的分区）
        valid_sectors = np.sum(votes >= minimal_of_partition)
        
        # 如果有效扇区数量小于minimal_of_acc_partition，说明分布不均匀
        return valid_sectors >= minimal_of_acc_partition
    
    def compute_boundary(self, points: np.ndarray) -> np.ndarray:
        """
        计算柱面边界
        
        Args:
            points: 内点 [N, 3]
            
        Returns:
            边界点 [M, 3]
        """
        if len(points) < 3:
            return np.array([])
        
        direction = self.get_direction()
        
        # 构建垂直于轴线的坐标系
        if abs(direction[2]) < 0.9:
            u = np.array([0, 0, 1])
        else:
            u = np.array([1, 0, 0])
        
        u = u - np.dot(u, direction) * direction
        u = u / (np.linalg.norm(u) + 1e-10)
        v = np.cross(direction, u)
        v = v / (np.linalg.norm(v) + 1e-10)
        
        # 在高度和角度上采样边界点
        height = self.get_height()
        height_samples = np.linspace(0, height, 10)
        angle_samples = np.linspace(0, 2*np.pi, 16)
        
        boundary_points = []
        for h in height_samples:
            axis_point_h = self.bottom_center + h * direction
            for angle in angle_samples:
                boundary_point = axis_point_h + self.radius * (np.cos(angle) * u + np.sin(angle) * v)
                boundary_points.append(boundary_point)
        
        self.boundary = np.array(boundary_points)
        return self.boundary
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'type': 'cylindrical_surface',
            'top_center': self.top_center.tolist(),
            'bottom_center': self.bottom_center.tolist(),
            'radius': float(self.radius),
            'height': float(self.get_height()),
            'boundary': self.boundary.tolist() if len(self.boundary) > 0 else []
        }
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """计算点到柱面的距离"""
        direction = self.get_direction()
        delta = point - self.bottom_center
        delta_cross = np.linalg.norm(np.cross(delta, direction))
        return abs(delta_cross - self.radius)


def fit_surface_to_points(points: np.ndarray, 
                         plane_threshold: float = 0.01,
                         cylinder_threshold: float = 0.01,
                         min_points: int = 10) -> Dict:
    """
    对点集进行两种面拟合，选择效果最好的
    
    Args:
        points: 点云 [N, 3]
        plane_threshold: 平面拟合阈值
        cylinder_threshold: 柱面拟合阈值
        min_points: 最少点数
        
    Returns:
        最佳拟合结果的字典
    """
    if len(points) < min_points:
        return None
    
    results = []
    
    # 1. 拟合平面
    try:
        plane = Plane(normal=np.array([0, 0, 1]), point=points[0])
        plane_error, plane_inliers = plane.fit(points, threshold=plane_threshold)
        plane_inlier_points = points[plane_inliers]
        
        if len(plane_inlier_points) >= min_points:
            plane.compute_boundary(plane_inlier_points)
            results.append({
                'type': 'plane',
                'object': plane,
                'error': plane_error,
                'inlier_count': len(plane_inlier_points),
                'inlier_ratio': len(plane_inlier_points) / len(points)
            })
    except Exception as e:
        print(f"平面拟合失败: {e}")
    
    # 2. 拟合柱面
    try:
        cylinder = CylindricalSurface(
            top_center=points[0],
            bottom_center=points[0],
            radius=0.1
        )
        cylinder_error, cylinder_inliers = cylinder.fit(points, threshold=cylinder_threshold)
        cylinder_inlier_points = points[cylinder_inliers]
        
        if len(cylinder_inlier_points) >= min_points:
            cylinder.compute_boundary(cylinder_inlier_points)
            results.append({
                'type': 'cylindrical_surface',
                'object': cylinder,
                'error': cylinder_error,
                'inlier_count': len(cylinder_inlier_points),
                'inlier_ratio': len(cylinder_inlier_points) / len(points)
            })
    except Exception as e:
        print(f"柱面拟合失败: {e}")
    
    if len(results) == 0:
        return None
    
    # 选择最佳结果（综合考虑误差和内点比例）
    best_result = None
    best_score = float('inf')
    
    for result in results:
        # 评分函数：误差 + 内点比例的倒数惩罚
        score = result['error'] / (result['inlier_ratio'] + 0.01)
        if score < best_score:
            best_score = score
            best_result = result
    
    return {
        'type': best_result['type'],
        'surface': best_result['object'].to_dict(),
        'error': best_result['error'],
        'inlier_count': best_result['inlier_count'],
        'inlier_ratio': best_result['inlier_ratio'],
        'total_points': len(points)
    }


def are_surfaces_similar_with_debug(result1: Dict, result2: Dict,
                        plane_normal_threshold: float = 0.1,
                        plane_distance_threshold: float = 0.05,
                        cylinder_axis_threshold: float = 0.1,
                        cylinder_radius_threshold: float = 0.02,
                        cylinder_position_threshold: float = 0.05) -> Tuple[bool, str]:
    """
    判断两个面是否相似（带调试信息，仅判断几何参数：位置和法向）
    
    Returns:
        (是否相似, 调试信息字符串)
    """
    surface1 = result1.get('surface', {})
    surface2 = result2.get('surface', {})
    type1 = surface1.get('type', result1.get('type', ''))
    type2 = surface2.get('type', result2.get('type', ''))
    
    debug_lines = []
    
    # 不同类型的面不相似
    if type1 != type2:
        return False, f"类型不同: {type1} vs {type2}"
    
    debug_lines.append(f"类型相同: {type1}")
    
    # 检查几何参数相似性（位置和法向）
    geometry_similar = False
    geometry_info = ""
    
    if type1 == 'plane':
        # 平面相似度判断
        normal1 = np.array(surface1.get('normal', []))
        normal2 = np.array(surface2.get('normal', []))
        point1 = np.array(surface1.get('point', []))
        point2 = np.array(surface2.get('point', []))
        
        if len(normal1) == 3 and len(normal2) == 3:
            # 计算法向量夹角
            cos_angle = np.dot(normal1, normal2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(abs(cos_angle))
            
            # 计算平面距离（点2到平面1的距离）
            d1 = surface1.get('d', 0)
            distance = abs(np.dot(point2, normal1) + d1)
            
            geometry_similar = (angle < plane_normal_threshold and distance < plane_distance_threshold)
            geometry_info = (f"法向量夹角={np.degrees(angle):.2f}°(阈值={np.degrees(plane_normal_threshold):.2f}°), "
                           f"平面距离={distance:.4f}(阈值={plane_distance_threshold:.4f})")
    
    elif type1 == 'cylindrical_surface':
        # 柱面相似度判断
        top1 = np.array(surface1.get('top_center', []))
        bottom1 = np.array(surface1.get('bottom_center', []))
        radius1 = surface1.get('radius', 0)
        
        top2 = np.array(surface2.get('top_center', []))
        bottom2 = np.array(surface2.get('bottom_center', []))
        radius2 = surface2.get('radius', 0)
        
        if len(top1) == 3 and len(bottom1) == 3 and len(top2) == 3 and len(bottom2) == 3:
            # 计算轴线方向
            axis1 = top1 - bottom1
            axis2 = top2 - bottom2
            axis1_len = np.linalg.norm(axis1)
            axis2_len = np.linalg.norm(axis2)
            
            if axis1_len > 1e-6 and axis2_len > 1e-6:
                axis1 = axis1 / axis1_len
                axis2 = axis2 / axis2_len
                
                # 计算轴线夹角
                cos_angle = np.dot(axis1, axis2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(abs(cos_angle))
                
                # 计算半径差
                radius_diff = abs(radius1 - radius2)
                
                # 计算位置距离（轴中点的距离）
                center1 = (top1 + bottom1) / 2
                center2 = (top2 + bottom2) / 2
                center_distance = np.linalg.norm(center1 - center2)
                
                geometry_similar = (angle < cylinder_axis_threshold and 
                                   radius_diff < cylinder_radius_threshold and
                                   center_distance < cylinder_position_threshold)
                geometry_info = (f"轴线夹角={np.degrees(angle):.2f}°(阈值={np.degrees(cylinder_axis_threshold):.2f}°), "
                               f"半径差={radius_diff:.4f}(阈值={cylinder_radius_threshold:.4f}), "
                               f"中心距离={center_distance:.4f}(阈值={cylinder_position_threshold:.4f})")
    
    debug_lines.append(f"几何参数: {geometry_info}, 相似={geometry_similar}")
    
    return geometry_similar, "; ".join(debug_lines)


def are_surfaces_similar(result1: Dict, result2: Dict,
                        plane_normal_threshold: float = 0.1,
                        plane_distance_threshold: float = 0.05,
                        cylinder_axis_threshold: float = 0.1,
                        cylinder_radius_threshold: float = 0.02,
                        cylinder_position_threshold: float = 0.05) -> bool:
    """
    判断两个面是否相似（仅判断几何参数：位置和法向）
    
    Args:
        result1, result2: 拟合结果字典
        plane_normal_threshold: 平面法向量夹角阈值（弧度）
        plane_distance_threshold: 平面距离阈值
        cylinder_axis_threshold: 柱面轴线夹角阈值（弧度）
        cylinder_radius_threshold: 柱面半径差阈值
        cylinder_position_threshold: 柱面位置距离阈值
        
    Returns:
        是否相似（几何参数相似）
    """
    surface1 = result1.get('surface', {})
    surface2 = result2.get('surface', {})
    type1 = surface1.get('type', result1.get('type', ''))
    type2 = surface2.get('type', result2.get('type', ''))
    
    # 不同类型的面不相似
    if type1 != type2:
        return False
    
    # 检查几何参数相似性（位置和法向）
    if type1 == 'plane':
        # 平面相似度判断
        normal1 = np.array(surface1.get('normal', []))
        normal2 = np.array(surface2.get('normal', []))
        point1 = np.array(surface1.get('point', []))
        point2 = np.array(surface2.get('point', []))
        
        if len(normal1) == 3 and len(normal2) == 3:
            # 计算法向量夹角
            cos_angle = np.dot(normal1, normal2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(abs(cos_angle))
            
            # 计算平面距离（点2到平面1的距离）
            d1 = surface1.get('d', 0)
            distance = abs(np.dot(point2, normal1) + d1)
            
            return (angle < plane_normal_threshold and distance < plane_distance_threshold)
    
    elif type1 == 'cylindrical_surface':
        # 柱面相似度判断
        top1 = np.array(surface1.get('top_center', []))
        bottom1 = np.array(surface1.get('bottom_center', []))
        radius1 = surface1.get('radius', 0)
        
        top2 = np.array(surface2.get('top_center', []))
        bottom2 = np.array(surface2.get('bottom_center', []))
        radius2 = surface2.get('radius', 0)
        
        if len(top1) == 3 and len(bottom1) == 3 and len(top2) == 3 and len(bottom2) == 3:
            # 计算轴线方向
            axis1 = top1 - bottom1
            axis2 = top2 - bottom2
            axis1_len = np.linalg.norm(axis1)
            axis2_len = np.linalg.norm(axis2)
            
            if axis1_len > 1e-6 and axis2_len > 1e-6:
                axis1 = axis1 / axis1_len
                axis2 = axis2 / axis2_len
                
                # 计算轴线夹角
                cos_angle = np.dot(axis1, axis2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(abs(cos_angle))
                
                # 计算半径差
                radius_diff = abs(radius1 - radius2)
                
                # 计算位置距离（轴中点的距离）
                center1 = (top1 + bottom1) / 2
                center2 = (top2 + bottom2) / 2
                center_distance = np.linalg.norm(center1 - center2)
                
                return (angle < cylinder_axis_threshold and 
                       radius_diff < cylinder_radius_threshold and
                       center_distance < cylinder_position_threshold)
    
    return False


def check_points_on_circle_face(face_center: np.ndarray, face_normal: np.ndarray, 
                                radius: float, points: np.ndarray, 
                                threshold: float = 0.01,
                                min_points: int = 10,
                                partition_of_circle: int = 12,
                                min_partitions_with_points: int = 8,
                                min_points_per_partition: int = 2) -> bool:
    """
    检查圆面上是否分布满点（不仅要有點，还要分布均匀）
    
    Args:
        face_center: 圆面中心点 [3]
        face_normal: 圆面法向量 [3]（归一化）
        radius: 圆面半径
        points: 要检查的点 [N, 3]
        threshold: 距离阈值
        min_points: 圆面上最少点数
        partition_of_circle: 圆周分区数
        min_partitions_with_points: 最少要有多少个分区有足够的点
        min_points_per_partition: 每个分区最少的点数
        
    Returns:
        是否在圆面上分布满点（点足够多且分布均匀）
    """
    if len(points) == 0:
        return False
    
    # 构建平面坐标系
    if abs(face_normal[2]) < 0.9:
        u = np.array([0, 0, 1])
    else:
        u = np.array([1, 0, 0])
    u = u - np.dot(u, face_normal) * face_normal
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(face_normal, u)
    v = v / (np.linalg.norm(v) + 1e-10)
    
    # 计算每个点到圆面的距离
    vec_to_points = points - face_center
    distances_to_plane = np.abs(np.dot(vec_to_points, face_normal))
    
    # 找到在圆面附近的点（距离圆面很近）
    on_plane_mask = distances_to_plane < threshold
    
    if not np.any(on_plane_mask):
        return False
    
    # 检查这些点是否在圆内
    points_on_plane = points[on_plane_mask]
    vec_on_plane = points_on_plane - face_center
    # 投影到圆面平面（移除法向量方向的分量）
    dot_products = np.dot(vec_on_plane, face_normal)
    vec_on_plane = vec_on_plane - dot_products[:, np.newaxis] * face_normal
    distances_from_center = np.linalg.norm(vec_on_plane, axis=1)
    
    # 检查是否在圆内（半径范围内）
    in_circle_mask = distances_from_center <= radius + threshold
    
    if not np.any(in_circle_mask):
        return False
    
    # 获取圆内的点
    points_in_circle = points_on_plane[in_circle_mask]
    
    # 检查点数是否足够
    if len(points_in_circle) < min_points:
        return False
    
    # 检查点的分布是否均匀
    # 投影到2D平面
    points_2d = []
    for pt in points_in_circle:
        vec = pt - face_center
        dot = np.dot(vec, face_normal)
        vec_on_plane = vec - dot * face_normal
        u_coord = np.dot(vec_on_plane, u)
        v_coord = np.dot(vec_on_plane, v)
        points_2d.append([u_coord, v_coord])
    
    points_2d = np.array(points_2d)
    
    # 计算每个点的角度
    angles = np.degrees(np.arctan2(points_2d[:, 1], points_2d[:, 0])) % 360
    
    # 将圆分成多个扇区
    sector_angle = 360 / partition_of_circle
    votes = np.zeros(partition_of_circle, dtype=np.int32)
    
    # 计算每个点属于哪个扇区
    sector_indices = ((angles - 1) // sector_angle).astype(np.int32) % partition_of_circle
    sector_indices = np.clip(sector_indices, 0, partition_of_circle - 1)
    
    # 统计每个扇区的点数
    unique, counts = np.unique(sector_indices, return_counts=True)
    unique = unique.astype(np.int32)
    unique = np.clip(unique, 0, partition_of_circle - 1)
    
    valid_mask = (unique >= 0) & (unique < partition_of_circle)
    if np.any(valid_mask):
        votes[unique[valid_mask]] += counts[valid_mask]
    
    # 检查有多少个扇区有足够的点
    partitions_with_enough_points = np.sum(votes >= min_points_per_partition)
    
    # 只有当足够多的扇区都有足够的点时，才认为分布满点
    return partitions_with_enough_points >= min_partitions_with_points


def get_plane_cylinder_intersection(plane: Dict, cylinder: Dict, 
                                   threshold: float = 0.01) -> Optional[Dict]:
    """
    计算平面与柱面的交线（圆形）
    
    Args:
        plane: 平面结果字典
        cylinder: 柱面结果字典
        threshold: 距离阈值
        
    Returns:
        交线信息字典：{'center': [3], 'normal': [3], 'radius': float} 或 None
    """
    plane_surface = plane.get('surface', {})
    cylinder_surface = cylinder.get('surface', {})
    
    if plane_surface.get('type') != 'plane' or cylinder_surface.get('type') != 'cylindrical_surface':
        return None
    
    # 获取平面参数
    plane_normal = np.array(plane_surface.get('normal', []))
    plane_point = np.array(plane_surface.get('point', []))
    
    # 获取柱面参数
    top_center = np.array(cylinder_surface.get('top_center', []))
    bottom_center = np.array(cylinder_surface.get('bottom_center', []))
    radius = cylinder_surface.get('radius', 0)
    
    if len(plane_normal) != 3 or len(plane_point) != 3:
        return None
    if len(top_center) != 3 or len(bottom_center) != 3:
        return None
    
    # 计算柱面轴线方向
    axis = top_center - bottom_center
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        return None
    axis = axis / axis_len
    
    # 计算平面与轴线的交点
    # 平面方程：n·(p - p0) = 0
    # 直线方程：p = bottom + t * axis
    # 代入：n·(bottom + t*axis - p0) = 0
    # n·bottom + t*(n·axis) - n·p0 = 0
    # t = (n·p0 - n·bottom) / (n·axis)
    
    denom = np.dot(plane_normal, axis)
    if abs(denom) < 1e-6:
        # 平面与轴线平行，可能无交线或交线为矩形
        return None
    
    t = (np.dot(plane_normal, plane_point) - np.dot(plane_normal, bottom_center)) / denom
    
    # 检查交点是否在柱面高度范围内
    if t < -threshold or t > axis_len + threshold:
        return None
    
    # 交点
    intersection_point = bottom_center + t * axis
    
    # 计算交线半径（投影到平面上的圆半径）
    # 从交点到柱面边缘的距离
    # 柱面在垂直于轴线的平面上的投影是圆
    # 平面与柱面的交线也是圆，半径等于柱面半径
    
    return {
        'center': intersection_point,
        'normal': plane_normal.copy(),
        'radius': radius
    }


def remove_points_in_circle(boundary: np.ndarray, circle_center: np.ndarray,
                           circle_normal: np.ndarray, circle_radius: float,
                           threshold: float = 0.01) -> np.ndarray:
    """
    从边界点中移除圆形区域内的点
    
    Args:
        boundary: 边界点 [N, 3]
        circle_center: 圆心 [3]
        circle_normal: 圆面法向量 [3]（归一化）
        circle_radius: 圆半径
        threshold: 距离阈值
        
    Returns:
        移除后的边界点 [M, 3]
    """
    if len(boundary) == 0:
        return boundary
    
    # 计算每个点到圆面的距离
    vec_to_points = boundary - circle_center
    distances_to_plane = np.abs(np.dot(vec_to_points, circle_normal))
    
    # 找到不在圆面附近的点（这些点不在交线上）
    not_on_plane_mask = distances_to_plane >= threshold
    
    # 对于在圆面附近的点，检查是否在圆内
    points_on_plane = boundary[~not_on_plane_mask]
    if len(points_on_plane) > 0:
        vec_on_plane = points_on_plane - circle_center
        # 投影到圆面平面（移除法向量方向的分量）
        dot_products = np.dot(vec_on_plane, circle_normal)
        vec_on_plane = vec_on_plane - dot_products[:, np.newaxis] * circle_normal
        distances_from_center = np.linalg.norm(vec_on_plane, axis=1)
        
        # 不在圆内的点保留
        outside_circle_mask = distances_from_center > circle_radius + threshold
        not_on_plane_mask[~not_on_plane_mask] = outside_circle_mask
    
    return boundary[not_on_plane_mask]


def remove_circle_from_boundary(boundary: np.ndarray, circle_center: np.ndarray,
                                circle_normal: np.ndarray, circle_radius: float,
                                threshold: float = 0.01) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    从边界中移除圆形区域，生成新的边界和洞
    
    Args:
        boundary: 边界点 [N, 3]
        circle_center: 圆心 [3]
        circle_normal: 圆面法向量 [3]（归一化）
        circle_radius: 圆半径
        threshold: 距离阈值
        
    Returns:
        (移除后的边界点 [M, 3], 洞的边界点 [K, 3] 或 None)
    """
    if len(boundary) == 0:
        return boundary, None
    
    # 构建平面坐标系
    if abs(circle_normal[2]) < 0.9:
        u = np.array([0, 0, 1])
    else:
        u = np.array([1, 0, 0])
    u = u - np.dot(u, circle_normal) * circle_normal
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(circle_normal, u)
    v = v / (np.linalg.norm(v) + 1e-10)
    
    # 将边界点投影到圆面平面
    vec_to_boundary = boundary - circle_center
    distances_to_plane = np.abs(np.dot(vec_to_boundary, circle_normal))
    
    # 只处理在圆面附近的点
    on_plane_mask = distances_to_plane < threshold
    
    if not np.any(on_plane_mask):
        # 如果边界点都不在圆面附近，不需要移除
        return boundary, None
    
    # 投影边界点到2D平面
    boundary_2d_on_plane = []
    boundary_3d_on_plane = []
    boundary_3d_outside = []
    
    for i, pt in enumerate(boundary):
        if on_plane_mask[i]:
            vec = pt - circle_center
            # 投影到圆面
            dot = np.dot(vec, circle_normal)
            vec_on_plane = vec - dot * circle_normal
            # 转换为2D坐标
            u_coord = np.dot(vec_on_plane, u)
            v_coord = np.dot(vec_on_plane, v)
            boundary_2d_on_plane.append([u_coord, v_coord])
            boundary_3d_on_plane.append(pt)
        else:
            boundary_3d_outside.append(pt)
    
    if len(boundary_2d_on_plane) == 0:
        return boundary, None
    
    boundary_2d_on_plane = np.array(boundary_2d_on_plane)
    boundary_3d_on_plane = np.array(boundary_3d_on_plane)
    
    # 计算2D点到圆心的距离
    distances_2d = np.linalg.norm(boundary_2d_on_plane, axis=1)
    
    # 找到在圆外的点
    outside_circle_mask = distances_2d > circle_radius + threshold
    
    # 保留圆外的点
    kept_3d = boundary_3d_on_plane[outside_circle_mask].tolist()
    
    # 添加圆外的边界点
    kept_3d.extend(boundary_3d_outside)
    
    # 如果在圆边缘附近有边界点被移除，添加切角点
    # 找到被移除点和保留点的边界
    removed_mask = ~outside_circle_mask
    if np.any(removed_mask) and np.any(outside_circle_mask):
        # 找到圆边缘上的点作为切角点
        # 计算圆边缘与边界多边形的交点
        edge_radius = circle_radius + threshold * 0.5
        
        # 对于每个被移除的区域，在圆边缘添加采样点
        num_samples = 16  # 在圆边缘采样16个点
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        circle_edge_2d = edge_radius * np.column_stack([np.cos(angles), np.sin(angles)])
        
        # 转换回3D
        circle_edge_3d = []
        for pt_2d in circle_edge_2d:
            pt_3d = circle_center + pt_2d[0] * u + pt_2d[1] * v
            circle_edge_3d.append(pt_3d)
        
        # 检查圆边缘点是否在边界多边形内（简化：检查是否接近边界）
        # 如果圆边缘点靠近某个保留的边界点，则添加它
        kept_2d = boundary_2d_on_plane[outside_circle_mask]
        for pt_2d in circle_edge_2d:
            # 检查是否靠近保留的边界
            distances_to_kept = np.linalg.norm(kept_2d - pt_2d, axis=1)
            if np.any(distances_to_kept < edge_radius * 0.3):
                pt_3d = circle_center + pt_2d[0] * u + pt_2d[1] * v
                kept_3d.append(pt_3d)
    
    if len(kept_3d) < 3:
        # 如果移除后点数太少，返回原始边界
        print(f"      警告：移除后边界点数过少({len(kept_3d)})，保留原始边界")
        return boundary, None
    
    # 生成圆形洞的边界点
    num_hole_samples = 32  # 洞边界采样点数
    angles = np.linspace(0, 2 * np.pi, num_hole_samples, endpoint=False)
    hole_2d = circle_radius * np.column_stack([np.cos(angles), np.sin(angles)])
    hole_3d = []
    for pt_2d in hole_2d:
        pt_3d = circle_center + pt_2d[0] * u + pt_2d[1] * v
        hole_3d.append(pt_3d)
    
    return np.array(kept_3d), np.array(hole_3d)


def apply_cylinder_boolean_operations(results: List[Dict], 
                                     label_points: Dict[int, np.ndarray]) -> List[Dict]:
    """
    对柱面和其他面进行布尔操作
    如果柱面的顶面或底面上没有点，则剔除与柱面相交的其他面的圆面部分
    
    Args:
        results: 拟合结果列表
        label_points: 按label分组的点云
        
    Returns:
        处理后的结果列表
    """
    updated_results = []
    
    # 找到所有柱面
    cylinders = []
    other_surfaces = []
    
    for i, result in enumerate(results):
        surface_type = result.get('surface', {}).get('type', result.get('type', ''))
        if surface_type == 'cylindrical_surface':
            cylinders.append((i, result))
        else:
            other_surfaces.append((i, result))
    
    print(f"\n{'='*60}")
    print(f"柱面布尔操作:")
    print(f"{'='*60}")
    print(f"找到 {len(cylinders)} 个柱面，{len(other_surfaces)} 个其他面")
    
    # 对每个柱面进行检查
    for cyl_idx, cylinder_result in cylinders:
        cylinder_surface = cylinder_result.get('surface', {})
        top_center = np.array(cylinder_surface.get('top_center', []))
        bottom_center = np.array(cylinder_surface.get('bottom_center', []))
        radius = cylinder_surface.get('radius', 0)
        
        if len(top_center) != 3 or len(bottom_center) != 3:
            continue
        
        # 计算柱面轴线方向
        axis = top_center - bottom_center
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-6:
            continue
        axis_normalized = axis / axis_len
        
        # 检查顶面和底面上是否有其他面的点
        top_has_points = False
        bottom_has_points = False
        
        # 收集所有其他面的点
        all_other_points = []
        for other_idx, other_result in other_surfaces:
            other_label = other_result.get('label')
            if other_label in label_points:
                all_other_points.append(label_points[other_label])
        
        if len(all_other_points) > 0:
            all_other_points = np.vstack(all_other_points)
            
            # 检查顶面
            top_has_points = check_points_on_circle_face(
                top_center, axis_normalized, radius, all_other_points, threshold=0.01
            )
            
            # 检查底面（底面法向量相反）
            bottom_has_points = check_points_on_circle_face(
                bottom_center, -axis_normalized, radius, all_other_points, threshold=0.01
            )
        
        print(f"\n柱面 {cyl_idx+1}:")
        print(f"  顶面分布满点: {top_has_points}, 底面分布满点: {bottom_has_points}")
        
        # 如果顶面或底面上没有点，需要剔除相交面的圆面部分
        if not top_has_points or not bottom_has_points:
            # 找到与柱面相交的其他面
            for other_idx, other_result in other_surfaces:
                other_surface = other_result.get('surface', {})
                other_type = other_surface.get('type', other_result.get('type', ''))
                
                if other_type == 'plane':
                    # 计算平面与柱面的交线
                    intersection = get_plane_cylinder_intersection(other_result, cylinder_result)
                    
                    if intersection is not None:
                        # 检查交线是否在柱面的顶面或底面附近
                        intersection_center = intersection['center']
                        top_distance = np.linalg.norm(intersection_center - top_center)
                        bottom_distance = np.linalg.norm(intersection_center - bottom_center)
                        
                        # 判断交线是否在顶面或底面附近
                        height = axis_len
                        near_top = top_distance < height * 0.1  # 距离顶部小于10%高度
                        near_bottom = bottom_distance < height * 0.1  # 距离底部小于10%高度
                        
                        should_remove = False
                        if not top_has_points and near_top:
                            should_remove = True
                        if not bottom_has_points and near_bottom:
                            should_remove = True
                        
                        if should_remove:
                            # 从平面边界中移除圆形区域内的点
                            boundary = np.array(other_surface.get('boundary', []))
                            if len(boundary) > 0:
                                new_boundary, hole_boundary = remove_circle_from_boundary(
                                    boundary,
                                    intersection['center'],
                                    intersection['normal'],
                                    intersection['radius'],
                                    threshold=0.01
                                )
                                
                                if hole_boundary is not None:
                                    # 更新边界和洞（确保更新到原始results中的对象）
                                    other_result['surface']['boundary'] = new_boundary.tolist()
                                    # 保存洞的边界
                                    if 'holes' not in other_result['surface']:
                                        other_result['surface']['holes'] = []
                                    other_result['surface']['holes'].append(hole_boundary.tolist())
                                    print(f"    从面 {other_idx+1} 移除了圆形区域，生成洞")
                                    print(f"    原始边界点数: {len(boundary)}, 新边界点数: {len(new_boundary)}, 洞边界点数: {len(hole_boundary)}")
                                elif len(new_boundary) != len(boundary):
                                    # 只更新边界（没有生成洞）
                                    other_result['surface']['boundary'] = new_boundary.tolist()
                                    print(f"    从面 {other_idx+1} 移除了 {len(boundary) - len(new_boundary)} 个边界点")
                                    print(f"    原始边界点数: {len(boundary)}, 新边界点数: {len(new_boundary)}")
    
    print(f"{'='*60}\n")
    
    return results


def get_color_for_result(result: Dict, index: int) -> List[float]:
    """
    根据结果索引获取颜色
    
    Args:
        result: 拟合结果
        index: 结果索引
        
    Returns:
        颜色 [R, G, B]
    """
    colors = [
        [1.0, 0.0, 0.0],    # 红色
        [0.0, 1.0, 0.0],    # 绿色
        [0.0, 0.0, 1.0],    # 蓝色
        [1.0, 1.0, 0.0],    # 黄色
        [1.0, 0.0, 1.0],    # 洋红
        [0.0, 1.0, 1.0],    # 青色
        [1.0, 0.5, 0.0],    # 橙色
        [0.5, 0.0, 1.0],    # 紫色
    ]
    return colors[index % len(colors)]


def remove_similar_surfaces(results: List[Dict],
                           plane_normal_threshold: float = 0.1,
                           plane_distance_threshold: float = 0.05,
                           cylinder_axis_threshold: float = 0.1,
                           cylinder_radius_threshold: float = 0.02,
                           cylinder_position_threshold: float = 0.05) -> List[Dict]:
    """
    移除相似的面，保留质量最好的（仅判断几何参数：位置和法向）
    
    Args:
        results: 拟合结果列表
        plane_normal_threshold: 平面法向量夹角阈值（弧度）
        plane_distance_threshold: 平面距离阈值
        cylinder_axis_threshold: 柱面轴线夹角阈值（弧度）
        cylinder_radius_threshold: 柱面半径差阈值
        cylinder_position_threshold: 柱面位置距离阈值
        
    Returns:
        去重后的结果列表
    """
    if len(results) <= 1:
        return results
    
    print(f"\n{'='*60}")
    print(f"剔除相近的面:")
    print(f"{'='*60}")
    print(f"原始结果数量: {len(results)}")
    
    # 为所有结果分配颜色并打印
    print(f"\n所有面的颜色信息:")
    for i, result in enumerate(results):
        color = get_color_for_result(result, i)
        label = result.get('label', 'unknown')
        surface_type = result.get('surface', {}).get('type', result.get('type', 'unknown'))
        status = "保留"
        print(f"  结果 {i+1}: label={label}, 类型={surface_type}, "
              f"颜色=RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}), 状态={status}")
    
    kept_results = []
    removed_indices = set()
    
    for i in range(len(results)):
        if i in removed_indices:
            continue
        
        current_result = results[i]
        current_color = get_color_for_result(current_result, i)
        similar_found = False
        
        # 检查是否与已保留的结果相似
        for j, kept_result in enumerate(kept_results):
            kept_color = get_color_for_result(kept_result, j)
            
            # 详细检查相似性（添加调试信息）
            is_similar, debug_info = are_surfaces_similar_with_debug(
                current_result, kept_result,
                plane_normal_threshold=plane_normal_threshold,
                plane_distance_threshold=plane_distance_threshold,
                cylinder_axis_threshold=cylinder_axis_threshold,
                cylinder_radius_threshold=cylinder_radius_threshold,
                cylinder_position_threshold=cylinder_position_threshold
            )
            
            if is_similar:
                # 比较质量，保留更好的
                score1 = current_result.get('error', float('inf')) / (current_result.get('inlier_ratio', 0.01) + 0.01)
                score2 = kept_result.get('error', float('inf')) / (kept_result.get('inlier_ratio', 0.01) + 0.01)
                
                print(f"\n  结果 {i+1} 与结果 {j+1} 相似:")
                print(f"    {debug_info}")
                
                if score1 < score2:
                    # 当前结果更好，替换已保留的结果
                    kept_results[j] = current_result
                    print(f"    结果 {i+1} (颜色RGB({current_color[0]:.2f}, {current_color[1]:.2f}, {current_color[2]:.2f}), "
                          f"质量得分={score1:.4f}) 替换结果 {j+1} "
                          f"(颜色RGB({kept_color[0]:.2f}, {kept_color[1]:.2f}, {kept_color[2]:.2f}), "
                          f"质量得分={score2:.4f})")
                else:
                    # 已保留的结果更好，移除当前结果
                    print(f"    移除结果 {i+1} (颜色RGB({current_color[0]:.2f}, {current_color[1]:.2f}, {current_color[2]:.2f}), "
                          f"质量得分={score1:.4f})，保留结果 {j+1} "
                          f"(颜色RGB({kept_color[0]:.2f}, {kept_color[1]:.2f}, {kept_color[2]:.2f}), "
                          f"质量得分={score2:.4f})")
                
                similar_found = True
                removed_indices.add(i)
                break
        
        if not similar_found:
            # 没有找到相似的结果，保留当前结果
            kept_results.append(current_result)
    
    # 打印最终保留的结果及其颜色
    print(f"\n{'='*60}")
    print(f"去重后结果数量: {len(kept_results)}")
    print(f"移除了 {len(results) - len(kept_results)} 个相似的面")
    print(f"\n最终保留的面及其颜色:")
    
    # 跟踪哪些结果被保留了（记录原始索引）
    kept_original_indices = []
    for kept_result in kept_results:
        # 找到这个结果在原始列表中的索引
        found_idx = None
        for orig_idx, orig_result in enumerate(results):
            if orig_result is kept_result or (
                orig_result.get('label') == kept_result.get('label') and
                orig_result.get('surface', {}).get('type') == kept_result.get('surface', {}).get('type')
            ):
                found_idx = orig_idx
                break
        kept_original_indices.append(found_idx if found_idx is not None else len(kept_original_indices))
    
    for i, (result, orig_idx) in enumerate(zip(kept_results, kept_original_indices)):
        color = get_color_for_result(result, orig_idx)
        label = result.get('label', 'unknown')
        surface_type = result.get('surface', {}).get('type', result.get('type', 'unknown'))
        print(f"  结果 {i+1} (原索引 {orig_idx+1}): label={label}, 类型={surface_type}, "
              f"颜色=RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
    
    # 打印被剔除的结果及其颜色
    if len(removed_indices) > 0:
        print(f"\n被剔除的面及其颜色:")
        for idx in sorted(removed_indices):
            result = results[idx]
            color = get_color_for_result(result, idx)
            label = result.get('label', 'unknown')
            surface_type = result.get('surface', {}).get('type', result.get('type', 'unknown'))
            print(f"  结果 {idx+1}: label={label}, 类型={surface_type}, "
                  f"颜色=RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
    
    print(f"{'='*60}\n")
    
    return kept_results


def process_surface_fitting(pcd_path: str,
                           output_path: str = None,
                           plane_threshold: float = 0.01,
                           cylinder_threshold: float = 0.01,
                           min_points: int = 10,
                           remove_similar: bool = True,
                           plane_normal_threshold: float = 0.1,
                           plane_distance_threshold: float = 0.05,
                           cylinder_axis_threshold: float = 0.1,
                           cylinder_radius_threshold: float = 0.02,
                           cylinder_position_threshold: float = 0.05,
                           visualize: bool = True):
    """
    处理PCD文件，按label分组拟合面
    
    Args:
        pcd_path: PCD文件路径
        output_path: 输出JSON文件路径
        plane_threshold: 平面拟合阈值
        cylinder_threshold: 柱面拟合阈值
        min_points: 每组最少点数
        remove_similar: 是否剔除相似的面
        plane_normal_threshold: 平面法向量夹角阈值（弧度）
        plane_distance_threshold: 平面距离阈值
        cylinder_axis_threshold: 柱面轴线夹角阈值（弧度）
        cylinder_radius_threshold: 柱面半径差阈值
        cylinder_position_threshold: 柱面位置距离阈值
        visualize: 是否可视化结果
    """
    print(f"读取PCD文件: {pcd_path}")
    
    # 读取并按label分组
    label_points = read_pcd_by_label(pcd_path)
    print(f"找到 {len(label_points)} 个不同的label")
    
    results = []
    
    for label, points in label_points.items():
        print(f"\n处理 label {label}: {len(points)} 个点")
        
        if len(points) < min_points:
            print(f"  点数不足，跳过")
            continue
        
        # 拟合面
        result = fit_surface_to_points(
            points,
            plane_threshold=plane_threshold,
            cylinder_threshold=cylinder_threshold,
            min_points=min_points
        )
        
        if result is not None:
            result['label'] = int(label)
            results.append(result)
            print(f"  最佳拟合: {result['type']}, 误差: {result['error']:.6f}, "
                  f"内点数: {result['inlier_count']}/{result['total_points']}")
        else:
            print(f"  拟合失败")
    
    # 打印所有拟合结果及其颜色（去重前）
    if len(results) > 0:
        print(f"\n{'='*60}")
        print(f"去重前所有面的颜色信息:")
        print(f"{'='*60}")
        for i, result in enumerate(results):
            color = get_color_for_result(result, i)
            label = result.get('label', 'unknown')
            surface_type = result.get('surface', {}).get('type', result.get('type', 'unknown'))
            print(f"  结果 {i+1}: label={label}, 类型={surface_type}, "
                  f"颜色=RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
        print(f"{'='*60}\n")
    
    # 剔除相似的面
    if remove_similar and len(results) > 1:
        results = remove_similar_surfaces(
            results,
            plane_normal_threshold=plane_normal_threshold,
            plane_distance_threshold=plane_distance_threshold,
            cylinder_axis_threshold=cylinder_axis_threshold,
            cylinder_radius_threshold=cylinder_radius_threshold,
            cylinder_position_threshold=cylinder_position_threshold
        )
    
    # 保存布尔操作之前的结果（用于导出OBJ）
    results_before_boolean = []
    for result in results:
        # 深拷贝结果，避免布尔操作修改原始数据
        result_copy = copy.deepcopy(result)
        results_before_boolean.append(result_copy)
    
    # 柱面与其他面的布尔操作
    results = apply_cylinder_boolean_operations(results, label_points)
    
    # 保存结果
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")
    
    # 可视化
    if visualize and len(results) > 0:
        visualize_surfaces(label_points, results)
    
    # 返回布尔操作后的结果和布尔操作前的结果
    return results, results_before_boolean


def point_in_polygon_2d(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    判断2D点是否在多边形内（射线法）
    
    Args:
        point: 点 [2]
        polygon: 多边形顶点 [N, 2]，按顺序排列
        
    Returns:
        是否在多边形内
    """
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def triangulate_polygon_2d(polygon: np.ndarray, holes: List[np.ndarray] = None) -> List[List[int]]:
    """
    对2D多边形进行三角剖分（支持非凸多边形和带洞）
    
    Args:
        polygon: 多边形顶点 [N, 2]，按顺序排列
        holes: 洞的顶点列表，每个洞是 [M, 2] 的数组
        
    Returns:
        三角形索引列表 [[i1, i2, i3], ...]
    """
    if len(polygon) < 3:
        return []
    
    # 使用Delaunay三角剖分
    # 将所有点（包括外边界和洞）组合在一起
    all_points = polygon.copy()
    hole_offsets = []
    
    if holes:
        for hole in holes:
            if len(hole) > 0:
                hole_offsets.append(len(all_points))
                all_points = np.vstack([all_points, hole])
    
    if len(all_points) < 3:
        return []
    
    try:
        # 使用Delaunay三角剖分
        tri = Delaunay(all_points)
        triangles = tri.simplices.tolist()
        
        # 过滤三角形：只保留所有顶点都在外多边形内且不在洞内的三角形
        valid_triangles = []
        n_outer = len(polygon)
        
        for tri_indices in triangles:
            # 获取三角形的三个顶点索引
            v1_idx, v2_idx, v3_idx = tri_indices
            
            # 获取三个顶点
            v1 = all_points[v1_idx]
            v2 = all_points[v2_idx]
            v3 = all_points[v3_idx]
            
            # 计算三角形中心
            center = (v1 + v2 + v3) / 3
            
            # 检查中心是否在外多边形内
            if not point_in_polygon_2d(center, polygon):
                continue
            
            # 检查中心是否在洞内
            in_hole = False
            if holes:
                for i, hole in enumerate(holes):
                    if len(hole) > 0 and point_in_polygon_2d(center, hole):
                        in_hole = True
                        break
            
            if not in_hole:
                # 对于带洞的情况，三角形可以使用所有点（包括洞的边界点）
                # 只要三角形的中心不在洞内就可以
                valid_triangles.append(list(tri_indices))
        
        return valid_triangles
    except:
        # 如果Delaunay失败，使用简单的扇形三角剖分（只对第一个洞处理）
        if len(polygon) >= 3:
            triangles = []
            # 简单的扇形三角剖分（从第一个顶点出发）
            for i in range(1, len(polygon) - 1):
                triangles.append([0, i, i + 1])
            return triangles
        return []


def create_plane_mesh(boundary: np.ndarray, color: List[float], 
                     holes: List[np.ndarray] = None) -> o3d.geometry.TriangleMesh:
    """
    创建平面的三角网格（双面填充，支持非凸多边形和带洞）
    
    Args:
        boundary: 边界点 [N, 3]
        color: 颜色 [R, G, B]
        holes: 洞的边界点列表，每个洞是 [M, 3] 的数组
        
    Returns:
        三角网格
    """
    if len(boundary) < 3:
        return None
    
    mesh = o3d.geometry.TriangleMesh()
    
    try:
        # 计算法向量
        p1, p2, p3 = boundary[0], boundary[1], boundary[2]
        normal = np.cross(p2 - p1, p3 - p1)
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        
        # 投影到2D平面进行三角剖分
        # 构建平面坐标系
        if abs(normal[2]) < 0.9:
            u = np.array([0, 0, 1])
        else:
            u = np.array([1, 0, 0])
        u = u - np.dot(u, normal) * normal
        u = u / (np.linalg.norm(u) + 1e-10)
        v = np.cross(normal, u)
        v = v / (np.linalg.norm(v) + 1e-10)
        
        # 投影边界到2D
        boundary_2d = np.array([
            np.dot(boundary - boundary[0], u),
            np.dot(boundary - boundary[0], v)
        ]).T
        
        # 投影洞到2D（如果有）
        holes_2d = None
        if holes:
            holes_2d = []
            for hole in holes:
                if len(hole) > 0:
                    hole_2d = np.array([
                        np.dot(hole - boundary[0], u),
                        np.dot(hole - boundary[0], v)
                    ]).T
                    holes_2d.append(hole_2d)
        
        # 三角剖分
        triangles = triangulate_polygon_2d(boundary_2d, holes_2d)
        is_fallback = False
        
        if len(triangles) == 0:
            # 如果三角剖分失败，使用简单的扇形三角剖分（仅当没有洞时）
            if holes and len(holes) > 0:
                print(f"      警告：带洞的三角剖分失败，边界点数: {len(boundary)}, 洞数: {len(holes)}")
                # 如果有洞但三角剖分失败，尝试不使用洞的简单三角剖分
                if len(boundary) >= 3:
                    triangles = []
                    for i in range(1, len(boundary) - 1):
                        triangles.append([0, i, i + 1])
                    is_fallback = True
                    print(f"      使用简单扇形三角剖分（忽略洞）")
                else:
                    print(f"      边界点数不足({len(boundary)})，无法创建网格")
                    return None
            elif len(boundary) >= 3:
                triangles = []
                for i in range(1, len(boundary) - 1):
                    triangles.append([0, i, i + 1])
            else:
                print(f"      警告：边界点数不足({len(boundary)})，无法创建网格")
                return None
        
        # 根据是否使用fallback来决定顶点
        if is_fallback:
            # fallback模式下，只使用boundary的点（忽略洞）
            all_vertices_3d = boundary
        else:
            # 正常模式：合并所有顶点（边界 + 洞）
            all_vertices = [boundary]
            if holes:
                for hole in holes:
                    if len(hole) > 0:
                        all_vertices.append(hole)
            if len(all_vertices) > 1:
                all_vertices_3d = np.vstack(all_vertices)
            else:
                all_vertices_3d = boundary
        
        # 验证三角形索引是否在有效范围内
        max_idx = len(all_vertices_3d) - 1
        valid_triangles = []
        for tri in triangles:
            if all(0 <= idx <= max_idx for idx in tri):
                valid_triangles.append(tri)
            else:
                print(f"      警告：三角形索引超出范围: {tri}, 最大索引: {max_idx}, 顶点数: {len(all_vertices_3d)}")
        
        if len(valid_triangles) == 0:
            print(f"      警告：没有有效的三角形，无法创建网格")
            return None
        
        mesh.vertices = o3d.utility.Vector3dVector(all_vertices_3d)
        mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
        
        # 双面填充：添加反向的面
        reverse_triangles = [[t[0], t[2], t[1]] for t in valid_triangles]
        all_triangles = valid_triangles + reverse_triangles
        mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
        
        # 设置颜色
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.tile(color, (len(all_vertices_3d), 1))
        )
        
        mesh.compute_vertex_normals()
        return mesh
    except Exception as e:
        print(f"创建平面网格失败: {e}")
        return None


def create_plane_mesh_with_rect_boundary(plane_normal: np.ndarray, plane_point: np.ndarray,
                                        boundary_points: np.ndarray, color: List[float],
                                        scale: float = 1.5) -> o3d.geometry.TriangleMesh:
    """
    创建平面的三角网格，使用矩形边界（忽略原始边界和洞）
    用于展示无边界的面
    
    Args:
        plane_normal: 平面法向量 [3]（归一化）
        plane_point: 平面上的一个点 [3]
        boundary_points: 原始边界点 [N, 3]（用于确定矩形大小）
        color: 颜色 [R, G, B]
        scale: 矩形相对于边界点范围的扩展倍数
        
    Returns:
        三角网格
    """
    if len(boundary_points) < 3:
        return None
    
    mesh = o3d.geometry.TriangleMesh()
    
    try:
        # 构建平面坐标系
        if abs(plane_normal[2]) < 0.9:
            u = np.array([0, 0, 1])
        else:
            u = np.array([1, 0, 0])
        u = u - np.dot(u, plane_normal) * plane_normal
        u = u / (np.linalg.norm(u) + 1e-10)
        v = np.cross(plane_normal, u)
        v = v / (np.linalg.norm(v) + 1e-10)
        
        # 将边界点投影到平面2D坐标系
        vec_to_points = boundary_points - plane_point
        points_2d = np.array([
            np.dot(vec_to_points, u),
            np.dot(vec_to_points, v)
        ]).T
        
        # 计算2D边界框
        min_2d = np.min(points_2d, axis=0)
        max_2d = np.max(points_2d, axis=0)
        center_2d = (min_2d + max_2d) / 2
        size_2d = max_2d - min_2d
        
        # 扩展边界框
        half_size = size_2d / 2.0 * scale
        
        # 创建矩形边界（4个顶点）
        rect_2d = np.array([
            center_2d + np.array([-half_size[0], -half_size[1]]),
            center_2d + np.array([ half_size[0], -half_size[1]]),
            center_2d + np.array([ half_size[0],  half_size[1]]),
            center_2d + np.array([-half_size[0],  half_size[1]])
        ])
        
        # 转换回3D
        rect_3d = []
        for pt_2d in rect_2d:
            pt_3d = plane_point + pt_2d[0] * u + pt_2d[1] * v
            rect_3d.append(pt_3d)
        rect_3d = np.array(rect_3d)
        
        # 创建简单的矩形网格（两个三角形）
        vertices = rect_3d
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        
        # 双面填充：添加反向的面
        reverse_triangles = [[t[0], t[2], t[1]] for t in triangles]
        all_triangles = np.vstack([triangles, reverse_triangles])
        
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
        
        # 设置颜色
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.tile(color, (len(vertices), 1))
        )
        
        mesh.compute_vertex_normals()
        return mesh
    except Exception as e:
        print(f"创建矩形边界平面网格失败: {e}")
        return None


def create_cylinder_mesh(top_center: np.ndarray, bottom_center: np.ndarray, 
                        radius: float, color: List[float]) -> o3d.geometry.TriangleMesh:
    """
    创建柱面的三角网格（双面填充）
    
    Args:
        top_center: 顶部中心点 [3]
        bottom_center: 底部中心点 [3]
        radius: 半径
        color: 颜色 [R, G, B]
        
    Returns:
        三角网格
    """
    direction = top_center - bottom_center
    height = np.linalg.norm(direction)
    
    if height < 1e-6:
        return None
    
    direction = direction / height
    
    # 构建垂直于轴线的坐标系
    if abs(direction[2]) < 0.9:
        u = np.array([0, 0, 1])
    else:
        u = np.array([1, 0, 0])
    u = u - np.dot(u, direction) * direction
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(direction, u)
    v = v / (np.linalg.norm(v) + 1e-10)
    
    # 在圆周和高度方向采样
    num_angles = 32
    num_heights = 10
    
    vertices = []
    angle_samples = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    height_samples = np.linspace(0, height, num_heights)
    
    for h in height_samples:
        center_point = bottom_center + h * direction
        for angle in angle_samples:
            point = center_point + radius * (np.cos(angle) * u + np.sin(angle) * v)
            vertices.append(point)
    
    vertices = np.array(vertices)
    
    # 创建三角网格
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # 生成三角面片索引
    triangles = []
    for i in range(num_heights - 1):
        for j in range(num_angles):
            idx = i * num_angles + j
            next_j = (j + 1) % num_angles
            next_idx = i * num_angles + next_j
            
            # 两个三角形组成一个四边形
            triangles.append([idx, idx + num_angles, next_idx])
            triangles.append([next_idx, idx + num_angles, next_idx + num_angles])
    
    # 双面填充：添加反向的面
    reverse_triangles = [[t[0], t[2], t[1]] for t in triangles]
    all_triangles = triangles + reverse_triangles
    mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    
    # 设置颜色
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile(color, (len(vertices), 1))
    )
    
    mesh.compute_vertex_normals()
    return mesh


def visualize_surfaces(label_points: Dict[int, np.ndarray], results: List[Dict]):
    """
    可视化拟合结果
    
    Args:
        label_points: 按label分组的点云
        results: 拟合结果列表
    """
    geometries = []
    
    # 定义颜色
    colors = [
        [1.0, 0.0, 0.0],    # 红色
        [0.0, 1.0, 0.0],    # 绿色
        [0.0, 0.0, 1.0],    # 蓝色
        [1.0, 1.0, 0.0],    # 黄色
        [1.0, 0.0, 1.0],    # 洋红
        [0.0, 1.0, 1.0],    # 青色
        [1.0, 0.5, 0.0],    # 橙色
        [0.5, 0.0, 1.0],    # 紫色
    ]
    
    # 不添加点云，只显示拟合的面
    
    # 打印可视化时的颜色信息
    print(f"\n{'='*60}")
    print(f"可视化中的面颜色:")
    print(f"{'='*60}")
    
    # 添加拟合的面（填充，不绘制控制点）
    for i, result in enumerate(results):
        surface_data = result['surface']
        surface_type = surface_data.get('type', result.get('type', 'unknown'))
        boundary = np.array(surface_data.get('boundary', []))
        
        if len(boundary) < 3:
            continue
        
        color = colors[result['label'] % len(colors)]
        label = result.get('label', 'unknown')
        print(f"  面 {i+1}: label={label}, 类型={surface_type}, "
              f"颜色=RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
        
        mesh = None
        
        if surface_type == 'plane':
            holes = None
            if 'holes' in surface_data and len(surface_data['holes']) > 0:
                holes = [np.array(hole) for hole in surface_data['holes']]
            mesh = create_plane_mesh(boundary, color, holes=holes)
        elif surface_type == 'cylindrical_surface':
            top_center = np.array(surface_data.get('top_center', []))
            bottom_center = np.array(surface_data.get('bottom_center', []))
            radius = surface_data.get('radius', 0.1)
            if len(top_center) == 3 and len(bottom_center) == 3:
                mesh = create_cylinder_mesh(top_center, bottom_center, radius, color)
        
        if mesh is not None:
            geometries.append(mesh)
        
        # 绘制边界线（可选，用于调试）
        # if len(boundary) > 1:
        #     line_indices = [[j, (j+1) % len(boundary)] for j in range(len(boundary))]
        #     line_set = o3d.geometry.LineSet()
        #     line_set.points = o3d.utility.Vector3dVector(boundary)
        #     line_set.lines = o3d.utility.Vector2iVector(line_indices)
        #     line_set.paint_uniform_color([c * 0.8 for c in color])  # 稍深的颜色
        #     geometries.append(line_set)
    
    print(f"\n可视化: {len(results)} 个拟合结果")
    print("按 'Q' 键退出")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"面拟合结果 - {len(results)} 个面",
        width=1200,
        height=800
    )


if __name__ == "__main__":
    import time
    
    start_time = time.time()
    
    results, results_before_boolean = process_surface_fitting(
        pcd_path="inputs_merge\\00007025.pcd",
        output_path="surface_fitting_results.json",
        plane_threshold=0.01,
        cylinder_threshold=0.01,
        min_points=10,
        remove_similar=True,                    # 是否剔除相似的面
        plane_normal_threshold=0.1,              # 平面法向量夹角阈值（弧度，约5.7度）
        plane_distance_threshold=0.05,           # 平面距离阈值
        cylinder_axis_threshold=0.1,              # 柱面轴线夹角阈值（弧度）
        cylinder_radius_threshold=0.02,            # 柱面半径差阈值
        cylinder_position_threshold=0.05,         # 柱面位置距离阈值
        visualize=True
    )
    
    # 导出不做布尔操作的平面/柱面 OBJ
    print(f"\n{'='*60}")
    print(f"导出不做布尔操作的平面/柱面 OBJ")
    print(f"{'='*60}")
    
    output_dir = "test_without_boolean"
    os.makedirs(output_dir, exist_ok=True)
    
    plane_count = 0
    cylinder_count = 0
    
    for i, result in enumerate(results_before_boolean):
        surface_data = result.get('surface', {})
        surface_type = surface_data.get('type', result.get('type', ''))
        boundary = np.array(surface_data.get('boundary', []))
        
        if len(boundary) < 3:
            continue
        
        color = get_color_for_result(result, i)
        mesh = None
        
        if surface_type == 'plane':
            # 平面：使用矩形边界创建三角网格（不做挖孔，展示无边界的面）
            normal = np.array(surface_data.get('normal', []))
            point = np.array(surface_data.get('point', []))
            
            if len(normal) == 3 and len(point) == 3:
                normal = normal / (np.linalg.norm(normal) + 1e-10)
                # 使用矩形边界，忽略原始边界和洞
                mesh = create_plane_mesh_with_rect_boundary(normal, point, boundary, color, scale=1.5)
                
                if mesh is not None:
                    obj_path = os.path.join(output_dir, f"plane_{plane_count:03d}.obj")
                    o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_normals=False)
                    plane_count += 1
                    print(f"  导出平面 {plane_count}: {obj_path}")
        
        elif surface_type == 'cylindrical_surface':
            # 柱面：使用参数化方式创建三角网格
            top_center = np.array(surface_data.get('top_center', []))
            bottom_center = np.array(surface_data.get('bottom_center', []))
            radius = surface_data.get('radius', 0.1)
            
            if len(top_center) == 3 and len(bottom_center) == 3:
                # 扩展柱面高度（向两端各扩展10%）
                axis = top_center - bottom_center
                axis_len = np.linalg.norm(axis)
                if axis_len > 1e-6:
                    axis_normalized = axis / axis_len
                    extension = axis_len * 0.1  # 向两端各扩展10%
                    extended_top = top_center + extension * axis_normalized
                    extended_bottom = bottom_center - extension * axis_normalized
                else:
                    extended_top = top_center
                    extended_bottom = bottom_center
                
                mesh = create_cylinder_mesh(extended_top, extended_bottom, radius, color)
                
                if mesh is not None:
                    obj_path = os.path.join(output_dir, f"cylinder_{cylinder_count:03d}.obj")
                    o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_normals=False)
                    cylinder_count += 1
                    print(f"  导出柱面 {cylinder_count}: {obj_path}")
    
    # 合并所有面片
    if plane_count > 0 or cylinder_count > 0:
        print(f"\n合并所有面片...")
        all_meshes = []
        
        for i, result in enumerate(results_before_boolean):
            surface_data = result.get('surface', {})
            surface_type = surface_data.get('type', result.get('type', ''))
            boundary = np.array(surface_data.get('boundary', []))
            
            if len(boundary) < 3:
                continue
            
            color = get_color_for_result(result, i)
            mesh = None
            
            if surface_type == 'plane':
                # 平面：使用矩形边界创建三角网格（不做挖孔，展示无边界的面）
                normal = np.array(surface_data.get('normal', []))
                point = np.array(surface_data.get('point', []))
                
                if len(normal) == 3 and len(point) == 3:
                    normal = normal / (np.linalg.norm(normal) + 1e-10)
                    # 使用矩形边界，忽略原始边界和洞
                    mesh = create_plane_mesh_with_rect_boundary(normal, point, boundary, color, scale=1.5)
            elif surface_type == 'cylindrical_surface':
                top_center = np.array(surface_data.get('top_center', []))
                bottom_center = np.array(surface_data.get('bottom_center', []))
                radius = surface_data.get('radius', 0.1)
                if len(top_center) == 3 and len(bottom_center) == 3:
                    # 扩展柱面高度（向两端各扩展10%）
                    axis = top_center - bottom_center
                    axis_len = np.linalg.norm(axis)
                    if axis_len > 1e-6:
                        axis_normalized = axis / axis_len
                        extension = axis_len * 0.1  # 向两端各扩展10%
                        extended_top = top_center + extension * axis_normalized
                        extended_bottom = bottom_center - extension * axis_normalized
                    else:
                        extended_top = top_center
                        extended_bottom = bottom_center
                    
                    mesh = create_cylinder_mesh(extended_top, extended_bottom, radius, color)
            
            if mesh is not None:
                all_meshes.append(mesh)
        
        if len(all_meshes) > 0:
            # 合并所有网格
            combined_vertices = []
            combined_triangles = []
            combined_colors = []
            vertex_offset = 0
            
            for mesh in all_meshes:
                vertices = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles)
                colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
                
                combined_vertices.append(vertices)
                combined_triangles.append(triangles + vertex_offset)
                if colors is not None:
                    combined_colors.append(colors)
                
                vertex_offset += len(vertices)
            
            combined_mesh = o3d.geometry.TriangleMesh()
            combined_mesh.vertices = o3d.utility.Vector3dVector(np.vstack(combined_vertices))
            combined_mesh.triangles = o3d.utility.Vector3iVector(np.vstack(combined_triangles))
            
            if len(combined_colors) > 0:
                combined_mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(combined_colors))
            
            combined_mesh.compute_vertex_normals()
            
            # 保存合并的 OBJ
            combined_obj_path = os.path.join(output_dir, "all_surfaces.obj")
            o3d.io.write_triangle_mesh(combined_obj_path, combined_mesh, write_vertex_normals=False)
            print(f"合并的 OBJ 已保存到: {combined_obj_path}")
    
    print(f"{'='*60}\n")
    
    elapsed_time = time.time() - start_time
    print(f"\n总耗时: {elapsed_time:.2f} 秒")
    print(f"面拟合完成，共 {len(results)} 个拟合结果")

