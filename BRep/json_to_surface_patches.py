import os
import sys
import json
import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial import ConvexHull
from typing import List, Tuple, Dict, Optional

# Import related functions from bool_plane.py if available
sys.path.append(os.path.dirname(__file__))
try:
    from bool_plane import clip_polygon_by_halfplane
    from ransac_surface_fitting import triangulate_polygon_2d
except ImportError:
    print("Warning: failed to import helper functions, falling back to a simplified implementation")


def load_json(json_path: str) -> dict:
    """Load a JSON file and return it as a dict."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def project_point_to_plane(point: np.ndarray, plane_normal: np.ndarray, 
                           plane_point: np.ndarray) -> np.ndarray:
    """Project a 3D point onto a plane."""
    vec = point - plane_point
    dist = np.dot(vec, plane_normal)
    return point - dist * plane_normal


def build_2d_coordinate_system(plane_normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a 2D coordinate system (u, v) on a plane."""
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    
    if abs(plane_normal[2]) < 0.9:
        u = np.array([0, 0, 1])
    else:
        u = np.array([1, 0, 0])
    
    u = u - np.dot(u, plane_normal) * plane_normal
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(plane_normal, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    
    return u, v


def point_3d_to_2d(point_3d: np.ndarray, plane_normal: np.ndarray, 
                   plane_point: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project a 3D point to the plane and convert it to 2D coordinates."""
    proj_3d = project_point_to_plane(point_3d, plane_normal, plane_point)
    vec = proj_3d - plane_point
    return np.array([np.dot(vec, u), np.dot(vec, v)])


def point_2d_to_3d(point_2d: np.ndarray, plane_normal: np.ndarray,
                   plane_point: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Convert 2D plane coordinates back to a 3D point."""
    return plane_point + point_2d[0] * u + point_2d[1] * v


def point_on_line_segment(point: np.ndarray, line_start: np.ndarray, 
                          line_end: np.ndarray, tolerance: float = 0.01) -> Tuple[bool, float]:
    """
    Check whether a point lies on a line segment and return the parameter t (0 <= t <= 1).
    """
    vec_line = line_end - line_start
    vec_to_point = point - line_start
    
    line_len = np.linalg.norm(vec_line)
    if line_len < 1e-10:
        return False, 0.0
    
    # Compute projection parameter t
    t = np.dot(vec_to_point, vec_line) / (line_len * line_len)
    
    # Check whether t is within the segment range
    if t < -tolerance or t > 1.0 + tolerance:
        return False, t
    
    # Compute the distance from the point to the infinite line
    proj_point = line_start + t * vec_line
    dist = np.linalg.norm(point - proj_point)
    
    if dist < tolerance:
        return True, np.clip(t, 0.0, 1.0)
    else:
        return False, t


def point_on_circle_arc(point: np.ndarray, center: np.ndarray, normal: np.ndarray,
                        radius: float, arc_start: np.ndarray, arc_end: np.ndarray,
                        tolerance: float = 0.01) -> Tuple[bool, float]:
    """
    Check whether a point lies on a circular arc and return its angle parameter.
    """
    # Check if the point lies on the circle
    vec_to_point = point - center
    dist_to_center = np.linalg.norm(vec_to_point)
    
    if abs(dist_to_center - radius) > tolerance:
        return False, 0.0
    
    # Check if the point lies on the circle plane
    dist_to_plane = abs(np.dot(vec_to_point, normal))
    if dist_to_plane > tolerance:
        return False, 0.0
    
    # Compute the angle by building a local coordinate system in the circle plane
    if abs(normal[2]) < 0.9:
        u = np.array([0, 0, 1])
    else:
        u = np.array([1, 0, 0])
    u = u - np.dot(u, normal) * normal
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    
    # Project to the plane and compute angle
    vec_proj = vec_to_point - np.dot(vec_to_point, normal) * normal
    x = np.dot(vec_proj, u)
    y = np.dot(vec_proj, v)
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += 2 * np.pi
    
    # Compute start and end angles
    vec_start = arc_start - center
    vec_start_proj = vec_start - np.dot(vec_start, normal) * normal
    x_start = np.dot(vec_start_proj, u)
    y_start = np.dot(vec_start_proj, v)
    angle_start = np.arctan2(y_start, x_start)
    if angle_start < 0:
        angle_start += 2 * np.pi
    
    vec_end = arc_end - center
    vec_end_proj = vec_end - np.dot(vec_end, normal) * normal
    x_end = np.dot(vec_end_proj, u)
    y_end = np.dot(vec_end_proj, v)
    angle_end = np.arctan2(y_end, x_end)
    if angle_end < 0:
        angle_end += 2 * np.pi
    
    # Check whether angle lies within the arc range (including wrap-around across 0)
    if angle_start <= angle_end:
        is_on_arc = angle_start <= angle <= angle_end
    else:
        # Wrap-around across 0 degrees
        is_on_arc = angle >= angle_start or angle <= angle_end
    
    return is_on_arc, angle


def split_line_by_corners(line_start: np.ndarray, line_end: np.ndarray,
                          corners: List[np.ndarray], start_corner_idx: int, end_corner_idx: int,
                          tolerance: float = 0.01) -> Tuple[List[np.ndarray], List[int]]:
    """
    Split a line segment into multiple parts according to intermediate corner points.
    """
    # Collect all corner points lying on the segment, together with their indices
    points_on_line = [(0.0, line_start, start_corner_idx), (1.0, line_end, end_corner_idx)]
    
    for corner_idx, corner in enumerate(corners):
        is_on, t = point_on_line_segment(corner, line_start, line_end, tolerance)
        if is_on and 0.0 < t < 1.0:  # exclude exact endpoints
            points_on_line.append((t, corner, corner_idx))
    
    # Sort by parameter t along the segment
    points_on_line.sort(key=lambda x: x[0])
    
    # Extract point sequence and indices (deduplicate near-duplicates)
    result_points = []
    result_indices = []
    last_point = None
    for t, point, corner_idx in points_on_line:
        if last_point is None or np.linalg.norm(point - last_point) > tolerance * 0.1:
            result_points.append(point)
            result_indices.append(corner_idx)
            last_point = point
    
    return result_points, result_indices


def split_curve_by_corners(center: np.ndarray, normal: np.ndarray, radius: float,
                           corners: List[np.ndarray], num_segments: int = 64,
                           tolerance: float = 0.01) -> List[np.ndarray]:
    """
    Split a circular curve into multiple segments according to corner points.
    """
    # Build a local coordinate system on the circle plane
    if abs(normal[2]) < 0.9:
        u = np.array([0, 0, 1])
    else:
        u = np.array([1, 0, 0])
    u = u - np.dot(u, normal) * normal
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    
    # Find all corner points lying on the circle
    points_on_circle = []
    for corner in corners:
        vec_to_corner = corner - center
        dist_to_center = np.linalg.norm(vec_to_corner)
        dist_to_plane = abs(np.dot(vec_to_corner, normal))
        
        if abs(dist_to_center - radius) < tolerance and dist_to_plane < tolerance:
            # Compute angular position
            vec_proj = vec_to_corner - np.dot(vec_to_corner, normal) * normal
            x = np.dot(vec_proj, u)
            y = np.dot(vec_proj, v)
            angle = np.arctan2(y, x)
            if angle < 0:
                angle += 2 * np.pi
            points_on_circle.append((angle, corner))
    
    # If there are no corners, return a full sampled circle
    if len(points_on_circle) == 0:
        circle_points = []
        for i in range(num_segments):
            angle = i * 2.0 * np.pi / num_segments
            point = center + radius * (np.cos(angle) * u + np.sin(angle) * v)
            circle_points.append(point)
        return circle_points
    
    # Sort by angle
    points_on_circle.sort(key=lambda x: x[0])
    
    # Generate points for each segment between consecutive corners
    result_points = []
    for i in range(len(points_on_circle)):
        angle_start, point_start = points_on_circle[i]
        angle_end, point_end = points_on_circle[(i + 1) % len(points_on_circle)]
        
        # Add segment start point
        result_points.append(point_start)
        
        # Determine number of intermediate samples based on angular span
        if i < len(points_on_circle) - 1:
            angle_diff = angle_end - angle_start
        else:
            # The last segment may wrap around 0 degrees
            angle_diff = (angle_end + 2 * np.pi) - angle_start
        
        num_intermediate = max(2, int(num_segments * angle_diff / (2 * np.pi)))
        
        # Generate intermediate points along the arc
        for j in range(1, num_intermediate):
            t = j / num_intermediate
            if i < len(points_on_circle) - 1:
                angle = angle_start + t * angle_diff
            else:
                angle = (angle_start + t * angle_diff) % (2 * np.pi)
            
            point = center + radius * (np.cos(angle) * u + np.sin(angle) * v)
            result_points.append(point)
    
    return result_points


def collect_boundary_edges_for_surface(surface_idx: int, lines: List[dict], 
                                       curves: List[dict], corners: List[List[float]],
                                       tolerance: float = 0.01) -> List[dict]:
    """
    Collect all boundary edges (lines and curves) belonging to a given surface.
    If a line contains intermediate corner points, it will be split into segments.
    """
    edges = []
    corners_np = [np.array(c) for c in corners]
    
    # Collect line edges
    for line in lines:
        surface_indices = line.get('surface_indices', [])
        if surface_idx in surface_indices:
            start_idx = line.get('start', -1)
            end_idx = line.get('end', -1)
            if 0 <= start_idx < len(corners) and 0 <= end_idx < len(corners):
                p1 = np.array(corners[start_idx])
                p2 = np.array(corners[end_idx])
                
                # Check if the segment contains additional corner points; split if necessary
                points_on_line, corner_indices = split_line_by_corners(
                    p1, p2, corners_np, start_idx, end_idx, tolerance
                )
                
                # Add split segments as individual edges:
                # - if there are only start and end points, a single edge is added
                # - if there are intermediate corners, multiple edges are added
                for i in range(len(points_on_line) - 1):
                    edge_points = [points_on_line[i], points_on_line[i + 1]]
                    # Use corner indices directly
                    start_corner_idx = corner_indices[i]
                    end_corner_idx = corner_indices[i + 1]
                    
                    edges.append({
                        'type': 'line',
                        'points_3d': edge_points,
                        'points_2d': None,  # 稍后投影
                        'start_corner_idx': start_corner_idx,
                        'end_corner_idx': end_corner_idx
                    })
    
    # Collect circular edges (curves)
    # Note: each circle is treated as a single closed loop, not split into multiple edges
    for curve in curves:
        surface_indices = curve.get('surface_indices', [])
        if surface_idx in surface_indices:
            center = np.array(curve.get('center', []))
            normal = np.array(curve.get('normal', []))
            radius = float(curve.get('radius', 0.0))
            
            if center.shape == (3,) and normal.shape == (3,) and radius > 0:
                # Check whether the circle has corner points; if so, split into segments
                points_on_curve = split_curve_by_corners(
                    center, normal, radius, corners_np, 
                    num_segments=max(16, int(2 * np.pi * radius / 0.01)),
                    tolerance=tolerance
                )
                
                # Treat the whole circle as one edge containing all points.
                # The circle itself is already a closed loop and does not need splitting.
                if len(points_on_curve) > 0:
                    # Ensure the circle is closed (first point == last point)
                    if not np.array_equal(points_on_curve[0], points_on_curve[-1]):
                        points_on_curve.append(points_on_curve[0].copy())
                    
                    edges.append({
                        'type': 'circle',
                        'points_3d': points_on_curve,  # full sequence of sampled 3D points
                        'points_2d': None,             # will be set after projection
                        'start_corner_idx': -1,        # closed circle: no explicit endpoints
                        'end_corner_idx': -1,
                        'curve_index': len([e for e in edges if e['type'] == 'circle'])  # index of this circle
                    })
    
    return edges


def connect_edges_into_loops(edges: List[dict], edges_2d: List[np.ndarray]) -> Tuple[List[List[np.ndarray]], List[List[int]], List[int]]:
    """
    Connect boundary edges into closed loops based on corner endpoint indices.
    """
    if len(edges) == 0:
        return [], [], []
    
    # Build a connectivity map based on corner indices:
    # corner_idx -> [edge_indices] (all edges incident to that corner)
    corner_to_edges = {}
    
    for edge_idx, edge in enumerate(edges):
        start_idx = edge.get('start_corner_idx', -1)
        end_idx = edge.get('end_corner_idx', -1)
        
        if start_idx >= 0:
            if start_idx not in corner_to_edges:
                corner_to_edges[start_idx] = []
            corner_to_edges[start_idx].append(edge_idx)
        
        if end_idx >= 0:
            if end_idx not in corner_to_edges:
                corner_to_edges[end_idx] = []
            corner_to_edges[end_idx].append(edge_idx)
    
    visited_edges = set()
    loops = []
    loop_edge_indices = []
    
    def find_all_loops_from_corner(start_corner_idx: int, start_edge_idx: int, 
                                    max_depth: int = 100) -> List[List[int]]:
        found_loops = []
        
        def dfs(current_corner: int, path_edges: List[int], path_corners: List[int],
                visited_in_path: set, visited_corners_in_path: set, depth: int):

            if depth > max_depth:
                return
            
            if current_corner == start_corner_idx and len(path_edges) > 1:
                loop_edges_set = set(path_edges)
                is_duplicate = False
                for existing_loop in found_loops:
                    if set(existing_loop) == loop_edges_set:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    found_loops.append(path_edges.copy())
                return
            
            if current_corner in visited_corners_in_path and current_corner != start_corner_idx:
                first_visit_idx = -1
                for i, corner in enumerate(path_corners):
                    if corner == current_corner:
                        first_visit_idx = i
                        break
                
                if first_visit_idx >= 0:
                    loop_edges = path_edges[first_visit_idx:].copy()
                    
                    if len(loop_edges) > 0:
                        loop_edges_set = set(loop_edges)
                        is_duplicate = False
                        for existing_loop in found_loops:
                            if set(existing_loop) == loop_edges_set:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            found_loops.append(loop_edges)

                return
            
            next_edges = []
            if current_corner in corner_to_edges:
                for edge_idx in corner_to_edges[current_corner]:
                    if edge_idx not in visited_in_path:
                        next_edges.append(edge_idx)
            
            if len(next_edges) == 0:
                return
            
            for next_edge_idx in next_edges:
                next_edge = edges[next_edge_idx]
                next_start = next_edge.get('start_corner_idx', -1)
                next_end = next_edge.get('end_corner_idx', -1)

                if next_start == current_corner and next_end >= 0:
                    next_corner = next_end
                elif next_end == current_corner and next_start >= 0:
                    next_corner = next_start
                else:
                    continue

                path_edges.append(next_edge_idx)
                path_corners.append(current_corner)
                visited_in_path.add(next_edge_idx)
                visited_corners_in_path.add(current_corner)
                dfs(next_corner, path_edges, path_corners, visited_in_path, visited_corners_in_path, depth + 1)

                path_edges.pop()
                path_corners.pop() 
                visited_in_path.remove(next_edge_idx)
                visited_corners_in_path.remove(current_corner)
        
        initial_visited_edges = {start_edge_idx}
        initial_path_edges = [start_edge_idx]
        initial_path_corners = [start_corner_idx]
        initial_visited_corners = {start_corner_idx} 
        
        start_edge = edges[start_edge_idx]
        start_edge_start = start_edge.get('start_corner_idx', -1)
        start_edge_end = start_edge.get('end_corner_idx', -1)
        
        if start_edge_start == start_corner_idx and start_edge_end >= 0:
            initial_corner = start_edge_end
        elif start_edge_end == start_corner_idx and start_edge_start >= 0:
            initial_corner = start_edge_start
        else:
            return found_loops
        
        dfs(initial_corner, initial_path_edges, initial_path_corners, initial_visited_edges, initial_visited_corners, 1)
        
        return found_loops
    
    def build_loop_points(loop_edges: List[int]) -> List[np.ndarray]:

        if len(loop_edges) == 0:
            return []
        
        loop_points_2d = []
        
        first_edge = edges[loop_edges[0]]
        first_start = first_edge.get('start_corner_idx', -1)
        first_end = first_edge.get('end_corner_idx', -1)
        
        if first_start >= 0:
            start_corner = first_start
        else:
            start_corner = first_end
        
        current_corner = start_corner
        
        for e_idx in loop_edges:
            edge_2d = edges_2d[e_idx]
            e = edges[e_idx]
            e_start = e.get('start_corner_idx', -1)
            e_end = e.get('end_corner_idx', -1)
            
            if e_start == current_corner:
                if len(loop_points_2d) == 0:
                    loop_points_2d.extend(edge_2d)
                else:
                    loop_points_2d.extend(edge_2d[1:])
                current_corner = e_end
            elif e_end == current_corner:
                reversed_edge = edge_2d[::-1]
                if len(loop_points_2d) == 0:
                    loop_points_2d.extend(reversed_edge)
                else:
                    loop_points_2d.extend(reversed_edge[1:])
                current_corner = e_start
            else:
                if len(loop_points_2d) == 0:
                    loop_points_2d.extend(edge_2d)
                else:
                    loop_points_2d.extend(edge_2d[1:])
                if e_start >= 0 and e_start != current_corner:
                    current_corner = e_start
                elif e_end >= 0:
                    current_corner = e_end
        
        if len(loop_points_2d) > 0:
            if not np.array_equal(loop_points_2d[0], loop_points_2d[-1]):
                loop_points_2d.append(loop_points_2d[0].copy())
        
        return loop_points_2d
    
    all_found_loops = []
    
    for edge_idx, edge in enumerate(edges):
        start_idx = edge.get('start_corner_idx', -1)
        end_idx = edge.get('end_corner_idx', -1)
        
        if start_idx < 0 or end_idx < 0:
            continue
        
        for start_corner in [start_idx, end_idx]:
            found_loops = find_all_loops_from_corner(start_corner, edge_idx)
            
            for loop_edges in found_loops:
                loop_edges_set = set(loop_edges)
                is_duplicate = False
                for existing_loop in all_found_loops:
                    if set(existing_loop) == loop_edges_set:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_found_loops.append(loop_edges)
                    loop_points_2d = build_loop_points(loop_edges)
                    if len(loop_points_2d) > 0:
                        loops.append(loop_points_2d)
                        loop_edge_indices.append(sorted(loop_edges))
    
    for loop_edges in loop_edge_indices:
        visited_edges.update(loop_edges)
    
    unvisited_edge_indices = [i for i in range(len(edges)) if i not in visited_edges]
    
    return loops, loop_edge_indices, unvisited_edge_indices


def is_loop_inside_other(inner_loop: List[np.ndarray], outer_loop: List[np.ndarray]) -> bool:

    def is_point_inside_loop(point: np.ndarray, loop: List[np.ndarray]) -> bool:
        x, y = point
        inside = False
        for i in range(len(loop)):
            j = (i + 1) % len(loop)
            xi, yi = loop[i]
            xj, yj = loop[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        return inside
    
    # For efficiency, only test a subset of points on inner_loop (at least 3 points)
    test_points = []
    if len(inner_loop) <= 3:
        test_points = inner_loop
    else:
        step = max(1, len(inner_loop) // 3)
        test_points = [inner_loop[i] for i in range(0, len(inner_loop), step)]
        if inner_loop[0] not in test_points:
            test_points.append(inner_loop[0])
    
    # If all sampled points lie inside outer_loop, treat inner_loop as inside it
    all_inside = True
    for point in test_points:
        if not is_point_inside_loop(np.array(point), outer_loop):
            all_inside = False
            break
    
    return all_inside


def classify_loops(loops: List[List[np.ndarray]]) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
    """
    Classify loops into outer boundaries and inner boundaries (holes).
    """
    if len(loops) == 0:
        return [], []
    
    def compute_loop_area(loop: List[np.ndarray]) -> float:
        """Compute polygon area using the shoelace formula."""
        if len(loop) < 3:
            return 0.0
        area = 0.0
        for i in range(len(loop)):
            j = (i + 1) % len(loop)
            area += loop[i][0] * loop[j][1]
            area -= loop[j][0] * loop[i][1]
        return abs(area) / 2.0
    
    # Compute area of each loop
    loop_areas = [(i, compute_loop_area(loop)) for i, loop in enumerate(loops)]
    loop_areas.sort(key=lambda x: x[1], reverse=True)  # sort by area descending
    
    outer_loops = []
    inner_loops = []
    processed_indices = set()
    
    # Process loops starting from the largest area
    for i, area in loop_areas:
        if i in processed_indices:
            continue
        
        loop = loops[i]
        is_inner = False
        
        # Check whether this loop is contained in any already-identified outer loop
        for outer_loop in outer_loops:
            if is_loop_inside_other(loop, outer_loop):
                is_inner = True
                break
        
        if is_inner:
            # This loop is contained by another outer loop: treat as inner boundary (hole)
            inner_loops.append(loop)
            processed_indices.add(i)
        else:
            # This is an outer loop; now check whether it contains other yet-unprocessed loops
            contained_loops = []
            for j, area_j in loop_areas:
                if j <= i or j in processed_indices:
                    continue
                other_loop = loops[j]
                if is_loop_inside_other(other_loop, loop):
                    # Other loop is inside current outer loop: mark as inner boundary
                    inner_loops.append(other_loop)
                    contained_loops.append(j)
                    processed_indices.add(j)
            
            # Mark current loop as an outer boundary
            outer_loops.append(loop)
            processed_indices.add(i)
    
    # Caller can ignore inner_loops if holes are not needed; we keep them to support polygons with holes.
    
    return outer_loops, inner_loops


def merge_outer_loops_with_boolean(outer_loops: List[List[np.ndarray]], 
                                   tolerance: float = 0.01) -> List[np.ndarray]:
    """
    Merge multiple outer loops using polygon boolean operations.
    """
    if len(outer_loops) <= 1:
        return outer_loops
    
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        print("    Warning: shapely is not installed, polygon boolean union is unavailable; returning original loops.")
        return outer_loops
    
    # Convert loops to shapely Polygon objects
    polygons = []
    for loop in outer_loops:
        if len(loop) < 3:
            continue
        try:
            # Remove duplicated first/last point if present
            loop_arr = np.array(loop)
            if len(loop_arr) > 0 and np.array_equal(loop_arr[0], loop_arr[-1]):
                loop_arr = loop_arr[:-1]
            
            if len(loop_arr) < 3:
                continue
            
            # Create a Polygon (outer boundary only, no holes)
            poly = Polygon(loop_arr)
            if poly.is_valid:
                polygons.append(poly)
            else:
                # Try to fix an invalid polygon
                poly = poly.buffer(0)
                if poly.is_valid and isinstance(poly, Polygon):
                    polygons.append(poly)
        except Exception as e:
            print(f"    Warning: failed to create polygon: {e}")
            continue
    
    if len(polygons) == 0:
        return outer_loops

    try:
        merged_geom = unary_union(polygons)
        
        merged_loops = []
        
        if isinstance(merged_geom, Polygon):

            coords = np.array(merged_geom.exterior.coords)
            if len(coords) > 0 and np.array_equal(coords[0], coords[-1]):
                coords = coords[:-1]
            if len(coords) >= 3:
                merged_loops.append(coords.tolist())
        elif hasattr(merged_geom, 'geoms'):

            for geom in merged_geom.geoms:
                if isinstance(geom, Polygon):
                    coords = np.array(geom.exterior.coords)
                    if len(coords) > 0 and np.array_equal(coords[0], coords[-1]):
                        coords = coords[:-1]
                    if len(coords) >= 3:
                        merged_loops.append(coords.tolist())
        
        if len(merged_loops) > 0:
            return merged_loops
        else:
            return outer_loops
            
    except Exception as e:
        print(f"    Warning: polygon boolean union failed: {e}, returning original loops.")
        return outer_loops


def create_polygon_mesh_from_loops(outer_loops: List[List[np.ndarray]], 
                                   inner_loops: List[List[np.ndarray]],
                                   plane_normal: np.ndarray, plane_point: np.ndarray,
                                   u: np.ndarray, v: np.ndarray) -> Optional[o3d.geometry.TriangleMesh]:
    """
    Create a polygon mesh from loops (supports polygons with holes and multiple outer boundaries).
    """
    if len(outer_loops) == 0:
        return None
    
    # If there are multiple outer boundaries, try to merge them
    if len(outer_loops) > 1:
        outer_loops = merge_outer_loops_with_boolean(outer_loops)
    
    # Process each outer boundary (if multiple, process separately and merge meshes later)
    all_meshes = []
    
    for outer_idx, main_loop in enumerate(outer_loops):
        # Find inner loops (holes) that belong to the current outer loop
        current_inner_loops = []
        if len(inner_loops) > 0:
            # Determine which inner loops are contained by this outer loop
            def is_point_inside_loop(point: np.ndarray, loop: List[np.ndarray]) -> bool:
                """Check whether a 2D point lies inside a loop using the ray casting method."""
                x, y = point
                inside = False
                for i in range(len(loop)):
                    j = (i + 1) % len(loop)
                    xi, yi = loop[i]
                    xj, yj = loop[j]
                    if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                        inside = not inside
                return inside
            
            for inner_loop in inner_loops:
                # Take one point from the inner loop and test if it lies inside the current outer loop
                if len(inner_loop) > 0:
                    test_point = inner_loop[0]
                    if is_point_inside_loop(test_point, main_loop):
                        current_inner_loops.append(inner_loop)
        
        # Triangulate the current outer boundary together with its inner holes
        try:
            from ransac_surface_fitting import triangulate_polygon_2d
            
            # Prepare polygon data
            polygon_2d = np.array(main_loop)
            holes_2d = [np.array(loop) for loop in current_inner_loops] if len(current_inner_loops) > 0 else []
            
            # Perform triangulation (use actual boundaries, keep non-convex shapes)
            triangles_2d = triangulate_polygon_2d(polygon_2d, holes=holes_2d)
            
            if len(triangles_2d) == 0:
                # Check polygon validity
                if len(polygon_2d) < 3:
                    continue
                
                # Try a simple fan triangulation (still preserves non-convex polygon, ignores holes)
                try:
                    triangles_2d = []
                    for i in range(1, len(polygon_2d) - 1):
                        triangles_2d.append([0, i, i + 1])
                except Exception as e:
                    # As a last resort, try convex hull (only when triangulation fails completely)
                    try:
                        hull = ConvexHull(polygon_2d)
                        hull_points_2d = polygon_2d[hull.vertices]
                        triangles_2d = []
                        for i in range(1, len(hull_points_2d) - 1):
                            triangles_2d.append([0, i, i + 1])
                        polygon_2d = hull_points_2d
                        holes_2d = []  # ignore holes in convex-hull fallback mode
                    except Exception as e2:
                        continue
            
            # Convert back to 3D
            vertices_3d = []
            for point_2d in polygon_2d:
                vertex_3d = point_2d_to_3d(point_2d, plane_normal, plane_point, u, v)
                vertices_3d.append(vertex_3d)
            
            # Process hole vertices
            vertex_offset = len(vertices_3d)
            for hole in holes_2d:
                for point_2d in hole:
                    vertex_3d = point_2d_to_3d(point_2d, plane_normal, plane_point, u, v)
                    vertices_3d.append(vertex_3d)
            
            # Create mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices_3d))
            mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles_2d, dtype=np.int32))
            mesh.compute_vertex_normals()
            
            # Ensure the mesh normal orientation is consistent with the plane normal
            normals = np.asarray(mesh.vertex_normals)
            avg_normal = normals.mean(axis=0)
            avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)
            
            if np.dot(avg_normal, plane_normal) < 0:
                triangles_array = np.asarray(mesh.triangles)
                mesh.triangles = o3d.utility.Vector3iVector(np.flip(triangles_array, axis=1))
                mesh.compute_vertex_normals()
            
            all_meshes.append(mesh)
            
        except Exception as e:
            continue
    
    # If there are multiple meshes, merge them into one
    if len(all_meshes) == 0:
        return None
    elif len(all_meshes) == 1:
        return all_meshes[0]
    else:
        # Merge multiple meshes using trimesh boolean-like utilities
        try:
            combined_vertices = []
            combined_triangles = []
            vertex_offset = 0
            
            for mesh in all_meshes:
                vertices = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles)
                
                combined_vertices.append(vertices)
                combined_triangles.append(triangles + vertex_offset)
                vertex_offset += len(vertices)
            
            # Convert to trimesh for cleanup / union-like processing
            combined_v = np.vstack(combined_vertices)
            combined_f = np.vstack(combined_triangles)
            
            tm = trimesh.Trimesh(vertices=combined_v, faces=combined_f)
            
            # Try to repair and clean the mesh
            tm.fill_holes()
            tm.remove_duplicate_faces()
            tm.remove_unreferenced_vertices()
            
            # Convert back to Open3D TriangleMesh
            result_mesh = o3d.geometry.TriangleMesh()
            result_mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
            result_mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
            result_mesh.compute_vertex_normals()
            
            return result_mesh
            
        except Exception as e:
            return all_meshes[0]


def compute_cylinder_circle_intersection_height(angle: float, axis_normalized: np.ndarray, 
                                                bottom_center: np.ndarray, cylinder_radius: float,
                                                u: np.ndarray, v: np.ndarray, 
                                                circle: dict, tolerance: float = 0.01) -> Optional[float]:
    """
    Compute the intersection height between a cylinder and a single circle at a given angle.
    """
    circle_center = np.array(circle.get('center', []), dtype=float)
    circle_normal = np.array(circle.get('normal', []), dtype=float)
    circle_radius = float(circle.get('radius', 0.0))
    
    if circle_center.shape != (3,) or circle_normal.shape != (3,) or circle_radius <= 0:
        return None
    
    # Circle plane equation: n·(p - c) = 0
    # Line on the cylinder at this angle: p = bottom + h*axis + r*radial_dir
    # where h is height along the axis, and r is cylinder_radius.
    # Substitute into the plane equation:
    # n·(bottom + h*axis + r*radial_dir - c) = 0
    # n·bottom + h*(n·axis) + r*(n·radial_dir) - n·c = 0
    # h = (n·c - n·bottom - r*(n·radial_dir)) / (n·axis)
    
    n = circle_normal / (np.linalg.norm(circle_normal) + 1e-10)
    denom = np.dot(n, axis_normalized)
    
    if abs(denom) < 1e-6:
        # Plane parallel to axis: no intersection
        return None
    
    # Radial direction at this angle
    radial_dir = np.cos(angle) * u + np.sin(angle) * v
    
    # Compute intersection height
    h = (np.dot(n, circle_center) - np.dot(n, bottom_center) - 
         cylinder_radius * np.dot(n, radial_dir)) / denom
    
    # Validate that the intersection point actually lies on the circle
    # Compute the intersection point on the cylinder
    intersection_point = bottom_center + h * axis_normalized + cylinder_radius * radial_dir
    
    # Distance from intersection point to circle center
    dist_to_circle_center = np.linalg.norm(intersection_point - circle_center)
    
    # Check if this distance matches the circle radius (within tolerance)
    if abs(dist_to_circle_center - circle_radius) < tolerance:
        return h
    
    return None


def create_cylinder_mesh(cylinder_top: np.ndarray, cylinder_bottom: np.ndarray, 
                        cylinder_radius: float, resolution: int = 128, 
                        height_segments: int = 32,
                        circles: List[dict] = None) -> o3d.geometry.TriangleMesh:
    """
    Create a cylinder mesh as a visually continuous high-resolution surface.
    Circle boundaries, if provided, determine the exact vertical range at each angle.
    
    Args:
        cylinder_top: Top center of the cylinder [3].
        cylinder_bottom: Bottom center of the cylinder [3].
        cylinder_radius: Cylinder radius.
        resolution: Number of samples along the circumference (default 128).
        height_segments: Number of segments along the height (default 32).
        circles: List of circle boundary dicts used to determine valid height range per angle.
        
    Returns:
        o3d.geometry.TriangleMesh: High-resolution cylinder mesh.
    """
    axis = cylinder_top - cylinder_bottom
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        return None
    
    axis_normalized = axis / axis_len
    center = (cylinder_top + cylinder_bottom) / 2
    
    # Build an orthonormal coordinate frame perpendicular to the axis
    if abs(axis_normalized[2]) < 0.9:
        u = np.array([0, 0, 1])
    else:
        u = np.array([1, 0, 0])
    u = u - np.dot(u, axis_normalized) * axis_normalized
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(axis_normalized, u)
    v = v / (np.linalg.norm(v) + 1e-10)
    
    # If no circles are provided, use a uniform height range
    use_uniform_height = (circles is None or len(circles) == 0)
    
    if use_uniform_height:
        # Generate cylinder vertices over a uniform height
        vertices = []
        for h_idx in range(height_segments + 1):
            t = h_idx / height_segments
            current_center = cylinder_bottom + t * axis
            for i in range(resolution):
                angle = i * 2.0 * np.pi / resolution
                radial_dir = np.cos(angle) * u + np.sin(angle) * v
                vertex = current_center + cylinder_radius * radial_dir
                vertices.append(vertex)
    else:
        # Use circle boundaries to determine valid height range for each angle
        # First compute intersection heights with all circles for each angular sample
        tolerance = 0.01
        angle_heights = {}  # map angle index -> sorted list of intersection heights
        
        # For each angle position, compute intersection heights with all circles
        for i in range(resolution):
            angle = i * 2.0 * np.pi / resolution
            intersection_heights = []
            for circle in circles:
                h = compute_cylinder_circle_intersection_height(
                    angle, axis_normalized, cylinder_bottom, cylinder_radius, u, v, circle, tolerance
                )
                if h is not None:
                    intersection_heights.append(h)
            angle_heights[i] = sorted(intersection_heights) if intersection_heights else []
        
        # Determine a global height range (used for parameterization when no local range exists)
        all_heights = []
        for heights in angle_heights.values():
            all_heights.extend(heights)
        
        if len(all_heights) > 0:
            global_min_h = min(all_heights)
            global_max_h = max(all_heights)
            margin = (global_max_h - global_min_h) * 0.01 if global_max_h > global_min_h else 0.01
            global_min_h -= margin
            global_max_h += margin
        else:
            global_min_h = 0.0
            global_max_h = axis_len
        
        # Generate vertices: for each angle, sample over its valid height range.
        # To maintain continuity, we use a global parameter t in [0,1] and map
        # it to a local height range at each angle.
        vertices = []
        
        # For each height layer (globally parameterized)
        for h_idx in range(height_segments + 1):
            t = h_idx / height_segments  # 0..1 unified parameter
            
            for i in range(resolution):
                angle = i * 2.0 * np.pi / resolution
                radial_dir = np.cos(angle) * u + np.sin(angle) * v
                
                # Get intersection heights for this angle
                heights = angle_heights[i]
                
                if len(heights) > 0:
                    # Local valid height range at this angle
                    min_h = min(heights)
                    max_h = max(heights)
                    
                    # Map global parameter t into local range to keep continuity between angles
                    local_h = min_h + t * (max_h - min_h)
                else:
                    # No intersection at this angle: fall back to global range
                    local_h = global_min_h + t * (global_max_h - global_min_h)
                
                current_center = cylinder_bottom + local_h * axis_normalized
                vertex = current_center + cylinder_radius * radial_dir
                vertices.append(vertex)
    
    vertices = np.array(vertices)
    
    # Generate triangle faces: each quad is split into two triangles
    triangles = []
    
    for h_idx in range(height_segments):
        for i in range(resolution):
            next_i = (i + 1) % resolution
            
            curr_base = h_idx * resolution
            next_base = (h_idx + 1) * resolution
            
            v0 = curr_base + i
            v1 = curr_base + next_i
            v2 = next_base + next_i
            v3 = next_base + i
            
            triangles.append([v0, v1, v2])
            triangles.append([v0, v2, v3])
    
    triangles = np.array(triangles, dtype=np.int32)
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Compute smooth vertex normals so the surface appears continuous
    mesh.compute_vertex_normals()
    
    return mesh


def point_3d_to_2d_cylinder(pt_3d: np.ndarray, axis: np.ndarray, center: np.ndarray, 
                            radius: float, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Project a 3D point to cylinder coordinates (angle, height).
    """
    # Vector from cylinder center to point
    vec_to_point = pt_3d - center
    # Project onto axis to get height
    height = np.dot(vec_to_point, axis)
    # Project to plane perpendicular to axis
    vec_projected = vec_to_point - height * axis
    # Radial distance from axis
    radial_dist = np.linalg.norm(vec_projected)
    if radial_dist < 1e-10:
        # Point lies on axis; define angle as 0
        angle = 0.0
    else:
        # Compute angle in (u, v) plane
        vec_projected_norm = vec_projected / radial_dist
        # Components along u and v
        u_component = np.dot(vec_projected_norm, u)
        v_component = np.dot(vec_projected_norm, v)
        angle = np.arctan2(v_component, u_component)
        if angle < 0:
            angle += 2 * np.pi
    
    return np.array([angle, height])


def point_2d_to_3d_cylinder(pt_2d: np.ndarray, axis: np.ndarray, center: np.ndarray,
                            radius: float, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Convert cylinder coordinates (angle, height) back to a 3D point.
    """
    angle, height = pt_2d[0], pt_2d[1]
    radial_dir = np.cos(angle) * u + np.sin(angle) * v
    pt_3d = center + height * axis + radius * radial_dir
    return pt_3d


def process_surface(surface: dict, lines: List[dict], curves: List[dict], 
                   corners: List[List[float]], output_dir: str, 
                   surface_idx: int, verbose: bool = True) -> Optional[o3d.geometry.TriangleMesh]:
    """
    处理单个面，生成面片网格
    """
    surf_type = surface.get('type', '')
    
    if surf_type == 'plane':
        normal = np.array(surface.get('normal', []), dtype=float)
        point = np.array(surface.get('point', []), dtype=float)
        
        if normal.shape != (3,) or point.shape != (3,):
            print(f"  警告: 面 {surface_idx} 参数不完整")
            return None
        
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        
        if verbose:
            print(f"\n处理面 {surface_idx} (平面)")
        
        u, v = build_2d_coordinate_system(normal)
        
        edges = collect_boundary_edges_for_surface(surface_idx, lines, curves, corners, tolerance=0.01)
        
        line_edges = [e for e in edges if e['type'] == 'line']
        circle_edges = [e for e in edges if e['type'] == 'circle']
        
        if len(edges) == 0:
            print(f"  警告: 面 {surface_idx} 没有边界边，跳过")
            return None
        
        edges_2d = []
        for edge in edges:
            points_2d = []
            for pt_3d in edge['points_3d']:
                pt_2d = point_3d_to_2d(pt_3d, normal, point, u, v)
                points_2d.append(pt_2d)
            edges_2d.append(np.array(points_2d))
            edge['points_2d'] = points_2d
        
        circle_loops = []
        for circle_edge in circle_edges:
            if len(circle_edge['points_2d']) > 0:

                circle_loop_2d = circle_edge['points_2d'].copy()
                if len(circle_loop_2d) > 0:
                    first_pt = np.array(circle_loop_2d[0])
                    last_pt = np.array(circle_loop_2d[-1])
                    if not np.array_equal(first_pt, last_pt):

                        if isinstance(circle_loop_2d, np.ndarray):
                            circle_loop_2d = np.vstack([circle_loop_2d, first_pt.reshape(1, -1)])
                        else:
                            circle_loop_2d.append(first_pt.tolist() if isinstance(first_pt, np.ndarray) else first_pt)

                    if isinstance(circle_loop_2d, np.ndarray):
                        circle_loops.append(circle_loop_2d.tolist())
                    else:
                        circle_loops.append(circle_loop_2d)
        

        loops = []
        loop_edge_indices = []
        if len(line_edges) > 0:

            line_edges_2d = [e['points_2d'] for e in line_edges]
            line_loops, line_loop_edge_indices, unvisited_edge_indices = connect_edges_into_loops(line_edges, line_edges_2d)
            
            loops = line_loops
            loop_edge_indices = line_loop_edge_indices
            

            if len(unvisited_edge_indices) > 0:
                try:
                    unvisited_points_2d = []
                    for edge_idx in unvisited_edge_indices:
                        unvisited_points_2d.extend(line_edges_2d[edge_idx])
                    
                    if len(unvisited_points_2d) > 2:
                        unvisited_points_2d = np.array(unvisited_points_2d)
                        hull = ConvexHull(unvisited_points_2d)
                        hull_loop = [unvisited_points_2d[idx] for idx in hull.vertices]

                        if len(hull_loop) > 0 and not np.array_equal(hull_loop[0], hull_loop[-1]):
                            hull_loop.append(hull_loop[0].copy())
                        loops.append(hull_loop)
                        loop_edge_indices.append(unvisited_edge_indices)
                except Exception as e:
                    pass
            
            tolerance = 0.01
            for loop_idx, loop in enumerate(loops):
                if len(loop) > 0:
                    first_point = np.array(loop[0])
                    last_point = np.array(loop[-1])
                    
                    is_same_point = len(loop) > 0 and np.array_equal(loop[0], loop[-1])
                    dist = np.linalg.norm(first_point - last_point)
                    is_closed = is_same_point or dist < tolerance * 2
                    
                    if not is_closed:

                        try:
                            loop_array = np.array(loop)
                            hull = ConvexHull(loop_array)
                            loops[loop_idx] = [loop_array[idx] for idx in hull.vertices]
                            if len(loops[loop_idx]) > 0 and not np.array_equal(loops[loop_idx][0], loops[loop_idx][-1]):
                                loops[loop_idx].append(loops[loop_idx][0].copy())
                        except Exception as e:
                            pass
            
            if len(loops) == 0 and len(line_edges) > 0:
                all_line_points_2d = np.vstack(line_edges_2d)
                try:
                    hull = ConvexHull(all_line_points_2d)
                    hull_loop = [all_line_points_2d[idx] for idx in hull.vertices]
                    if len(hull_loop) > 0 and not np.array_equal(hull_loop[0], hull_loop[-1]):
                        hull_loop.append(hull_loop[0].copy())
                    loops = [hull_loop]
                    loop_edge_indices = [list(range(len(line_edges)))]
                except Exception as e:
                    if len(circle_loops) == 0:
                        return None
        
        all_loops = loops + circle_loops
        
        if len(loop_edge_indices) > 0:
            loop_with_edges = [(i, loop, edge_indices) for i, (loop, edge_indices) in enumerate(zip(loops, loop_edge_indices))]
            loop_with_edges.sort(key=lambda x: len(x[2]), reverse=True)
            
            filtered_loops = []
            filtered_edge_indices = []
            
            for loop_idx, loop, edge_indices in loop_with_edges:
                edge_set = set(edge_indices)
                is_subset = False
                
                for kept_edge_set in [set(ei) for ei in filtered_edge_indices]:
                    if edge_set.issubset(kept_edge_set):
                        is_subset = True
                        break
                
                if not is_subset:
                    to_remove = []
                    for i, kept_edge_set in enumerate([set(ei) for ei in filtered_edge_indices]):
                        if kept_edge_set.issubset(edge_set):
                            to_remove.append(i)
                    
                    if len(to_remove) > 0:
                        for i in sorted(to_remove, reverse=True):
                            filtered_loops.pop(i)
                            filtered_edge_indices.pop(i)
                    
                    filtered_loops.append(loop)
                    filtered_edge_indices.append(edge_indices)
            
            loops = filtered_loops
        
        all_loops = loops + circle_loops
        
        outer_loops, inner_loops = classify_loops(all_loops)
        
        mesh = create_polygon_mesh_from_loops(outer_loops, inner_loops, normal, point, u, v)
        
        if mesh is None:
            print(f"  fatal error: failed to create mesh")
            return None
        
        return mesh
    
    # cylindrical surface
    elif surf_type == 'cylindrical_surface' or surf_type == 'cylinder':

        top_center = np.array(surface.get('top_center', []), dtype=float)
        bottom_center = np.array(surface.get('bottom_center', []), dtype=float)
        radius = float(surface.get('radius', 0.0))
        
        if top_center.shape != (3,) or bottom_center.shape != (3,) or radius <= 0:
            print(f"  警告: 面 {surface_idx} 柱面参数不完整")
            return None
        
        axis = top_center - bottom_center
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-6:
            print(f"  警告: 面 {surface_idx} 柱面高度为0，跳过")
            return None
        
        axis_normalized = axis / axis_len
        center = (top_center + bottom_center) / 2
        
        if verbose:
            print(f"\n处理面 {surface_idx} (柱面)")
        
        relevant_circles = []
        for curve in curves:
            surface_indices = curve.get('surface_indices', [])
            if surface_idx in surface_indices:
                circle_center = np.array(curve.get('center', []), dtype=float)
                circle_normal = np.array(curve.get('normal', []), dtype=float)
                circle_radius = float(curve.get('radius', 0.0))
                if circle_center.shape == (3,) and circle_normal.shape == (3,) and circle_radius > 0:
                    relevant_circles.append({
                        'center': circle_center,
                        'normal': circle_normal,
                        'radius': circle_radius
                    })
        
        mesh = create_cylinder_mesh(top_center, bottom_center, radius, 
                                  resolution=128, height_segments=32,
                                  circles=relevant_circles if len(relevant_circles) > 0 else None)
        
        if mesh is None:
            print(f"  fatal error: failed to create mesh")
            return None
        
        return mesh
    
    else:
        print(f"  warning: surface {surface_idx} type '{surf_type}' not supported, skipping")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='generate surface patches from JSON file')
    parser.add_argument('--json', type=str, required=True, help='input JSON file path')
    parser.add_argument('--output', type=str, default='surface_patches_output', help='output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json):
        print(f"fatal error: JSON file does not exist: {args.json}")
        return
    
    data = load_json(args.json)
    
    surfaces = data.get('surfaces', [])
    lines = data.get('lines', [])
    curves = data.get('curves', [])
    corners = data.get('corners', [])
    
    print(f"加载 JSON: {args.json}")
    print(f"找到 {len(surfaces)} 个面, {len(lines)} 条线, {len(curves)} 条曲线, {len(corners)} 个角点")
    
    os.makedirs(args.output, exist_ok=True)
    
    all_meshes = []
    for i, surface in enumerate(surfaces):
        mesh = process_surface(surface, lines, curves, corners, args.output, i)
        if mesh is not None:
            obj_path = os.path.join(args.output, f"surface_{i:03d}.obj")
            o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_normals=False)
            all_meshes.append(mesh)
    
    if len(all_meshes) > 0:
        combined_vertices = []
        combined_triangles = []
        vertex_offset = 0
        
        for mesh in all_meshes:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            
            combined_vertices.append(vertices)
            combined_triangles.append(triangles + vertex_offset)
            vertex_offset += len(vertices)
        
        combined_mesh = o3d.geometry.TriangleMesh()
        combined_mesh.vertices = o3d.utility.Vector3dVector(np.vstack(combined_vertices))
        combined_mesh.triangles = o3d.utility.Vector3iVector(np.vstack(combined_triangles))
        combined_mesh.compute_vertex_normals()
        
        combined_obj_path = os.path.join(args.output, "all_surfaces.obj")
        o3d.io.write_triangle_mesh(combined_obj_path, combined_mesh, write_vertex_normals=False)
        print(f"Combined OBJ saved to: {combined_obj_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

