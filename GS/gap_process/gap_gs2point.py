import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from .export_ply import _load_gaussians_from_checkpoint

try:
    from scipy.spatial import cKDTree  # optional, speeds up NN search
except Exception:
    cKDTree = None


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=q.device, dtype=q.dtype)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
    # Build RS using the provided scaling and rotation, following the user's API
    N = center.shape[0]
    if scaling.dim() == 1:
        scaling = scaling.unsqueeze(1)
    if scaling.shape[1] == 2:
        ones = torch.ones((N, 1), device=scaling.device, dtype=scaling.dtype)
        s_ext = torch.cat([scaling * scaling_modifier, ones], dim=1)
    else:
        s_ext = scaling * scaling_modifier
    RS = build_scaling_rotation(s_ext, rotation).permute(0, 2, 1)
    trans = torch.zeros((center.shape[0], 4, 4), dtype=center.dtype, device=center.device)
    trans[:, :3, :3] = RS
    trans[:, 3, :3] = center
    trans[:, 3, 3] = 1
    return trans

def sample_points_on_ellipse(xyz, scaling, rotation, num_samples=8):
    # Ensure shapes and device
    device = xyz.device
    dtype = xyz.dtype
    N = xyz.shape[0]

    k = int(num_samples)
    angles = torch.arange(k, device=device, dtype=dtype) * (2 * torch.pi / k)
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    # 3D ellipse sampling using transform from covariance_activation
    trans = build_covariance_from_scaling_rotation(xyz, scaling, 1.0, rotation)
    RS = trans[:, :3, :3]              # (N,3,3) = diag(s) @ R^T
    RS_T = RS.transpose(1, 2)          # (N,3,3) = R @ diag(s)
    centers = trans[:, 3, :3]          # (N,3)

    # Columns 0/1 of RS_T are already scaled tangent vectors (su*ru, sv*rv)
    u_scaled = RS_T[:, :, 0]           # (N,3)
    v_scaled = RS_T[:, :, 1]           # (N,3)

    # offsets for all angles
    pts_u = cos_vals.unsqueeze(0).unsqueeze(-1) * u_scaled.unsqueeze(1)   # (N,k,3)
    pts_v = sin_vals.unsqueeze(0).unsqueeze(-1) * v_scaled.unsqueeze(1)   # (N,k,3)
    offsets = pts_u + pts_v

    points = centers.unsqueeze(1) + offsets  # (N,k,3)
    return points


def read_pcd_ascii(pcd_path):
    with open(pcd_path, 'r') as f:
        lines = f.readlines()
    header = []
    data_start = None
    for idx, line in enumerate(lines):
        header.append(line.strip())
        if line.strip().lower().startswith('data '):
            data_start = idx + 1
            data_mode = line.strip().split()[1].lower()
            break
    if data_start is None:
        raise RuntimeError("Invalid PCD: missing DATA line")
    if data_mode != 'ascii':
        raise NotImplementedError("Only ASCII PCD is supported in this script")

    # Parse fields to find indices
    fields_line = next((h for h in header if h.lower().startswith('fields ')), None)
    if fields_line is None:
        raise RuntimeError("Invalid PCD: missing FIELDS line")
    fields = fields_line.split()[1:]
    try:
        ix = fields.index('x')
        iy = fields.index('y')
        iz = fields.index('z')
    except ValueError:
        raise RuntimeError("PCD must contain x y z fields")
    if 'label' in fields:
        ilabel = fields.index('label')
    else:
        raise RuntimeError("PCD must contain a 'label' field")
    if 'edge' in fields:
        iedge = fields.index('edge')
    else:
        raise RuntimeError("PCD must contain an 'edge' field")

    # Load data
    data = np.loadtxt(lines[data_start:])
    if data.ndim == 1:
        data = data[None, :]
    pts = data[:, [ix, iy, iz]].astype(np.float32)
    labels = data[:, ilabel]
    # Cast labels to int if possible
    if np.all(np.isfinite(labels)) and np.all(np.mod(labels, 1) == 0):
        labels = labels.astype(np.int32)
    else:
        labels = labels.astype(np.float32)
    edges = data[:, iedge]
    if np.all(np.isfinite(edges)) and np.all(np.mod(edges, 1) == 0):
        edges = edges.astype(np.int32)
    else:
        edges = edges.astype(np.float32)
    return pts, labels, edges


def write_pcd_ascii(points_xyz, labels, out_path, edges):
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    labels = np.asarray(labels)
    if labels.dtype.kind != 'i':
        # try to make integer if close
        if np.all(np.isfinite(labels)) and np.all(np.mod(labels, 1) == 0):
            labels = labels.astype(np.int32)
        else:
            labels = labels.astype(np.float32)
    edges_arr = np.asarray(edges)
    if edges_arr.dtype.kind != 'i':
        if np.all(np.isfinite(edges_arr)) and np.all(np.mod(edges_arr, 1) == 0):
            edges_arr = edges_arr.astype(np.int32)
        else:
            edges_arr = edges_arr.astype(np.float32)
    n = points_xyz.shape[0]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Header
    is_int_label = (labels.dtype.kind == 'i')
    size_label = 4
    type_label = 'I' if is_int_label else 'F'
    is_int_edge = (edges_arr.dtype.kind == 'i')
    size_edge = 4
    type_edge = 'I' if is_int_edge else 'F'
    with open(out_path, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z label edge\n")
        f.write("SIZE 4 4 4 %d %d\n" % (size_label, size_edge))
        f.write("TYPE F F F %s %s\n" % (type_label, type_edge))
        f.write("COUNT 1 1 1 1 1\n")
        f.write(f"WIDTH {n}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")
        if is_int_label and (edges_arr.dtype.kind == 'i'):
            for (x, y, z), lab, ed in zip(points_xyz, labels, edges_arr):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(lab)} {int(ed)}\n")
        elif is_int_label and (edges_arr.dtype.kind != 'i'):
            for (x, y, z), lab, ed in zip(points_xyz, labels, edges_arr):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(lab)} {float(ed):.6f}\n")
        elif (not is_int_label) and (edges_arr.dtype.kind == 'i'):
            for (x, y, z), lab, ed in zip(points_xyz, labels, edges_arr):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {float(lab):.6f} {int(ed)}\n")
        else:
            for (x, y, z), lab, ed in zip(points_xyz, labels, edges_arr):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {float(lab):.6f} {float(ed):.6f}\n")


def map_labels_by_nearest(centers_xyz_np, pcd_xyz_np, pcd_labels_np):
    centers = centers_xyz_np.astype(np.float32)
    pts = pcd_xyz_np.astype(np.float32)
    if cKDTree is not None:
        tree = cKDTree(pts)
        dists, idxs = tree.query(centers, k=1)
        return pcd_labels_np[idxs]
    # Fallback: chunked brute-force
    M = pts.shape[0]
    max_elems = 2e7  # ~80MB for float32
    B = max(1, int(max_elems // max(1, M)))
    labels_out = np.empty(centers.shape[0], dtype=pcd_labels_np.dtype)
    for start in range(0, centers.shape[0], B):
        end = min(centers.shape[0], start + B)
        block = centers[start:end]  # (B,3)
        # distances
        d2 = np.sum((block[:, None, :] - pts[None, :, :])**2, axis=2)  # (B,M)
        idxs = np.argmin(d2, axis=1)
        labels_out[start:end] = pcd_labels_np[idxs]
    return labels_out

# def process_single_file(checkpoint_path, pcd_path, output_pcd, num_samples=4, 
#                         circle_threshold_min=0.3, circle_threshold_max=3):
def gs2point(checkpoint_path, pcd_path, output_pcd, num_samples=4, 
                        circle_threshold_min=0.3, circle_threshold_max=3):
    """处理单个文件对"""
    ckpt_path = str(Path(checkpoint_path))
    k = int(num_samples)
    if k < 1:
        raise ValueError("num_samples must be >= 1")

    print(f"  Loading checkpoint: {ckpt_path}")
    model = _load_gaussians_from_checkpoint(ckpt_path)

    xyz = model._xyz
    scaling = torch.exp(model._scaling)
    rotation = model._rotation
    features_dc = model._features_dc

    # filter out overly flat Gaussians using circle-like ratio su/sv bounds
    device = scaling.device
    su = scaling[:, 0]
    sv = scaling[:, 1] if scaling.shape[1] >= 2 else scaling[:, 0]
    eps = torch.tensor(1e-12, device=device, dtype=scaling.dtype)
    scale_ratio = su / torch.maximum(sv, eps)
    mask = (scale_ratio >= circle_threshold_min) & (scale_ratio <= circle_threshold_max)

    total = xyz.shape[0]
    kept = int(mask.sum().item())
    skipped = total - kept
    if skipped > 0:
        print(f"  Filtering: circle_threshold_min={circle_threshold_min}, circle_threshold_max={circle_threshold_max}")
        print(f"  Kept {kept}/{total} gaussians, skipped {skipped} as too flat")

    # Keep all gaussians: unmasked -> only center; masked -> center + k boundary
    N = xyz.shape[0]
    centers = xyz.unsqueeze(1)  # (N,1,3)
    # precompute boundary only for masked subset
    masked_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    boundary_masked = None
    if kept > 0 and k > 0:
        boundary_masked = sample_points_on_ellipse(xyz[mask], scaling[mask], rotation[mask], num_samples=k)  # (kept,k,3)

    # colors from DC SH
    if features_dc.dim() == 3:
        colors = features_dc[:, 0, :]
    else:
        colors = features_dc
    C0 = 0.28209479177387814
    colors_rgb = torch.clamp(colors * C0 + 0.5, 0, 1)  # (N,3)

    # map original index -> masked position
    masked_pos = None
    if kept > 0:
        masked_pos = -torch.ones((N,), dtype=torch.long, device=xyz.device)
        masked_pos[masked_indices] = torch.arange(kept, device=xyz.device, dtype=torch.long)

    pts_chunks = []
    col_chunks = []
    for i in range(N):
        # center point
        pts_chunks.append(centers[i])            # (1,3)
        col_chunks.append(colors_rgb[i].unsqueeze(0))  # (1,3)
        # boundary if masked
        if mask[i] and boundary_masked is not None:
            mi = masked_pos[i].item()
            bnd_i = boundary_masked[mi]  # (k,3)
            pts_chunks.append(bnd_i)
            col_chunks.append(colors_rgb[i].unsqueeze(0).expand(k, -1))

    all_points = torch.cat(pts_chunks, dim=0)
    all_colors = torch.cat(col_chunks, dim=0)

    # convert to numpy
    pts_np = all_points.detach().cpu().numpy().astype(np.float32)

    # read PCD labels and transfer to centers via nearest neighbor
    try:
        pcd_xyz, pcd_labels, pcd_edges = read_pcd_ascii(str(Path(pcd_path)))
    except Exception as e:
        print(f"  Failed to read PCD '{pcd_path}': {e}")
        return False

    centers_np = centers.squeeze(1).detach().cpu().numpy().astype(np.float32)  # (N,3)
    gaussian_labels = map_labels_by_nearest(centers_np, pcd_xyz, pcd_labels)   # (N,)
    gaussian_edges = map_labels_by_nearest(centers_np, pcd_xyz, pcd_edges)

    # replicate labels/edges matching assembled points (center always, boundary only if masked)
    labels_list = []
    edges_list = []
    for i in range(N):
        labels_list.append(gaussian_labels[i])
        edges_list.append(gaussian_edges[i])
        if mask[i]:
            labels_list.extend([gaussian_labels[i]] * k)
            edges_list.extend([gaussian_edges[i]] * k)
    labels_rep = np.asarray(labels_list)
    edges_rep = np.asarray(edges_list)

    # write PCD with label per point
    print(f"  Saving labeled PCD to: {output_pcd}")
    write_pcd_ascii(pts_np, labels_rep, str(Path(output_pcd)), edges=edges_rep)
    print(f"  ✓ Saved: {output_pcd}")
    return True


def main():
    import os
    
    # 参数设置
    source_dir = Path('./save_origin')
    output_dir = Path('./save_densify')
    num_samples = 4
    circle_threshold_min = 0.3
    circle_threshold_max = 3
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 .pcd 文件
    pcd_files = sorted(source_dir.glob('*.pcd'))
    
    print(f"Found {len(pcd_files)} PCD files in {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: num_samples={num_samples}, circle_threshold_min={circle_threshold_min}, circle_threshold_max={circle_threshold_max}")
    print("-" * 80)
    
    success_count = 0
    fail_count = 0
    
    for pcd_file in pcd_files:
        # 获取文件ID (去掉扩展名)
        file_id = pcd_file.stem
        
        # 查找对应的 .pth 文件
        pth_file = source_dir / f"{file_id}.pth"
        
        if not pth_file.exists():
            print(f"⚠ Skipping {file_id}: No matching .pth file found")
            fail_count += 1
            continue
        
        # 输出文件路径
        output_pcd = output_dir / f"{file_id}.pcd"
        
        print(f"\n[{success_count + fail_count + 1}/{len(pcd_files)}] Processing: {file_id}")
        
        try:
            success = process_single_file(
                checkpoint_path=pth_file,
                pcd_path=pcd_file,
                output_pcd=output_pcd,
                num_samples=num_samples,
                circle_threshold_min=circle_threshold_min,
                circle_threshold_max=circle_threshold_max
            )
            
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  ✗ Error processing {file_id}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
    
    print("\n" + "=" * 80)
    print(f"Batch processing complete!")
    print(f"Success: {success_count}, Failed: {fail_count}, Total: {len(pcd_files)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
