#!/usr/bin/env python3

import torch
import numpy as np
from typing import Optional, Tuple
import os

def compute_feature_similarity_labels(features: torch.Tensor,
                                    similarity_threshold: float = 0.8,
                                    min_cluster_size: int = 10,
                                    num_hyperplanes: int = 16,
                                    seed: int = 12345) -> torch.Tensor:
    """
    Compute similarity-based labels for features using cosine LSH (random hyperplane hashing).

    Rationale: Pairwise connectivity with a threshold easily percolates into a single
    connected component. Instead, we hash normalized features by the signs of their
    projections onto random hyperplanes. Nearby directions tend to share the same code.

    Args:
        features: (N, D) tensor. Will be normalized along dim=1 if not already.
        similarity_threshold: Unused in LSH mode (kept for API compatibility).
        min_cluster_size: If >0, optionally collapse very small buckets to their nearest large bucket.
        num_hyperplanes: Number of random hyperplanes (bits) for the hash label.
        seed: RNG seed to make labels deterministic across runs.

    Returns:
        labels: (N,) int64 tensor. Label is the integer code from the binary hash.
    """
    assert features.dim() == 2, "features must be (N, D)"
    N, D = features.shape
    device = features.device

    # Normalize to unit length for cosine-based hashing
    feats = torch.nn.functional.normalize(features, dim=1, eps=1e-8)

    # Create random hyperplanes (D, num_hyperplanes)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    hyperplanes = torch.randn((D, num_hyperplanes), generator=gen, device=device)
    hyperplanes = torch.nn.functional.normalize(hyperplanes, dim=0, eps=1e-8)

    # Project and take sign -> binary code
    proj = feats @ hyperplanes  # (N, num_hyperplanes)
    bits = (proj >= 0).to(torch.int64)  # (N, num_hyperplanes) of {0,1}

    # Pack bits into integer labels
    # labels = sum(bits[:,k] << k)
    powers = (2 ** torch.arange(num_hyperplanes, device=device, dtype=torch.int64))
    labels = (bits * powers).sum(dim=1)

    if min_cluster_size > 0:
        # Optionally merge very small buckets to nearest large bucket centroid
        # Compute bucket sizes
        unique, counts = torch.unique(labels, return_counts=True)
        small_buckets = unique[counts < min_cluster_size]
        if small_buckets.numel() > 0:
            # Compute centroids for large buckets
            large_mask = counts >= min_cluster_size
            large_labels = unique[large_mask]
            if large_labels.numel() > 0:
                # Map from label -> centroid
                # Build index for each large label
                centroids = []
                for lbl in large_labels.tolist():
                    idx = (labels == lbl)
                    centroids.append(feats[idx].mean(dim=0, keepdim=True))
                centroids = torch.cat(centroids, dim=0)  # (L, D)
                centroids = torch.nn.functional.normalize(centroids, dim=1, eps=1e-8)

                # For each small bucket point, assign to nearest large centroid by cosine sim
                for lbl in small_buckets.tolist():
                    idx = (labels == lbl)
                    if idx.any():
                        sims = feats[idx] @ centroids.T  # (n_small, L)
                        nearest = sims.argmax(dim=1)
                        labels[idx] = large_labels[nearest]
            else:
                # All buckets are small; keep as-is
                pass

    return labels

def gaussians_to_pcd_stage1(gaussian_model, 
                           output_path: str) -> None:
    """
    Convert Gaussian Splatting model to PCD file for Stage 1 (basic attributes + segment only).
    
    Args:
        gaussian_model: Trained GaussianModel instance
        output_path: Path to save the PCD file
        default_color: Default RGB color for points (R, G, B) in [0, 1]
    """
    # Extract basic data from Gaussian model
    xyz = gaussian_model.get_xyz.detach().cpu().numpy()  # (N, 3)
    print(f"Converting {len(xyz)} Gaussians to PCD...")
    # Get RGB colors from SH features
    rgb = gaussian_model.get_features[:, 0, :].detach().cpu().numpy()  # (N, 3)
    rgb = np.clip(rgb, 0, 1)  # Ensure valid RGB range
    
    # Get segment as edge attribute
    try:
        edge = gaussian_model.get_segment.detach().cpu().numpy().reshape(-1)
    except Exception:
        edge = np.zeros((xyz.shape[0],), dtype=np.float32)

    
    print(f"Converting {len(xyz)} Gaussians to PCD (Stage 1 - basic attributes only)...")
    
    # Save PCD file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb edge\n")
        f.write("SIZE 4 4 4 4 4\n")
        f.write("TYPE F F F F F\n")
        f.write("COUNT 1 1 1 1 1\n")
        f.write(f"WIDTH {len(xyz)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(xyz)}\n")
        f.write("DATA ascii\n")
        
        # Write point data
        for i in range(len(xyz)):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            rgb_packed = int(r * 255) << 16 | int(g * 255) << 8 | int(b * 255)
            edge_val = edge[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {rgb_packed} {edge_val:.6f}\n")
    
    print(f"Stage 1 PCD saved to: {output_path}")


def gaussians_to_pcd_stage2(gaussian_model, 
                           output_path: str,
                           similarity_threshold: float = 0.8,
                           min_cluster_size: int = 10,
                           default_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                           feature_tag: str = "stage2") -> None:
    """
    Convert Gaussian Splatting model to PCD file for Stage 2 (includes extra_features).
    
    Args:
        gaussian_model: Trained GaussianModel instance
        output_path: Path to save the PCD file
        similarity_threshold: Cosine similarity threshold for feature grouping
        min_cluster_size: Minimum cluster size
        default_color: Default RGB color for points (R, G, B) in [0, 1]
    """
    # Extract data from Gaussian model
    xyz = gaussian_model.get_xyz.detach().cpu().numpy()  # (N, 3)
    features = gaussian_model.get_extra_features.detach().cpu()  # (N, D)
    
    # Get RGB colors from SH features
    rgb = gaussian_model.get_features[:, 0, :].detach().cpu().numpy()  # (N, 3)
    rgb = np.clip(rgb, 0, 1)  # Ensure valid RGB range
    
    # Get segment as edge attribute
    try:
        edge = gaussian_model.get_segment.detach().cpu().numpy().reshape(-1)
    except Exception:
        edge = np.zeros((xyz.shape[0],), dtype=np.float32)

    print(f"Converting {len(xyz)} Gaussians to PCD (Stage 2 - with extra features)...")
    print(f"Feature dimension: {features.shape[1]}")
    
    # Compute similarity-based labels
    print("Computing similarity-based labels...")
    labels = compute_feature_similarity_labels(features, similarity_threshold, min_cluster_size)

    if feature_tag == "stage2_line":
        # 将edge转换为PyTorch张量
        edge_tensor = torch.from_numpy(edge).to(labels.device)
        edge_mask = edge_tensor > 0.5
        # 将非边缘区域的标签设置为0
        labels = torch.where(edge_mask, labels, torch.zeros_like(labels))

    labels_np = labels.cpu().numpy()
    # Count unique labels
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"Found {len(unique_labels)} unique groups:")
    for label, count in zip(unique_labels.cpu().numpy(), counts.cpu().numpy()):
        print(f"  Group {label}: {count} points")
    
    # Create colors based on labels
    colors = np.tile(default_color, (len(xyz), 1))
    
    # Generate colors for different groups
    if len(unique_labels) > 1:
        # Create a simple color palette
        num_colors = len(unique_labels)
        color_palette = np.random.RandomState(42).rand(num_colors, 3)  # Fixed seed for reproducibility
        
        for i, label in enumerate(unique_labels.cpu().numpy()):
            mask = labels_np == label
            colors[mask] = color_palette[i]
    
    # Create PCD file content
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write PCD header with feature dimensions
        feature_dim = features.shape[1]
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        if feature_tag == "stage2_line":
            f.write("FIELDS x y z r g b line edge")
        else:
            f.write("FIELDS x y z r g b label edge")
        for i in range(feature_dim):
            f.write(f" feature_{i}")
        f.write("\n")
        f.write("SIZE 4 4 4 4 4 4 4 4")
        for i in range(feature_dim):
            f.write(" 4")
        f.write("\n")
        f.write("TYPE F F F F F F U F")
        for i in range(feature_dim):
            f.write(" F")
        f.write("\n")
        f.write("COUNT 1 1 1 1 1 1 1 1")
        for i in range(feature_dim):
            f.write(" 1")
        f.write("\n")
        f.write(f"WIDTH {len(xyz)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(xyz)}\n")
        f.write("DATA ascii\n")
        
        # Write point data
        features_np = features.cpu().numpy()
        for i in range(len(xyz)):
            x, y, z = xyz[i]
            r, g, b = colors[i]
            label = labels_np[i]
            edge_val = edge[i]
            feature_vec = features_np[i]
            
            # Write basic attributes
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f} {label} {edge_val:.6f}")
            # Write feature vector
            for j in range(feature_dim):
                f.write(f" {feature_vec[j]:.6f}")
            f.write("\n")
    
    # Save additional files for reference
    labels_path = output_path.replace('.pcd', '_labels.txt')
    features_path = output_path.replace('.pcd', '_features.txt')
    
    with open(labels_path, 'w') as f:
        f.write("Point_ID Label\n")
        for i, label in enumerate(labels_np):
            f.write(f"{i} {label}\n")
    
    with open(features_path, 'w') as f:
        f.write("Point_ID " + " ".join([f"feature_{i}" for i in range(feature_dim)]) + "\n")
        for i, feature_vec in enumerate(features_np):
            f.write(f"{i} " + " ".join([f"{val:.6f}" for val in feature_vec]) + "\n")
    
    print(f"Stage 2 PCD saved to: {output_path}")
    print(f"Labels saved to: {labels_path}")
    print(f"Features saved to: {features_path}")


# Backward compatibility - keep original function name
def gaussians_to_pcd(gaussian_model, 
                    output_path: str,
                    similarity_threshold: float = 0.8,
                    min_cluster_size: int = 10,
                    default_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                    feature_tag: str = None) -> None:
    """
    Backward compatibility function. Determines stage based on output_path.
    """
    if "stage1" in output_path.lower():
        gaussians_to_pcd_stage1(gaussian_model, output_path)
    else:
        gaussians_to_pcd_stage2(gaussian_model, output_path, similarity_threshold, min_cluster_size, default_color,feature_tag=feature_tag)

# Example usage function
def example_usage():
    """Example of how to use the PCD saving functionality."""
    # This would be called from your training script
    # gaussian_model = your_trained_gaussian_model
    # output_path = save_gaussian_pcd_with_labels(
    #     gaussian_model, 
    #     output_dir="./output",
    #     filename="trained_gaussians.pcd",
    #     similarity_threshold=0.7,
    #     min_cluster_size=5
    # )
    pass
