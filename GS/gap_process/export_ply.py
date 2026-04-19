import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def mkdir_p(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class GaussianModel:
    def __init__(self, xyz, features_dc, features_rest, scaling, rotation, opacity, active_sh_degree=0):
        self.active_sh_degree = active_sh_degree
        self._xyz = xyz
        self._features_dc = features_dc
        self._features_rest = features_rest
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity

    def construct_list_of_attributes(self):
        attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        # DC SH coefficients
        dc_cols = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).shape[1]
        attributes += [f'f_dc_{i}' for i in range(dc_cols)]

        # Rest SH coefficients (could be empty)
        rest_cols = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).shape[1]
        attributes += [f'f_rest_{i}' for i in range(rest_cols)]

        # Opacity (assume last dim == 1 after concat)
        attributes += ['opacity'] if self._opacity.shape[1] == 1 else [f'opacity_{i}' for i in range(self._opacity.shape[1])]

        # Scaling (2 or 3 depending on representation)
        attributes += [f'scale_{i}' for i in range(self._scaling.shape[1])]

        # Rotation quaternion (4) or other representation
        attributes += [f'rot_{i}' for i in range(self._rotation.shape[1])]

        return attributes

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


def _load_gaussians_from_checkpoint(ckpt_path):
    data = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Support both tuple (from capture()) and dict-like formats
    if isinstance(data, (list, tuple)):
        # Expected order from capture():
        # (
        #   active_sh_degree, _xyz, _features_dc, _features_rest,
        #   _scaling, _rotation, _opacity, max_radii2D, xyz_gradient_accum,
        #   denom, optimizer_state, spatial_lr_scale, _segment
        # )
        active_sh_degree = int(data[0]) if not isinstance(data[0], torch.Tensor) else int(data[0].item())
        xyz = data[1]
        features_dc = data[2]
        features_rest = data[3]
        scaling = data[4]
        rotation = data[5]
        opacity = data[6]
    elif isinstance(data, dict):
        params = data.get('model_params', data)
        if isinstance(params, (list, tuple)):
            active_sh_degree = int(params[0]) if not isinstance(params[0], torch.Tensor) else int(params[0].item())
            xyz = params[1]
            features_dc = params[2]
            features_rest = params[3]
            scaling = params[4]
            rotation = params[5]
            opacity = params[6]
        else:
            xyz = params.get('_xyz') or params.get('xyz') or params.get('means')
            features_dc = params.get('_features_dc') or params.get('features_dc') or params.get('sh_dc')
            features_rest = params.get('_features_rest') or params.get('features_rest') or torch.zeros_like(features_dc)
            scaling = params.get('_scaling') or params.get('scaling') or params.get('scales')
            rotation = params.get('_rotation') or params.get('rotation') or params.get('quats')
            opacity = params.get('_opacity') or params.get('opacity')
            active_sh_degree = int(params.get('active_sh_degree', 0))
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(data)}")

    # Ensure tensors and shapes
    def to_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x)

    xyz = to_tensor(xyz).float()
    features_dc = to_tensor(features_dc).float()
    features_rest = to_tensor(features_rest).float()
    scaling = to_tensor(scaling).float()
    rotation = to_tensor(rotation).float()
    opacity = to_tensor(opacity).float()

    # Normalize shapes to expected:
    # features_*: (N, C, 3) where C=1 for DC, others for rest
    if features_dc.dim() == 2 and features_dc.shape[1] == 3:
        features_dc = features_dc.unsqueeze(1)
    if features_rest.numel() == 0:
        features_rest = torch.zeros((xyz.shape[0], 0, 3), dtype=features_dc.dtype)
    elif features_rest.dim() == 2 and features_rest.shape[1] == 3:
        features_rest = features_rest.unsqueeze(1)

    # opacity: (N, 1)
    if opacity.dim() == 1:
        opacity = opacity.unsqueeze(1)

    # scaling: (N, S) where S in {2,3}
    if scaling.dim() == 1:
        scaling = scaling.unsqueeze(1)

    # rotation: (N, R) usually 4 (quat)
    if rotation.dim() == 1:
        rotation = rotation.unsqueeze(1)

    return GaussianModel(xyz, features_dc, features_rest, scaling, rotation, opacity, active_sh_degree)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default="./chkpnt15000.pth")
    parser.add_argument("--output_ply", default="./origin.ply")
    args = parser.parse_args()

    ckpt_path = args.checkpoint_path
    out_path = args.output_ply

    ckpt_path = str(Path(ckpt_path))
    out_path = str(Path(out_path))

    print(f"Loading checkpoint: {ckpt_path}")
    model = _load_gaussians_from_checkpoint(ckpt_path)
    print(f"Gaussians: {model._xyz.shape[0]}")

    print(f"Saving PLY: {out_path}")
    model.save_ply(out_path)
    print("Done.")


if __name__ == '__main__':
    main()
