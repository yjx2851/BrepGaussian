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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.segment_activation = torch.sigmoid
        self.extra_features_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.inverse_segment_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        self._segment = torch.empty(0)
        self._extra_features = torch.empty(0)
        self.extra_feature_degree = 16  # Default feature dimension

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._segment,
        )
    
    def restore(self, model_args, training_args,feature_tag):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._segment) = model_args
        # self.training_setup(training_args) # 之后会覆盖 optimizer，这行代码应该是冗余的
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if feature_tag == "stage2" or feature_tag == "stage2_line":
            #extra_features = torch.zeros((self._xyz.shape[0], self.extra_feature_degree), dtype=torch.float, device="cuda")
            extra_features = torch.randn((self._xyz.shape[0], self.extra_feature_degree), dtype=torch.float, device="cuda")
            self._extra_features = nn.Parameter(extra_features.requires_grad_(True))
            # self.optimizer.load_state_dict(opt_dict)
            self.optimizer = torch.optim.Adam(
                [({'params': [self._extra_features], 'lr': training_args.extra_features_lr, "name": "extra_features"})],
                lr=0.0, eps=1e-15
                )

        # # 只对部分的点加入优化器
        # if feature_tag == "stage2_line":
        #     segment_mask = (self._segment > 0).squeeze()
        #     self._xyz = nn.Parameter(self._xyz.detach()[segment_mask].requires_grad_(True))
        #     self._features_dc = nn.Parameter(self._features_dc.detach()[segment_mask].requires_grad_(True))
        #     self._features_rest = nn.Parameter(self._features_rest.detach()[segment_mask].requires_grad_(True))
        #     self._scaling = nn.Parameter(self._scaling.detach()[segment_mask].requires_grad_(True))
        #     self._rotation = nn.Parameter(self._rotation.detach()[segment_mask].requires_grad_(True))
        #     self._opacity = nn.Parameter(self._opacity.detach()[segment_mask].requires_grad_(True))
        #     self.max_radii2D = nn.Parameter(self.max_radii2D.detach()[segment_mask].requires_grad_(True))
        #     self.xyz_gradient_accum = self.xyz_gradient_accum[segment_mask]
        #     self.denom = nn.Parameter(self.denom.detach()[segment_mask].requires_grad_(True))
        #     self._segment = nn.Parameter(self._segment.detach()[segment_mask].requires_grad_(True))
        #     # self._xyz = self._xyz[segment_mask]
        #     # self._features_dc = self._features_dc[segment_mask]
        #     # self._features_rest = self._features_rest[segment_mask]
        #     # self._scaling = self._scaling[segment_mask]
        #     # self._rotation = self._rotation[segment_mask]
        #     # self._opacity = self._opacity[segment_mask]
        #     # self.max_radii2D = self.max_radii2D[segment_mask]
        #     # self.xyz_gradient_accum = self.xyz_gradient_accum[segment_mask]
        #     # self.denom = self.denom[segment_mask]
        #     # self._segment = self._segment[segment_mask]

        #     extra_features = torch.zeros((self._xyz.shape[0], self.extra_feature_degree), dtype=torch.float, device="cuda")
        #     self._extra_features = nn.Parameter(extra_features.requires_grad_(True))
        #     self.training_setup_for_line(training_args)
        #     # self.optimizer = torch.optim.Adam(
        #     #     [({'params': [self._extra_features], 'lr': training_args.extra_features_lr, "name": "extra_features"})],
        #     #     lr=0.0, eps=1e-15
        #     #     )



    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_segment(self):
        return self.segment_activation(self._segment)
    
    @property
    def get_extra_features(self):
        return self.extra_features_activation(self._extra_features)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float,feature_tag):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        segments=self.inverse_segment_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        # Initialize extra_features with random values
        extra_features = torch.randn((fused_point_cloud.shape[0], self.extra_feature_degree), dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._segment = nn.Parameter(segments.requires_grad_(True))
        if feature_tag == "stage2":
            self._extra_features = nn.Parameter(extra_features.requires_grad_(True))
        else:
            self._extra_features = nn.Parameter(extra_features.requires_grad_(False))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = []
        
        l.extend([
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._segment], 'lr': training_args.segment_lr, "name": "segment"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ])
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
    def training_setup_for_line(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._extra_features = nn.Parameter(torch.zeros((self.get_xyz.shape[0], self.extra_feature_degree), dtype=torch.float, device="cuda").requires_grad_(True))
        l = []
        
        l.extend([
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._segment], 'lr': training_args.segment_lr, "name": "segment"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._extra_features], 'lr': training_args.extra_features_lr, "name": "extra_features"},
        ])
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply_with_segment(self, path):
        """
        保存带segment颜色的PLY文件
        将segment值映射为不同颜色，相近的值使用相近的颜色
        """
        mkdir_p(os.path.dirname(path))
        
        xyz = self._xyz.detach().cpu().numpy()
        segments = self._segment.detach().cpu().numpy()
        
        # 将segment值转换为整数（只保留整数部分）
        segment_values = segments.astype(int).flatten()
        
        # 获取唯一的segment值并排序
        unique_segments = np.unique(segment_values)
        sorted_segments = np.sort(unique_segments)
        num_segments = len(sorted_segments)
        
        print(f"保存segment PLY文件: {path}")
        print(f"唯一segment值数量: {num_segments}")
        print(f"Segment值范围: {sorted_segments.min()} - {sorted_segments.max()}")
        
        # 为每个唯一的segment值分配随机颜色
        color_map = {}
        
        # 设置随机种子，确保结果可重现
        np.random.seed(42)
        
        for seg_val in sorted_segments:
            # 生成随机RGB颜色 (0-255)
            color = np.random.randint(0, 256, 3)
            color_map[seg_val] = color
            print(f"Segment {seg_val} -> RGB {color}")
        
        # 为每个点分配颜色
        colors = np.zeros((len(segment_values), 3), dtype=np.uint8)
        for i, seg_val in enumerate(segment_values):
            colors[i] = color_map[seg_val]
        
        # 创建PLY数据
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        
        elements = np.empty(xyz.shape[0], dtype=dtype)
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1] 
        elements['z'] = xyz[:, 2]
        elements['red'] = colors[:, 0]
        elements['green'] = colors[:, 1]
        elements['blue'] = colors[:, 2]
        
        # 写入PLY文件
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"成功保存 {len(xyz)} 个点到 {path}")
    

    def save_pcd(self, path, similarity_threshold=0.7, min_cluster_size=5, feature_tag=None):
        """
        Save Gaussian model as PCD file with feature-based similarity labels.
        
        Args:
            path: Path to save the PCD file
            similarity_threshold: Cosine similarity threshold for feature grouping (0-1)
            min_cluster_size: Minimum number of points required to form a group
        """
        from utils.pcd_utils import gaussians_to_pcd
        
        print(f"Saving Gaussian model as PCD with feature-based labels...")
        gaussians_to_pcd(
            self, 
            path, 
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
            default_color=(0.5, 0.5, 0.5),
            feature_tag=feature_tag
        )

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
        print(f"Successfully saved {len(xyz)} points to {path}")

    
    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # def load_ply(self, path):
    #     # no storage of segmentations
    #     plydata = PlyData.read(path)

    #     xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
    #                     np.asarray(plydata.elements[0]["y"]),
    #                     np.asarray(plydata.elements[0]["z"])),  axis=1)
    #     opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    #     features_dc = np.zeros((xyz.shape[0], 3, 1))
    #     features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    #     features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    #     features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    #     extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    #     extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    #     assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
    #     features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    #     for idx, attr_name in enumerate(extra_f_names):
    #         features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    #     # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    #     features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

    #     scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    #     scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    #     scales = np.zeros((xyz.shape[0], len(scale_names)))
    #     for idx, attr_name in enumerate(scale_names):
    #         scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    #     rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    #     rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    #     rots = np.zeros((xyz.shape[0], len(rot_names)))
    #     for idx, attr_name in enumerate(rot_names):
    #         rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    #     self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    #     self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    #     self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._segment = optimizable_tensors["segment"]
        # self._extra_features = optimizable_tensors["extra_features"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities,new_segment, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "segment": new_segment,
        # "extra_features": new_extra_features,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._segment = optimizable_tensors["segment"]
        # self._extra_features = optimizable_tensors["extra_features"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_segment = self._segment[selected_pts_mask].repeat(N, 1)
        # new_extra_features = self._extra_features[selected_pts_mask].repeat(N, 1)
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_segment, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_segments = self._segment[selected_pts_mask]
        # new_extra_features = self._extra_features[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_segments, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    # TODO:check
    def load_from_pcd(self, pcd_path):
        """Load Gaussian parameters from a PCD file"""
        import open3d as o3d
        import numpy as np
        
        # Load PCD file
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            raise ValueError(f"No points found in PCD file: {pcd_path}")
        
        # Initialize parameters
        self._xyz = nn.Parameter(torch.tensor(points, dtype=torch.float, device="cuda").requires_grad_(True))
        
        # Get colors if available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            # Convert from [0,1] to [-1,1] range for SH
            colors = colors * 2.0 - 1.0
            self._features_dc = nn.Parameter(torch.tensor(colors, dtype=torch.float, device="cuda").requires_grad_(True))
        else:
            # Default white color
            self._features_dc = nn.Parameter(torch.ones((len(points), 3), dtype=torch.float, device="cuda").requires_grad_(True))
        
        # Initialize other parameters
        self._features_rest = nn.Parameter(torch.zeros((len(points), 3, (self.max_sh_degree + 1) ** 2 - 1), dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.ones((len(points), 1), dtype=torch.float, device="cuda").requires_grad_(True))
        self._segment = nn.Parameter(torch.zeros((len(points), 1), dtype=torch.float, device="cuda").requires_grad_(True))
        self._extra_features = nn.Parameter(torch.ones((len(points), self.extra_feature_degree), dtype=torch.float, device="cuda").requires_grad_(True))
        
        # Initialize scaling and rotation
        scales = torch.ones((len(points), 3), dtype=torch.float, device="cuda")
        self._scaling = nn.Parameter(torch.log(scales).requires_grad_(True))
        
        rots = torch.zeros((len(points), 4), dtype=torch.float, device="cuda")
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        
        # Initialize other attributes
        self.max_radii2D = torch.zeros((len(points)), device="cuda")
        
        # Try to load additional data from PCD if available
        if hasattr(pcd, 'data') and hasattr(pcd.data, 'keys'):
            # Load segment data if available
            if 'segment' in pcd.data:
                segments = np.asarray(pcd.data['segment']).reshape(-1, 1)
                self._segment = nn.Parameter(torch.tensor(segments, dtype=torch.float, device="cuda").requires_grad_(True))
            
            # Load extra features if available
            feature_keys = [key for key in pcd.data.keys() if key.startswith('feature_')]
            if len(feature_keys) > 0:
                feature_values = []
                for key in sorted(feature_keys):
                    feature_values.append(np.asarray(pcd.data[key]))
                extra_features = np.column_stack(feature_values)
                if extra_features.shape[1] == self.extra_feature_degree:
                    self._extra_features = nn.Parameter(torch.tensor(extra_features, dtype=torch.float, device="cuda").requires_grad_(True))
        
        print(f"Loaded {len(points)} points from PCD file: {pcd_path}")