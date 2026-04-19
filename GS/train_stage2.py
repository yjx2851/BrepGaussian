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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, triplet_loss_for_segmentation, triplet_loss_for_segmentation_matrix, line_constrative_loss,get_loss_weights, line_constrative_loss_matrix
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
from gap_process.gap_gs2point import gs2point
from gap_process.gap_label_filter import cluster_then_filter_then_remap_save
from gap_process.gap_label_merge import merge_labels

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, feature_tag):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians,feature_tag=feature_tag)
    if feature_tag == "stage1":
        gaussians.training_setup(opt)

    
    if checkpoint:
        model_params = torch.load(scene.model_path.replace("_stage2", "_stage1") + "/chkpnt" + str(checkpoint) + ".pth")
        gaussians.restore(model_params, opt, feature_tag)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        # TODO: 获取分割渲染图
        _, segment_image, feature_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["segment"], render_pkg["extra_features"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_edge = viewpoint_cam.original_edge.cuda()  
        gt_mask = viewpoint_cam.original_mask.cuda()

        surface_loss_value = line_constrative_loss_matrix(feature_image, gt_mask, gt_edge)

 
        triplet_loss = triplet_loss_for_segmentation_matrix(
            mask_image=feature_image,
            gt_mask=gt_mask,
            gt_edge=None,
            num_anchors_per_mask=64,  # 每个Mask采样64个锚点（减少计算量）
            margin=0.3,                # 初始margin=0.3，可后续调参
            num_neg_candidates=8      # 每个锚点采样8个负样本候选
        )
        
        # loss = surface_loss_value
        # loss = triplet_loss

        surface_w, triplet_w = get_loss_weights(iteration, opt.iterations)
        loss = surface_w * surface_loss_value + triplet_w * triplet_loss
        #loss = surface_loss_value * 0.3 + triplet_loss * 0.7

        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "mask_loss": f"{ema_loss_for_log:.{5}f}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration,feature_tag=feature_tag)
                # 保存完整checkpoint状态：模型参数 + 优化器状态 + 当前迭代数
                complete_checkpoint = {
                    "model_params": gaussians.capture(),  # 高斯模型核心参数（原保存内容）
                    "optimizer_state": gaussians.optimizer.state_dict(),  # 优化器状态（新增）
                    "current_iteration": iteration  # 当前迭代数（新增，用于后续续训定位）
                }
                torch.save(complete_checkpoint, scene.model_path + "/chkpnt" + str(iteration) + ".pth")  # 保持原文件名，仅修改保存内容


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 3_000, 7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 3_000, 7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = 30000)
    parser.add_argument("--feature_tag", type=str, default = "stage2")
    # parser.add
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.feature_tag == "stage2":
        #args.iterations = 30_000
        args.iterations = 15_000
    else:
        args.iterations = 30_000

    args.model_path = args.model_path + "_" + args.feature_tag
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint,args.feature_tag)

    # All done
    print("\nTraining complete.")

    # --------------------- convert to points ------------------
    checkpoint_path = os.path.join(args.model_path, 'chkpnt15000.pth')
    pcd_path = os.path.join(args.model_path, "point_cloud/iteration_15000/point_cloud.pcd")
    output_pcd = os.path.join(args.model_path, 'densify.pcd')
    num_samples = 4
    circle_threshold_min = 0.3
    circle_threshold_max = 3
    gs2point(checkpoint_path, pcd_path, output_pcd, num_samples, circle_threshold_min, circle_threshold_max)

    eps = 0.08
    min_points = 150
    pcd_path = os.path.join(args.model_path, "densify.pcd")
    out_path = os.path.join(args.model_path, 'filtered.pcd')
    cluster_then_filter_then_remap_save(
                pcd_path=pcd_path,
                out_path=out_path,
                eps=eps,
                min_points=min_points,
                strict_greater=True,
            )

    threshold = 0.5
    min_points = 150
    max_points = 300
    pcd_path = os.path.join(args.model_path, "filtered.pcd")
    out_path = os.path.join(args.model_path, 'merged.pcd')
    merge_labels(
        input_pcd_path=pcd_path,
        output_pcd_path=out_path,
        threshold=threshold,
        max_points=max_points,
        min_points=min_points,
    )

