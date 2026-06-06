from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import cv2
import os
import numpy as np
import argparse
import torch

parser = argparse.ArgumentParser()
# parser.add_argument("--input_dir", type=str, default="/data1/yjx/research_data/ABC_modelnet_for_cvpr/multi_view/abc_simple_50/00067592/train_img/", help="input images directory")
parser.add_argument("--input_dir", type=str, default="../00000699/train_img", help="input images directory")
parser.add_argument("--ckpt", type=str, default="./checkpoints/checkpoint_best.pt", help="SAM checkpoint path")
parser.add_argument("--output_dir", type=str, default="../00000699/mask_img", help="output directory for overlays")
parser.add_argument("--gpu", type=int, default=0, help="GPU device id (default: 0)")

# 使用SAM默认参数，避免额外耗时
# 过滤背景相关：过滤掉“几乎整张图”的mask；以及可选过滤太小的mask占比
# parser.add_argument("--max_area_ratio", type=float, default=0.9, help="丢弃面积占比>=该值的近乎整幅背景mask")
# parser.add_argument("--min_area_ratio", type=float, default=0.0, help="丢弃面积占比<该值的极小噪声mask")
args = parser.parse_args()

device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
# sam = sam_model_registry["vit_h"](checkpoint=args.ckpt)
sam = sam_model_registry["vit_h"](checkpoint=None)
ckpt = torch.load(args.ckpt, map_location="cpu")
sam.load_state_dict(ckpt["model_state_dict"], strict=True)

sam.to(device=device)
sam.eval()
mask_generator = SamAutomaticMaskGenerator(sam)

os.makedirs(args.output_dir, exist_ok=True)

for t in os.listdir(args.input_dir):
    img_path = os.path.join(args.input_dir, t)
    image = cv2.imread(img_path)
    if image is None:
        continue

    # edge_path = os.path.join(args.seg_dir, t)
    # edge = Image.open(edge_path).convert("L")
    # edge_np = np.array(edge)
    # edge_mask = edge_np > 0
    # ys, xs = np.where(edge_mask)
    # sel = np.random.choice(len(xs), size=min(10, len(xs)), replace=False)
    # pos_pts = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float32)  # (x,y)
    # pos_lab = np.ones(len(pos_pts), dtype=np.int32)

    masks = mask_generator.generate(image)

    print(img_path, len(masks))        # 分割出来的目标数量
    # print(masks[0].keys())   # 每个mask包含坐标、面积、bbox、二值mask等信息

    stem, _ = os.path.splitext(t)
    # 仅保存叠加图到统一输出目录；背景为白色
    overlay = np.full_like(image, 255, dtype=np.uint8)
    covered = np.zeros(image.shape[:2], dtype=bool)

    rng = np.random.default_rng(123)
    H, W = image.shape[:2]
    total_pixels = float(H * W)
    filtered = []
    for m in masks:
        area = float(m.get("area", 0))
        x, y, w, h = m.get("bbox", [0, 0, 0, 0])
        area_ratio = area / total_pixels if total_pixels > 0 else 0.0
        # 过滤几乎全幅的mask（典型为背景），或bbox几乎覆盖整幅图
        bbox_is_full = (x <= 1 and y <= 1 and w >= W - 2 and h >= H - 2)
        if bbox_is_full:
            continue
        # if area_ratio < args.min_area_ratio:
        #     continue
        filtered.append(m)

    for i, m in enumerate(filtered):
        seg = m["segmentation"].astype(np.uint8)  # 0/1
        color = rng.integers(0, 256, size=(3,), dtype=np.uint8)
        inds = seg.astype(bool)
        overlay[inds] = color
        covered |= inds
    # 显式确保非mask区域为白色
    overlay[~covered] = 0

    cv2.imwrite(os.path.join(args.output_dir, f"{stem}.png"), overlay)
    # cv2.imwrite("overlay.png", overlay)