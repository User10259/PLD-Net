# demo.py
import os
import argparse

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from models.PLDNet import PLDNet
from utils.postprocessPLDU import convert_predictions_to_lines, lines_to_vis_and_mask
from utils.postprocessTTPLA import convert_predictions_to_lines_ttpla, lines_to_vis_and_mask_ttpla


def load_model(dataset: str, device: torch.device) -> PLDNet:
    """
    Select the corresponding checkpoint based on the dataset and load PLDNet.
    """
    ckpt_map = {
        "pldu": "checkpoints/bestpldu.pth",
        "ttpla": "checkpoints/bestttpla.pth",
    }
    ckpt_path = ckpt_map[dataset]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = PLDNet(
        num_lines=10,
        pretrained_backbone=False,
        pretrained_path=None,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model

def preprocess_image(img_path: str):
    """
    Read in a single image：
      - Output tensor: [1,3,H,W], value range [0,1]
      - Also return a numpy RGB image for post-processing
    """
    img = Image.open(img_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0)  # [1,3,H,W]
    img_np_rgb = np.array(img)                # (H,W,3), RGB
    return img_tensor, img_np_rgb


def choose_default_image(dataset: str) -> str:
    folder = os.path.join("datasets", dataset)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    candidates = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ]
    if not candidates:
        raise RuntimeError(f"No images found in {folder}")

    candidates.sort()
    return os.path.join(folder, candidates[0])


def run_inference_single(model: PLDNet,
                         img_path: str,
                         dataset: str,
                         device: torch.device,
                         out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Image reading + preprocessing
    img_tensor, img_np_rgb = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)
    H, W = img_np_rgb.shape[:2]

    # 2. forward
    with torch.no_grad():
        out = model(img_tensor)
        pred_cls = out["pred_logits"].squeeze(0).cpu().numpy()   # [G,G,N,2]
        pred_reg = out["pred_lines"].squeeze(0).cpu().numpy()    # [G,G,N,3]

    # 3. Mesh and geometric parameters
    G = 16
    grid_size_x = W / G
    grid_size_y = H / G
    max_d = ((grid_size_x ** 2 + grid_size_y ** 2) ** 0.5) / 2.0

    # 4. Post-processing involving different datasets
    img_bgr = img_np_rgb[..., ::-1].copy()  # BGR 给 OpenCV / 可视化用

    if dataset == "pldu":
        final_lines = convert_predictions_to_lines(
            pred_cls, pred_reg, img_np_rgb,
            grid_size_x, grid_size_y, max_d,
            conf_thresh=0.25, merge=True,
        )
        vis_img, mask_img = lines_to_vis_and_mask(
            final_lines, H, W,
            vis_base=img_bgr,
            render_color=(0, 0, 255),
            render_thickness=3,
            aa=True,
            blur_ksize=9, blur_sigma=1.5,
            otsu=True, fixed_thresh=None,
        )
    else:
        final_lines = convert_predictions_to_lines_ttpla(
            pred_cls, pred_reg, img_np_rgb,
            grid_size_x, grid_size_y, max_d,
            conf_thresh=0.45,
            min_seg_len=3.0,
            join_dist=3.0,
            join_angle_deg=15.0,
        )
        vis_img, mask_img = lines_to_vis_and_mask_ttpla(
            final_lines,
            img_bgr,
            render_color=(0, 0, 255),
            render_thickness=2,
            aa=True,
            blur_ksize=7, blur_sigma=1.0,
            otsu=True, fixed_thresh=None,
        )

    # 5. Save results
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_vis = os.path.join(out_dir, f"{base}_{dataset}_vis.png")

    cv2.imwrite(out_vis, vis_img)

    print(f"[OK] Input : {img_path}")
    print(f"     Vis   : {out_vis}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ttpla",
        choices=["pldu", "ttpla"],
        help="choose which checkpoint & postprocess to use"
    )
    parser.add_argument(
        "--img",
        type=str,
        default='datasets/ttpla/74_00453.jpg',
        help="path to input image; if omitted, will pick one from datasets/<dataset>/"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="directory to save visualization results"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.img is None:
        img_path = choose_default_image(args.dataset)
        print(f"[Info] --img not specified, use sample image: {img_path}")
    else:
        img_path = args.img

    model = load_model(args.dataset, device)
    run_inference_single(model, img_path, args.dataset, device, args.out_dir)


if __name__ == "__main__":
    main()
