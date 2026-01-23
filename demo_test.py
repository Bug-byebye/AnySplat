#!/usr/bin/env python3
"""
Gradio demo with custom checkpoint loading for AnySplat.
This file is standalone and does not modify existing code.
"""
import gc
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import gradio as gr
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_typed_root_config
from src.global_cfg import set_cfg
from src.misc.image_io import save_interpolated_video
from src.model.model import get_model
from src.model.ply_export import export_ply
from src.utils.image import process_image

_MODEL_CACHE: Dict[str, torch.nn.Module] = {}


def _load_config_from_run_dir(run_dir: Path):
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find config at {cfg_path}")
    cfg_dict = OmegaConf.load(cfg_path)
    return cfg_dict


def _build_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_dir = Path(checkpoint_path).resolve().parent.parent
    cfg_dict = _load_config_from_run_dir(run_dir)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    model = get_model(cfg.model.encoder, cfg.model.decoder)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned_state[k[len("model."):]] = v
    if not cleaned_state:
        cleaned_state = state_dict
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"Warning: missing keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_model_cached(checkpoint_path: str, device: torch.device):
    key = f"{checkpoint_path}|{device}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model = _build_model_from_checkpoint(checkpoint_path, device)
    _MODEL_CACHE[key] = model
    return model


def get_reconstructed_scene(outdir: str, model: torch.nn.Module, device: torch.device):
    image_files = sorted(
        [
            os.path.join(outdir, "images", f)
            for f in os.listdir(os.path.join(outdir, "images"))
        ]
    )
    images = [process_image(img_path) for img_path in image_files]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device)
    b, v, c, h, w = images.shape
    assert c == 3, "Images must have 3 channels"

    with torch.no_grad():
        gaussians, pred_context_pose = model.inference((images + 1) * 0.5)

    pred_all_extrinsic = pred_context_pose["extrinsic"]
    pred_all_intrinsic = pred_context_pose["intrinsic"]
    video, depth_colored = save_interpolated_video(
        pred_all_extrinsic,
        pred_all_intrinsic,
        b,
        h,
        w,
        gaussians,
        outdir,
        model.decoder,
    )

    plyfile = os.path.join(outdir, "gaussians.ply")
    export_ply(
        gaussians.means[0],
        gaussians.scales[0],
        gaussians.rotations[0],
        gaussians.harmonics[0],
        gaussians.opacities[0],
        Path(plyfile),
        save_sh_dc_only=True,
    )

    torch.cuda.empty_cache()
    return plyfile, video, depth_colored


def handle_uploads(input_video, input_images):
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []
    if input_images is not None:
        for file_data in input_images:
            file_path = file_data["name"] if isinstance(file_data, dict) else file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    if input_video is not None:
        video_path = input_video["name"] if isinstance(input_video, dict) else input_video
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps * 1))
        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1
        vs.release()

    image_paths = sorted(image_paths)
    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


def update_gallery_on_upload(input_video, input_images):
    if not input_video and not input_images:
        return None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths


def gradio_demo(target_dir: str, checkpoint_path: str, device_str: str):
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, None, None

    device = torch.device(device_str)
    model = get_model_cached(checkpoint_path, device)

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    plyfile, video, depth_colored = get_reconstructed_scene(target_dir, model, device)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")

    return plyfile, video, depth_colored


def clear_fields():
    return None, None, None


def main():
    default_ckpt = "/home-ldap/sunchang/3dProjects/AnySplat/output/exp_co3d/2026-01-21_15-29-59/checkpoints/epoch_138-step_5000.ckpt"
    device_default = "cuda" if torch.cuda.is_available() else "cpu"

    with gr.Blocks(title="AnySplat Custom Checkpoint Demo") as demo:
        gr.Markdown("# AnySplat Demo (Custom Checkpoint)")

        checkpoint_box = gr.Textbox(label="Checkpoint Path", value=default_ckpt)
        device_radio = gr.Radio(
            choices=["cuda", "cpu"],
            value=device_default,
            label="Device",
            interactive=True,
        )

        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tab("Input Data"):
                    input_video = gr.Video(label="Upload Video", interactive=True)
                    input_images = gr.File(
                        file_count="multiple",
                        label="Upload Images",
                        interactive=True,
                    )
                    image_gallery = gr.Gallery(
                        label="Preview",
                        columns=4,
                        height="300px",
                        object_fit="contain",
                        preview=True,
                    )
            with gr.Column(scale=4):
                with gr.Tab("AnySplat Output"):
                    reconstruction_output = gr.Model3D(
                        label="3D Reconstructed Gaussian Splat",
                        height=540,
                        zoom_speed=0.5,
                        pan_speed=0.5,
                        camera_position=[20, 20, 20],
                    )
                    with gr.Row():
                        rgb_video = gr.Video(label="RGB Video", interactive=False, autoplay=True)
                        depth_video = gr.Video(label="Depth Video", interactive=False, autoplay=True)
                    with gr.Row():
                        submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                        clear_btn = gr.ClearButton(
                            [
                                input_video,
                                input_images,
                                reconstruction_output,
                                target_dir_output,
                                image_gallery,
                                rgb_video,
                                depth_video,
                            ],
                            scale=1,
                        )

        def run_pipeline(target_dir, checkpoint_path, device_str):
            return gradio_demo(target_dir, checkpoint_path, device_str)

        submit_btn.click(
            fn=clear_fields,
            inputs=[],
            outputs=[reconstruction_output, rgb_video, depth_video],
        ).then(
            fn=run_pipeline,
            inputs=[target_dir_output, checkpoint_box, device_radio],
            outputs=[reconstruction_output, rgb_video, depth_video],
        )

        input_video.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images],
            outputs=[reconstruction_output, target_dir_output, image_gallery],
        )
        input_images.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images],
            outputs=[reconstruction_output, target_dir_output, image_gallery],
        )

        demo.queue(max_size=20).launch(show_error=True, share=True)


if __name__ == "__main__":
    main()
