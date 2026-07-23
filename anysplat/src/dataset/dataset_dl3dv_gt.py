import os

import numpy as np
import torch

from .dataset_dl3dv import DatasetDL3DV
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from ..geometry.projection import get_fov
from ..misc.cam_utils import camera_normalization


class DatasetDL3DVGT(DatasetDL3DV):
    def _read_depth(self, image_path: str) -> torch.Tensor | None:
        depth_path = str(image_path).replace("/images/", "/depth/").replace(".png", ".npy")
        if not depth_path.endswith(".npy"):
            depth_path = depth_path.rsplit(".", 1)[0] + ".npy"
        if not os.path.exists(depth_path):
            return None
        depth = torch.from_numpy(np.load(depth_path)).float()
        if depth.ndim == 3:
            depth = depth.squeeze(-1)
        return depth

    def _load_depth_stack(self, frames) -> torch.Tensor:
        depths = []
        fallback_shape = None
        for frame in frames:
            depth = self._read_depth(frame["file_path"])
            if depth is not None:
                fallback_shape = depth.shape
                depths.append(depth)
            else:
                depths.append(None)

        if fallback_shape is None:
            img = self.load_frames([frames[0]])[0]
            fallback_shape = img.shape[-2:]

        fixed_depths = []
        for depth in depths:
            if depth is None:
                fixed_depths.append(torch.zeros(fallback_shape, dtype=torch.float32))
            else:
                if depth.shape != fallback_shape:
                    import cv2

                    depth_np = cv2.resize(
                        depth.numpy(),
                        (fallback_shape[1], fallback_shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    depth = torch.from_numpy(depth_np).float()
                fixed_depths.append(depth)
        return torch.stack(fixed_depths, dim=0)

    def getitem(self, index: int, num_context_views: int, patchsize: tuple) -> dict:
        scene = self.scene_ids[index]
        scene_frames = self.scenes[scene]

        extrinsics, intrinsics = [], []
        for frame in scene_frames:
            extrinsics.append(frame["extrinsics"])
            intrinsics.append(frame["intrinsics"])

        extrinsics = torch.tensor(np.array(extrinsics), dtype=torch.float32)
        intrinsics = torch.tensor(np.array(intrinsics), dtype=torch.float32)

        try:
            context_indices, target_indices, _ = self.view_sampler.sample(
                scene, num_context_views, extrinsics, intrinsics
            )
        except ValueError:
            raise Exception("Not enough frames")

        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            raise Exception("Field of view too wide")

        input_frames = [scene_frames[i] for i in context_indices]
        target_frames = [scene_frames[i] for i in target_indices]
        context_images = self.load_frames(input_frames)
        target_images = self.load_frames(target_frames)
        context_depth = self._load_depth_stack(input_frames)
        target_depth = self._load_depth_stack(target_frames)

        context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
        target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
        if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
            raise Exception("Bad example image shape")

        context_extrinsics = extrinsics[context_indices]
        if self.cfg.make_baseline_1:
            a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                raise Exception("baseline out of range")
            extrinsics[:, :3, 3] /= scale
            # Depth is in the same world scale as translations; keep them consistent.
            context_depth = context_depth / scale
            target_depth = target_depth / scale
        else:
            scale = 1

        if self.cfg.relative_pose:
            extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

        if self.cfg.rescale_to_1cube:
            scene_scale = torch.max(torch.abs(extrinsics[context_indices][:, :3, 3]))
            rescale_factor = 1 * scene_scale
            extrinsics[:, :3, 3] /= rescale_factor
            # Keep depth scale consistent with translation rescaling.
            context_depth = context_depth / rescale_factor
            target_depth = target_depth / rescale_factor

        if torch.isnan(extrinsics).any() or torch.isinf(extrinsics).any():
            raise Exception("encounter nan or inf in input poses")

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "depth": context_depth,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "depth": target_depth,
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,
            },
            "scene": "dl3dv_" + scene,
        }
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        intr_aug = self.stage == "train" and self.cfg.intr_augment
        example = apply_crop_shim(example, (patchsize[0] * 14, patchsize[1] * 14), intr_aug=intr_aug)

        context_pts3d = torch.ones_like(example["context"]["image"]).permute(0, 2, 3, 1)
        target_pts3d = torch.ones_like(example["target"]["image"]).permute(0, 2, 3, 1)
        context_valid_mask = example["context"]["depth"] > 0
        target_valid_mask = example["target"]["depth"] > 0

        if self.cfg.normalize_by_pts3d:
            if context_valid_mask.any():
                transformed_pts3d = context_pts3d[context_valid_mask]
                scene_factor = transformed_pts3d.norm(dim=-1).mean().clip(min=1e-8)
            else:
                scene_factor = torch.tensor(1.0, dtype=torch.float32, device=context_pts3d.device)

            context_pts3d /= scene_factor
            example["context"]["depth"] /= scene_factor
            example["context"]["extrinsics"][:, :3, 3] /= scene_factor
            target_pts3d /= scene_factor
            example["target"]["depth"] /= scene_factor
            example["target"]["extrinsics"][:, :3, 3] /= scene_factor

        example["context"]["pts3d"] = context_pts3d
        example["target"]["pts3d"] = target_pts3d
        example["context"]["valid_mask"] = context_valid_mask
        example["target"]["valid_mask"] = target_valid_mask
        # Used by mixed training (src/main_pretrained_mixed.py) to select GT vs teacher distillation.
        example["use_gt_supervision"] = True
        return example
