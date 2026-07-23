"""
Feed-forward camera pose benchmark on CO3D (multi-backend) — reference implementation
======================================================================================

This script standardizes **relative pose error** evaluation on the **Common Objects
in 3D (CO3D)** test annotations, following the protocol in ``eval_pose_vggt.py``.

Supported backends (``--backends``)
-----------------------------------
- ``anysplat_baseline``: ``AnySplat.from_pretrained`` (feed-forward encoder + pose head path used in ``src.eval_pose.process_sequence``).
- ``vggt``: ``VGGT.from_pretrained`` with ``pose_enc`` decoded via ``pose_encoding_to_extri_intri``.
- ``anysplat_finetune``: optional; requires ``--finetune_ckpt``.

Sampling & reproducibility
--------------------------
- By default, each category builds a **fixed sampling plan**: ``num_frames`` indices per
  sequence, gated by ``min_num_images`` and max image size ``>= 448`` (same gates as the
  legacy script). The plan is written to ``<output>/co3d_sampling_plan.json``.
- Pass ``--sampling_plan_path`` to reuse a saved plan (recommended for paper numbers).

Metrics
-------
- Per-frame relative rotation / translation errors vs. GT extrinsics, then **AUC**
  curves at thresholds **5°, 10°, 20°, 30°** (``calculate_auc_np``), aggregated per
  category and mean over categories.

Pose alignment (``--pose_postprocess``)
---------------------------------------
- ``legacy``: align GT to the first camera only (AnySplat path flag ``gt_only``; VGGT ``gt_only``).
- ``align_both``: align **both** predictions and GT to the first camera before error.

Outputs
-------
- ``co3d_pose_metrics.json``, ``co3d_pose_summary.txt`` under ``--output_dir/<run_tag>/``.

Dependencies when vendoring
---------------------------
Requires CO3D images + ``*_test.jgz`` annotations, VGGT / AnySplat code paths under
``src.model``, ``src.utils.pose``, and ``src.eval_pose.process_sequence``.

``BENCHMARK_VERSION`` documents the protocol; bump when sampling, AUC definition, or
alignment semantics change.
"""

from __future__ import annotations

import os
import sys
import json
import gzip
import argparse
import datetime
import gc
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.encoder.vggt.models.vggt import VGGT
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.model.encoder.vggt.utils.load_fn import load_and_preprocess_images
from src.model.model.anysplat import AnySplat
from src.utils.pose import (
    align_to_first_camera,
    calculate_auc_np,
    convert_pt3d_RT_to_opencv,
    se3_to_relative_pose_error,
)
from src.eval_pose import process_sequence as process_sequence_anysplat

BENCHMARK_VERSION = "1.0.0"
BENCHMARK_NAME = "feed_forward_pose_co3d_multi_backend"


def setup_args():
    parser = argparse.ArgumentParser(
        description=(
            f"{BENCHMARK_NAME} v{BENCHMARK_VERSION}: CO3D pose eval for feed-forward models "
            "(AnySplat baseline, VGGT, optional finetune; fixed frame ids via sampling plan; "
            "pose postprocess options)."
        )
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (only test on specific category)")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Bundle adjustment (AnySplat backends only)")
    parser.add_argument("--fast_eval", action="store_true", default=False, help="Only evaluate 10 sequences per category")
    parser.add_argument("--min_num_images", type=int, default=50, help="Minimum number of images for a sequence")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to use for testing")
    parser.add_argument("--co3d_dir", type=str, required=True, help="Path to CO3D dataset")
    parser.add_argument("--co3d_anno_dir", type=str, required=True, help="Path to CO3D annotations")
    parser.add_argument(
        "--categories",
        type=str,
        default="auto",
        help='Comma-separated categories, or "auto" to detect from *_test.jgz',
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling (scan phase only)")
    parser.add_argument(
        "--vggt_repo_id",
        type=str,
        default="facebook/VGGT-1B",
        help='HuggingFace repo id for VGGT weights (default: "facebook/VGGT-1B")',
    )
    parser.add_argument(
        "--anysplat_pretrained_id",
        type=str,
        default="lhjiang/anysplat",
        help="HuggingFace id for baseline AnySplat",
    )
    parser.add_argument(
        "--finetune_ckpt",
        type=str,
        default=None,
        help="Finetuned weights: .ckpt, run dir with checkpoints/, or HF-style AnySplat folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/exp_output2_bench_finetune_singlegpu_gt",
        help="Root directory for this evaluation run.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help='Subfolder under output_dir (default: timestamp "%%Y-%%m-%%d_%%H-%%M-%%S").',
    )
    parser.add_argument(
        "--sampling_plan_path",
        type=str,
        default=None,
        help="If set, load co3d_sampling_plan.json from this path and skip the scan phase (must match categories / annotations).",
    )
    parser.add_argument(
        "--pose_postprocess",
        type=str,
        choices=["legacy", "align_both"],
        default="legacy",
        help="legacy: align GT to first camera only (previous behavior). align_both: align pred and GT to first camera on all backends.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="all",
        help='Which models to run, comma-separated: anysplat_baseline, vggt, anysplat_finetune. '
        'Use "all" to run baseline + VGGT and also anysplat_finetune when --finetune_ckpt is set.',
    )
    return parser.parse_args()


def parse_backends_selection(backends: str, finetune_ckpt: Optional[str]) -> List[str]:
    raw = backends.strip().lower()
    allowed = frozenset({"anysplat_baseline", "vggt", "anysplat_finetune"})
    if raw == "all":
        out = ["anysplat_baseline", "vggt"]
        if finetune_ckpt:
            out.append("anysplat_finetune")
        else:
            print("[info] --backends all: omitting anysplat_finetune (pass --finetune_ckpt to include it)")
        return out
    names = [x.strip().lower() for x in backends.split(",") if x.strip()]
    if not names:
        raise SystemExit("--backends is empty; use e.g. anysplat_finetune or all")
    seen = set()
    deduped: List[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        deduped.append(n)
    names = deduped
    for n in names:
        if n not in allowed:
            raise SystemExit(f"Unknown backend {n!r}. Allowed: {sorted(allowed)}")
    if "anysplat_finetune" in names and not finetune_ckpt:
        raise SystemExit("Including anysplat_finetune requires --finetune_ckpt")
    return names


def build_backends_for_eval(
    selected: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> List[Tuple[str, Callable[[], torch.nn.Module], str]]:
    """Return list of (name, loader, kind) in the same order as ``selected``."""
    out: List[Tuple[str, Callable[[], torch.nn.Module], str]] = []
    for name in selected:
        if name == "anysplat_baseline":
            hf = args.anysplat_pretrained_id
            out.append((name, lambda hf=hf: AnySplat.from_pretrained(hf), "anysplat"))
        elif name == "vggt":
            rid = args.vggt_repo_id
            out.append((name, lambda rid=rid: VGGT.from_pretrained(rid), "vggt"))
        elif name == "anysplat_finetune":
            ck = args.finetune_ckpt
            bid = args.anysplat_pretrained_id
            out.append((name, lambda ck=ck, bid=bid: load_finetune_anysplat(ck, device, bid), "anysplat"))
        else:
            raise RuntimeError(f"Unhandled backend: {name}")
    return out


def _anysplat_pose_align_flag(pose_postprocess: str) -> str:
    return "both" if pose_postprocess == "align_both" else "gt_only"


def _vggt_align_mode(pose_postprocess: str) -> str:
    return "both" if pose_postprocess == "align_both" else "gt_only"


def prepare_co3d_views(
    seq_data: list,
    co3d_dir: str,
    min_num_images: int,
    num_frames: int,
    rng: np.random.Generator,
) -> Optional[Dict[str, Any]]:
    """
    Same gate + RNG semantics as the original per-sequence eval:
    rng.choice is executed only after a valid metadata list is built; if max_size < 448 after that, returns None but RNG was already consumed.
    """
    if len(seq_data) < min_num_images:
        return None

    metadata = []
    for data in seq_data:
        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            return None
        extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append({"filepath": data["filepath"], "extri": extri_opencv})

    ids = rng.choice(len(metadata), num_frames, replace=False)
    frame_filepaths = [metadata[int(i)]["filepath"] for i in ids]
    image_names = [os.path.join(co3d_dir, fp) for fp in frame_filepaths]
    gt_extri = np.stack([np.array(metadata[int(i)]["extri"]) for i in ids], axis=0)

    max_size = max(Image.open(image_names[0]).size)
    if max_size < 448:
        return None

    return {
        "ids": ids.astype(np.int64),
        "frame_filepaths": frame_filepaths,
        "gt_extri": gt_extri,
    }


def plan_entry_to_json(entry: Dict[str, Any]) -> dict:
    return {
        "seq_name": entry["seq_name"],
        "ids": [int(x) for x in entry["ids"]],
        "frame_filepaths": list(entry["frame_filepaths"]),
        "gt_extri": np.asarray(entry["gt_extri"], dtype=np.float64).tolist(),
    }


def plan_entry_from_json(obj: dict, co3d_dir: str) -> Dict[str, Any]:
    ids = np.asarray(obj["ids"], dtype=np.int64)
    fps = list(obj["frame_filepaths"])
    image_names = [os.path.join(co3d_dir, fp) for fp in fps]
    gt_extri = np.asarray(obj["gt_extri"], dtype=np.float64)
    return {
        "seq_name": obj["seq_name"],
        "ids": ids,
        "frame_filepaths": fps,
        "image_names": image_names,
        "gt_extri": gt_extri,
    }


def build_sampling_plan(
    args: argparse.Namespace,
    categories: List[str],
) -> Dict[str, List[dict]]:
    rng = np.random.default_rng(args.seed)
    plan: Dict[str, List[dict]] = {c: [] for c in categories}

    for category in categories:
        annotation_file = os.path.join(args.co3d_anno_dir, f"{category}_test.jgz")
        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping plan")
            continue

        n_success = 0
        for seq_name, seq_data in annotation.items():
            if args.debug and not os.path.exists(os.path.join(args.co3d_dir, category, seq_name)):
                continue

            prep = prepare_co3d_views(
                seq_data,
                args.co3d_dir,
                args.min_num_images,
                args.num_frames,
                rng,
            )
            if prep is None:
                continue

            entry = {"seq_name": seq_name, **prep}
            plan[category].append(plan_entry_to_json(entry))
            n_success += 1

            if args.fast_eval and n_success >= 10:
                break

    return plan


def save_sampling_plan(path: Path, plan: Dict[str, List[dict]], meta: dict) -> None:
    payload = {"version": 1, "meta": meta, "plan": plan}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved sampling plan to {path}")


def load_sampling_plan(path: Path, co3d_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    with open(path, "r") as f:
        payload = json.load(f)
    raw_plan = payload.get("plan", payload)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for cat, entries in raw_plan.items():
        out[cat] = [plan_entry_from_json(e, co3d_dir) for e in entries]
    print(f"Loaded sampling plan from {path} ({sum(len(v) for v in out.values())} entries)")
    return out


def _relative_pose_errors_from_extrinsics(
    pred_extrinsic: torch.Tensor,
    gt_extri_np: np.ndarray,
    num_frames: int,
    device: torch.device,
    align_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    gt_extrinsic = torch.from_numpy(gt_extri_np).to(device)
    add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)
    pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
    gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

    if align_mode == "gt_only":
        gt_se3 = align_to_first_camera(gt_se3)
    elif align_mode == "both":
        pred_se3 = align_to_first_camera(pred_se3)
        gt_se3 = align_to_first_camera(gt_se3)
    else:
        raise ValueError(f"Unknown align_mode: {align_mode}")

    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)
    return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()


def infer_vggt_on_plan_entry(
    model,
    prepared: Dict[str, Any],
    category: str,
    seq_name: str,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    align_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    image_names = prepared["image_names"]
    gt_extri = prepared["gt_extri"]
    images = load_and_preprocess_images(image_names)[None].to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        pred = model(images)
        pred_all_pose_enc = pred["pose_enc"]

    with torch.cuda.amp.autocast(dtype=torch.float32):
        pred_all_extrinsic, _ = pose_encoding_to_extri_intri(pred_all_pose_enc, images.shape[-2:])
        pred_extrinsic = pred_all_extrinsic[0]

    rel_r, rel_t = _relative_pose_errors_from_extrinsics(
        pred_extrinsic, gt_extri, num_frames, device, align_mode
    )
    print(f"{category} sequence {seq_name} Rot Error: {rel_r.mean():.4f}")
    print(f"{category} sequence {seq_name} Trans Error: {rel_t.mean():.4f}")
    return rel_r, rel_t


def _resolve_checkpoint_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        ckpt_dir = path / "checkpoints"
        if ckpt_dir.is_dir():
            ckpts = list(ckpt_dir.glob("*.ckpt"))
            if ckpts:
                ckpts.sort(key=lambda p: p.stat().st_mtime)
                return ckpts[-1]
        ckpts = sorted(path.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            return ckpts[-1]
        if (path / "config.json").exists():
            return path
    raise FileNotFoundError(f"No checkpoint or AnySplat bundle found at: {path}")


def load_finetune_anysplat(ckpt: str, device: torch.device, base_hf_id: str = "lhjiang/anysplat") -> AnySplat:
    resolved = _resolve_checkpoint_file(Path(ckpt))
    if resolved.is_dir():
        model = AnySplat.from_pretrained(str(resolved))
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    model = AnySplat.from_pretrained(base_hf_id)
    try:
        blob = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(resolved, map_location="cpu")
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format in {resolved}")

    stripped: Dict[str, Any] = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        stripped[nk] = v

    model_keys = set(model.state_dict().keys())
    stripped_keys = set(stripped.keys())
    overlap = len(model_keys & stripped_keys)
    print(
        f"[finetune] ckpt keys={len(stripped)} overlap_with_AnySplat={overlap} / {len(model_keys)} "
        f"(resolved file: {resolved})"
    )

    missing, unexpected = model.load_state_dict(stripped, strict=False)
    print(f"[finetune] load_state_dict strict=False: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[finetune] missing (first 8): {missing[:8]}")
    if unexpected:
        print(f"[finetune] unexpected (first 8): {unexpected[:8]}")
    if overlap < 50 or len(missing) > len(model_keys) * 0.25:
        print(
            "[finetune][warn] Few keys matched the Hub AnySplat — weights may be mostly baseline or load is wrong; "
            "pose metrics can be misleading."
        )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _finalize_category(r_error_list: List[float], t_error_list: List[float]) -> Optional[Dict[str, Any]]:
    if not r_error_list:
        return None
    r_error = np.array(r_error_list)
    t_error = np.array(t_error_list)
    thresholds = [5, 10, 20, 30]
    aucs = {}
    for th in thresholds:
        auc, _ = calculate_auc_np(r_error, t_error, max_threshold=th)
        aucs[th] = auc
    return {
        "rError": r_error,
        "tError": t_error,
        "Auc_5": aucs[5],
        "Auc_10": aucs[10],
        "Auc_20": aucs[20],
        "Auc_30": aucs[30],
    }


def _run_one_backend_on_plan(
    backend_name: str,
    model: torch.nn.Module,
    backend_kind: str,
    plan: Dict[str, List[Dict[str, Any]]],
    args: argparse.Namespace,
    categories: List[str],
    device: torch.device,
    dtype: torch.dtype,
    pose_align_anysplat: str,
    vggt_align_mode: str,
) -> Dict[str, Any]:
    per_category: Dict[str, Any] = {}
    _unused_rng = np.random.default_rng(0)

    for category in categories:
        entries = plan.get(category) or []
        if not entries:
            print(f"[{backend_name}] No cached entries for {category}, skipping")
            continue

        annotation_file = os.path.join(args.co3d_anno_dir, f"{category}_test.jgz")
        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping")
            continue

        print(f"[{backend_name}] Evaluating {len(entries)} cached sequences for {category}")
        r_err: List[float] = []
        t_err: List[float] = []

        for prepared in entries:
            seq_name = prepared["seq_name"]
            print("-" * 50)
            print(f"[{backend_name}] {category} / {seq_name}")

            if args.debug and not os.path.exists(os.path.join(args.co3d_dir, category, seq_name)):
                print(f"Skipping {seq_name} (not found)")
                continue

            seq_data = annotation.get(seq_name)
            if seq_data is None:
                print(f"No annotation for {seq_name}, skipping")
                continue

            if backend_kind == "vggt":
                seq_r, seq_t = infer_vggt_on_plan_entry(
                    model,
                    prepared,
                    category,
                    seq_name,
                    args.num_frames,
                    device,
                    dtype,
                    vggt_align_mode,
                )
            elif backend_kind == "anysplat":
                seq_r, seq_t = process_sequence_anysplat(
                    model,
                    seq_name,
                    seq_data,
                    category,
                    args.co3d_dir,
                    args.min_num_images,
                    args.num_frames,
                    args.use_ba,
                    device,
                    dtype,
                    _unused_rng,
                    frame_ids=prepared["ids"],
                    pose_align=pose_align_anysplat,
                )
            else:
                raise ValueError(f"Unknown backend_kind: {backend_kind}")

            print("-" * 50)
            if seq_r is not None and seq_t is not None:
                r_err.extend(np.asarray(seq_r).reshape(-1).tolist())
                t_err.extend(np.asarray(seq_t).reshape(-1).tolist())

        fin = _finalize_category(r_err, t_err)
        if fin is None:
            print(f"No valid results for {category} ({backend_name}), skipping")
            continue

        print("=" * 80)
        print(f"[{backend_name}] AUC of {category} test set: {fin['Auc_30']:.4f}")
        print("=" * 80)
        per_category[category] = fin

    return per_category


def _print_and_collect_means(per_category: Dict[str, Any]) -> Dict[str, float]:
    means = {}
    if not per_category:
        return means
    for key in ["Auc_5", "Auc_10", "Auc_20", "Auc_30"]:
        means[key] = float(np.mean([per_category[c][key] for c in per_category]))
    print("\nSummary of AUC results:")
    print("-" * 50)
    for category in sorted(per_category.keys()):
        print(f"{category:<15} AUC_5: {per_category[category]['Auc_5']:.4f}")
        print(f"{category:<15} AUC_30: {per_category[category]['Auc_30']:.4f}")
        print(f"{category:<15} AUC_20: {per_category[category]['Auc_20']:.4f}")
        print(f"{category:<15} AUC_10: {per_category[category]['Auc_10']:.4f}")
    print("-" * 50)
    print(f"Mean AUC_5: {means['Auc_5']:.4f}")
    print(f"Mean AUC_30: {means['Auc_30']:.4f}")
    print(f"Mean AUC_20: {means['Auc_20']:.4f}")
    print(f"Mean AUC_10: {means['Auc_10']:.4f}")
    return means


def _save_results(
    out_root: Path,
    args: argparse.Namespace,
    all_backends: Dict[str, Dict[str, Any]],
    all_means: Dict[str, Dict[str, float]],
):
    out_root.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for bname, per_cat in all_backends.items():
        serializable[bname] = {}
        for cat, d in per_cat.items():
            serializable[bname][cat] = {
                "Auc_5": float(d["Auc_5"]),
                "Auc_10": float(d["Auc_10"]),
                "Auc_20": float(d["Auc_20"]),
                "Auc_30": float(d["Auc_30"]),
            }

    payload = {
        "benchmark": BENCHMARK_NAME,
        "version": BENCHMARK_VERSION,
        "args": vars(args),
        "per_category_auc": serializable,
        "mean_auc": {k: v for k, v in all_means.items()},
    }
    with open(out_root / "co3d_pose_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        f"CO3D pose evaluation — {BENCHMARK_NAME} v{BENCHMARK_VERSION} (AnySplat / VGGT / finetune)",
        "=" * 60,
        json.dumps(vars(args), indent=2),
        "",
    ]
    for bname in sorted(all_backends.keys()):
        lines.append(f"### {bname}")
        lines.append("-" * 40)
        pc = all_backends[bname]
        for cat in sorted(pc.keys()):
            lines.append(
                f"{cat:<15} AUC_5/10/20/30: {pc[cat]['Auc_5']:.4f} / {pc[cat]['Auc_10']:.4f} / "
                f"{pc[cat]['Auc_20']:.4f} / {pc[cat]['Auc_30']:.4f}"
            )
        if bname in all_means and all_means[bname]:
            m = all_means[bname]
            lines.append(
                f"MEAN            AUC_5/10/20/30: {m['Auc_5']:.4f} / {m['Auc_10']:.4f} / "
                f"{m['Auc_20']:.4f} / {m['Auc_30']:.4f}"
            )
        lines.append("")

    with open(out_root / "co3d_pose_summary.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {out_root / 'co3d_pose_metrics.json'} and {out_root / 'co3d_pose_summary.txt'}")


def run_feed_forward_co3d_pose_benchmark(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    if args.categories.strip().lower() == "auto":
        anno_dir = Path(args.co3d_anno_dir)
        categories = sorted(p.name[:-9] for p in anno_dir.glob("*_test.jgz") if p.name.endswith("_test.jgz"))
    else:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    if not categories:
        raise RuntimeError(f"No categories found to evaluate in {args.co3d_anno_dir}")

    if args.debug:
        categories = categories[:1]
        print(f"Debug mode on, only evaluating category: {categories[0]}")

    tag = args.run_tag or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(args.output_dir).expanduser().resolve() / tag
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {out_root}")

    pose_align_anysplat = _anysplat_pose_align_flag(args.pose_postprocess)
    vggt_align = _vggt_align_mode(args.pose_postprocess)

    if args.sampling_plan_path:
        plan_path = Path(args.sampling_plan_path).expanduser().resolve()
        hydrated_full = load_sampling_plan(plan_path, args.co3d_dir)
        hydrated = {c: hydrated_full.get(c, []) for c in categories}
        unknown = [c for c in categories if c not in hydrated_full]
        if unknown:
            print(f"[warn] No entries in loaded plan for categories (empty lists): {unknown}")
    else:
        raw_plan = build_sampling_plan(args, categories)
        plan_serializable = {cat: list(raw_plan.get(cat, [])) for cat in categories}
        meta = {
            "benchmark": BENCHMARK_NAME,
            "version": BENCHMARK_VERSION,
            "seed": args.seed,
            "co3d_dir": os.path.abspath(args.co3d_dir),
            "co3d_anno_dir": os.path.abspath(args.co3d_anno_dir),
            "num_frames": args.num_frames,
            "min_num_images": args.min_num_images,
            "fast_eval": args.fast_eval,
            "categories": categories,
        }
        save_sampling_plan(out_root / "co3d_sampling_plan.json", plan_serializable, meta)
        hydrated = {cat: [plan_entry_from_json(e, args.co3d_dir) for e in plan_serializable[cat]] for cat in categories}

    total_entries = sum(len(hydrated.get(c, [])) for c in categories)
    if total_entries == 0:
        raise RuntimeError("Sampling plan is empty — no valid sequences. Check CO3D paths and filters.")

    selected = parse_backends_selection(args.backends, args.finetune_ckpt)
    print(f"Backends to evaluate (in order): {selected}")
    backends = build_backends_for_eval(selected, args, device)

    all_backends: Dict[str, Dict[str, Any]] = {}
    all_means: Dict[str, Dict[str, float]] = {}

    for backend_name, loader, kind in backends:
        print("\n" + "#" * 80)
        print(f"Loading backend: {backend_name}")
        print("#" * 80)
        model = loader()
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        per_cat = _run_one_backend_on_plan(
            backend_name,
            model,
            kind,
            hydrated,
            args,
            categories,
            device,
            dtype,
            pose_align_anysplat,
            vggt_align,
        )
        all_backends[backend_name] = per_cat
        all_means[backend_name] = _print_and_collect_means(per_cat)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _save_results(out_root, args, all_backends, all_means)


def main() -> None:
    args = setup_args()
    run_feed_forward_co3d_pose_benchmark(args)


# Backward-compatible name for callers that imported ``evaluate`` from ``eval_pose_vggt``.
evaluate = run_feed_forward_co3d_pose_benchmark


if __name__ == "__main__":
    main()
