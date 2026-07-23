# Deprecated Files

These files are from the v1/v3 codebases and have been superseded by the
unified v4 implementation in `post/gaussian_restorer/`.

Kept for reference. Do not import in new code.

| Old File | Replaced By |
|----------|------------|
| `refiner.py` | `repairer.py` |
| `fusion.py` | `gaussian_encoder.py` |
| `gaussian_features.py` | `gaussian_encoder.py` |
| `video_prior.py` | `video_extractor.py` |
| `gaussian_scene_restorer.py` | `repairer.py` (UnifiedRepairer) |
| `experiment_runner.py` | Use `phase1_train_unified.py` directly |
