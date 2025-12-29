CHANGES FROM GitHub Copilot
Date: 2025-12-28

概要：本文件记录由 Copilot（开发助理）在本仓库内做的修改。

修改记录：
- 2025-12-28  修改：`src/dataset/shims/crop_shim.py`
  - 说明：在 `rescale_and_crop` 中加入了 aspect-ratio 随机采样（范围 [0.5, 1.0]），并添加了中文注释；取消了原先在该函数中不再稳健的严格 assert。此改动用于在训练时（当 `intr_aug=True`）实现论文所述的长宽比随机化。
  - 参见文件：[AnySplat/src/dataset/shims/crop_shim.py](src/dataset/shims/crop_shim.py)

- 2025-12-28  修改：`scripts/preprocess_co3d.py`
  - 说明：将脚本参数 `--img_size` 的默认值从 `512` 改为 `448`，以匹配论文要求的最长边 448 px（可在运行时通过命令行覆盖）。
  - 参见文件：[AnySplat/scripts/preprocess_co3d.py](scripts/preprocess_co3d.py)

- 2025-12-28  修改：`config/experiment/co3d.yaml`
  - 说明：更新实验配置中的 loss 权重以匹配论文给出的 λ 值（在文件中添加了中文注释，说明了 λ 与具体 loss 的映射）。具体调整示例：`mse.weight=0.05`，`depth_consis.weight=0.1`（保持），`chamfer_distance.weight=10.0`，`opacity.weight=1.0`。
  - 参见文件：[AnySplat/config/experiment/co3d.yaml](config/experiment/co3d.yaml)

备注与建议：
- 已在代码中以中文注释标注了对 `src` 下文件的修改（见相应文件）。
- 推荐在训练前运行一次小规模的数据预处理与训练验证，确认新增的 aspect-ratio 增强行为在实践中不会引入边缘问题。

回滚：如需回滚这些变更，请使用 git 查看变更并重置相关提交；我也可以为你生成对应的回滚补丁。
