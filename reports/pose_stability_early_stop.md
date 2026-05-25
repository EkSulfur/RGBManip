# Pose Stability Early Stop 改进记录

## 目标

在不重新训练 Global Scheduling Policy (GSP) 的前提下，复用现有 RL 视角调度策略；每次新增视角并完成 6D pose / 3D bbox 估计后，比较最近估计结果是否已经稳定。如果平移、旋转和 SAM mask 质量连续满足阈值，则提前结束探索并直接进入 Manipulation Module。

## 规则设计

- **平移稳定性**：由预测 3D bbox 中心近似物体平移，计算 `Δt = ||t_i - t_{i-1}||`。
- **旋转稳定性**：由预测 3D bbox 的三条局部轴构造旋转矩阵，计算 `ΔR = arccos((trace(R_i R_{i-1}^T)-1)/2)`，代码中以角度记录。
- **mask 质量门控**：使用最新视角的 mask bbox，要求 bbox 不贴边且 bbox 中心仍接近图像中心。
- **连续帧要求**：默认连续 2 次新增视角满足稳定性后触发 early stop。
- **批量环境处理**：单环境可立即 early stop；向量化环境中记录每个 env 的触发候选，只有所有 env 都稳定时才整体提前进入 manipulation，避免未稳定样本被过早截断。

当前默认参数位于 `cfg/controller/rl.yaml`：

```yaml
pose_stability_early_stop:
  enabled: True
  translation_threshold: 0.03
  rotation_threshold_deg: 10.0
  mask_edge_margin: 0.03
  mask_center_margin: 0.25
  min_views: 3
  stable_frames: 2
  recent_window: 2
```

## 代码修改位置

- `models/controller/rl_pose.py`
  - 新增 `bbox_to_pose()`：从预测 3D bbox 恢复中心和平滑正交化后的旋转矩阵。
  - 新增 `pose_delta()`：计算相邻 pose 估计的 `Δt` 和 `ΔR`。
  - 新增 `mask_quality()`：检查 mask bbox 是否可见、不贴边、处于视野中心区域。
  - 新增 `update_early_stop()`：维护连续稳定计数、记录 early-stop 指标并返回触发状态。
  - 修改 `step()`：每次 `add_bbox()` 后立即更新稳定性判定，并把指标写入 `info`。
  - 修改 `run()`：当 early-stop 条件满足时跳出视角探索循环，直接调用 `call_manipulation()`。
  - 修复 `add_view()` 中多环境 mask bbox 统计：改为按 env 单独判断是否有 mask 像素，避免某个 env 无 mask 时被错误标为 available。
- `cfg/controller/rl.yaml`
  - 增加 `pose_stability_early_stop` 配置块，默认启用 3 cm / 10° / 连续 2 帧规则。

## 验证命令

语法和规则 smoke test：

```bash
.venv/bin/python -m py_compile models/controller/rl_pose.py
```

单环境 early-stop 展示运行：

```bash
timeout 180s .venv/bin/python train.py \
  dataset=cabinet_test task=open_cabinet pose_estimator=adapose_cabinet \
  manipulation=open_cabinet controller=rl train=test headless=True viewerless=False \
  exp_name=early_stop_demo_single train.total_round=3 task.num_envs=1 \
  controller.load=downloads/global_scheduling_policy/Cabinet_0.pt
```

关闭 early-stop 的对照运行：

```bash
timeout 180s .venv/bin/python train.py \
  dataset=cabinet_test task=open_cabinet pose_estimator=adapose_cabinet \
  manipulation=open_cabinet controller=rl train=test headless=True viewerless=False \
  exp_name=no_early_stop_demo_single train.total_round=3 task.num_envs=1 \
  controller.load=downloads/global_scheduling_policy/Cabinet_0.pt \
  controller.controller.pose_stability_early_stop.enabled=False
```

## 运行结果展示

| 设置 | 日志路径 | 回合数 | Early-stop 触发 | 成功率 | 平均移动距离 |
| --- | --- | ---: | --- | ---: | ---: |
| 启用 pose stability early stop | `outputs/2026-05-24/23-29-51/train.log` | 3 | 第 2 回合 step 3；第 3 回合 step 4 | 1.000000 | 10.324941 |
| 关闭 early stop | `outputs/2026-05-24/23-30-37/train.log` | 3 | 无 | 1.000000 | 10.428985 |

Early-stop 样例日志：

```text
Pose stability candidate at step 3: triggered_envs=[0], delta_t=[0.00753514], delta_r_deg=[2.3638045], stable_count=[2], mask_quality=[ True]
Pose stability candidate at step 4: triggered_envs=[0], delta_t=[0.00297632], delta_r_deg=[2.09506185], stable_count=[2], mask_quality=[ True]
```

结果说明：

- 启用 early-stop 后，3 个单环境 demo 中有 2 个在达到最大探索视角前满足稳定条件并提前进入 manipulation。
- 两组 3 回合 demo 的成功率均为 100%，启用 early-stop 的平均移动距离略低。
- 该对比样本量较小且未固定同一 task seed，仅作为功能展示；正式量化建议在固定 object/seed 的 50–100 回合上统计平均探索步数、成功率和移动距离。

## 参数调整建议

当前 3 cm / 10° / 连续 2 帧是偏宽松但实测能触发的配置。如果后续大规模评估发现效果不好，可按现象调整：

- **误触发导致成功率下降**：收紧到 `translation_threshold=0.02`、`rotation_threshold_deg=5.0`，或将 `stable_frames` 提高到 3。
- **几乎不触发 early stop**：保持 3 cm / 10°，适当放宽 `mask_center_margin` 到 0.30，或将 `recent_window` 保持 2 避免三帧窗口过严。
- **mask 贴边误判**：增大 `mask_edge_margin` 到 0.05，确保目标完整后才停止探索。
- **批量评估过保守**：建议用 `task.num_envs=1` 评估 early-stop 收益；当前向量化 batch 为安全起见需要所有 env 稳定才整体停止。
