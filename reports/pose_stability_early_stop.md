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

当前默认配置已调整为 2 cm / 5° / 连续 2 帧；3 cm / 10° 是偏宽松但更容易触发的展示配置。如果后续大规模评估发现效果不好，可按现象调整：

- **误触发导致成功率下降**：收紧到 `translation_threshold=0.02`、`rotation_threshold_deg=5.0`，或将 `stable_frames` 提高到 3。
- **几乎不触发 early stop**：保持 3 cm / 10°，适当放宽 `mask_center_margin` 到 0.30，或将 `recent_window` 保持 2 避免三帧窗口过严。
- **mask 贴边误判**：增大 `mask_edge_margin` 到 0.05，确保目标完整后才停止探索。
- **批量评估过保守**：建议用 `task.num_envs=1` 评估 early-stop 收益；当前向量化 batch 为安全起见需要所有 env 稳定才整体停止。

## 多任务固定 Seed 对比（2026-05-25 补充）

为避免 early-stop 与 baseline 使用不同初始物体/姿态导致对比不公平，本次补充加入全局 `seed`：

- `cfg/config.yaml` 新增 `seed: null`，命令行可指定 `seed=20260525`。
- `train.py` 新增 `set_global_seed()`，同步设置 Python / NumPy / Torch seed，并把 seed 传入 task。
- `prepare_env()` 按 `seed + env_id` 初始化每个 vector env worker。
- `env/sapien_envs/base_manipulation.py` 新增 `seed()`，在环境构造和后续需要时固定 Python `random` 与 NumPy。
- `RLPoseController.run()` 返回 `exploration_steps` 与 `early_stop`，`test()` 汇总日志中新增平均探索步数和 early-stop 回合数。

固定 seed 对照命令模板：

```bash
# 启用 early stop
python train.py dataset=<task_test> task=<task> pose_estimator=<adapose_task> \
  manipulation=<manipulation> controller=rl train=test headless=True viewerless=False \
  train.total_round=3 task.num_envs=1 seed=20260525 \
  controller.load=downloads/global_scheduling_policy/<Task>_0.pt

# 关闭 early stop
python train.py dataset=<task_test> task=<task> pose_estimator=<adapose_task> \
  manipulation=<manipulation> controller=rl train=test headless=True viewerless=False \
  train.total_round=3 task.num_envs=1 seed=20260525 \
  controller.load=downloads/global_scheduling_policy/<Task>_0.pt \
  controller.controller.pose_stability_early_stop.enabled=False
```

### 固定 Seed 结果表

| 任务 | 设置 | 日志路径 | 成功率 | 平均移动距离 | 平均探索步数 | Early-stop 回合 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Cabinet | 旧阈值 3cm/10° | `outputs/2026-05-25/10-21-13/train.log` | 0.666667 | 12.050686 | 3.666667 | 3/3 |
| Cabinet | 关闭 early stop | `outputs/2026-05-25/10-21-55/train.log` | 1.000000 | 11.598612 | 4.000000 | 0/3 |
| Cabinet | 调参后 2cm/5° | `outputs/2026-05-25/10-27-23/train.log` | 1.000000 | 11.405696 | 4.000000 | 0/3 |
| Drawer | 旧阈值 3cm/10° | `outputs/2026-05-25/10-22-35/train.log` | 1.000000 | 8.532054 | 4.000000 | 0/3 |
| Drawer | 关闭 early stop | `outputs/2026-05-25/10-23-20/train.log` | 1.000000 | 8.580352 | 4.000000 | 0/3 |
| Pot | 旧阈值 3cm/10° | `outputs/2026-05-25/10-24-08/train.log` | 0.333333 | 5.435121 | 4.000000 | 0/3 |
| Pot | 关闭 early stop | `outputs/2026-05-25/10-24-45/train.log` | 0.333333 | 5.429500 | 4.000000 | 0/3 |
| Mug | 旧阈值 3cm/10° | `outputs/2026-05-25/10-25-20/train.log` | 0.333333 | 7.046344 | 4.000000 | 0/3 |
| Mug | 关闭 early stop | `outputs/2026-05-25/10-25-57/train.log` | 0.000000 | 7.030477 | 4.000000 | 0/3 |

### 结论与参数调整

- **有效性证据**：Cabinet 在旧阈值下确实能触发 early stop，并把平均探索步数从 `4.0` 降到 `3.67`，说明该机制已接入调度循环并能提前进入 manipulation。
- **鲁棒性问题**：同一 seed 的 Cabinet 对照显示 3cm/10° 过宽松，会在部分 pose 尚未足够稳定时提前停止，导致成功率下降。
- **参数调整**：默认阈值已从 `translation_threshold=0.03`、`rotation_threshold_deg=10.0` 调整为更保守的 `0.02`、`5.0`；补跑 Cabinet 后误触发消失，成功率恢复到 `1.0`。
- **跨任务表现**：Drawer/Pot/Mug 在该 seed 下均未触发 early stop，说明 mask + pose 稳定门控不会强行截断不稳定任务；Pot/Mug 的成功率主要受 manipulation / planner 本身影响。
- **推荐展示口径**：展示时同时给出“旧阈值可节省视角但可能误停”和“调参后保守门控避免误停”的对比，强调该方法是无需重训 GSP 的安全后处理，可以通过阈值在速度收益与成功率之间调节。

## 更多 10 回合固定 Seed 对照（2026-05-25 追加）

本轮使用 `seed=20260526`、`task.num_envs=1`、`train.total_round=10`，比较默认保守 early-stop 与关闭 early-stop。默认配置已改为：

```yaml
translation_threshold: 0.02
rotation_threshold_deg: 5.0
min_views: 4
stable_frames: 2
recent_window: 3
```

同时修正了统计口径：只有 `cur_step < controller.early_stop` 时触发才计为 **有效 early stop**；如果在最后一个探索 step 才满足稳定条件，只记录为 candidate，不计入提前停止。

| 任务 | 设置 | 日志路径 | 成功率 | 平均移动距离 | 平均探索步数 | 有效 Early-stop | Candidate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cabinet | 默认保守 early stop | `outputs/2026-05-25/10-35-47/train.log` | 0.800000 | 11.059422 | 4.000000 | 0/10 | 0 |
| Cabinet | 关闭 early stop | `outputs/2026-05-25/10-37-22/train.log` | 0.900000 | 11.048709 | 4.000000 | 0/10 | 0 |
| Cabinet | 宽松阈值 3cm/10° | `outputs/2026-05-25/10-39-00/train.log` | 0.800000 | 11.065854 | 4.000000 | 0/10 | 3 |
| Drawer | 默认保守 early stop | `outputs/2026-05-25/10-41-03/train.log` | 0.800000 | 8.725354 | 4.000000 | 0/10 | 2 |
| Drawer | 关闭 early stop | `outputs/2026-05-25/10-42-34/train.log` | 0.800000 | 8.726346 | 4.000000 | 0/10 | 0 |
| Pot | 默认保守 early stop | `outputs/2026-05-25/10-44-23/train.log` | 0.600000 | 5.392665 | 4.000000 | 0/10 | 0 |
| Pot | 关闭 early stop | `outputs/2026-05-25/10-45-13/train.log` | 0.600000 | 5.393830 | 4.000000 | 0/10 | 0 |
| Mug | 默认保守 early stop | `outputs/2026-05-25/10-46-04/train.log` | 0.400000 | 6.811201 | 4.000000 | 0/10 | 0 |
| Mug | 关闭 early stop | `outputs/2026-05-25/10-47-05/train.log` | 0.400000 | 6.805081 | 4.000000 | 0/10 | 0 |

### 退化分析

- **默认保守配置未发生有效提前停止**：4 个任务 40 回合中有效 early-stop 为 `0/40`，因此 Pot/Mug/Drawer 的成功率与 baseline 基本一致；Cabinet 的 `0.8 vs 0.9` 差异不是由提前停止造成，因为平均探索步数均为 `4.0` 且有效 early-stop 为 `0`，更可能来自仿真/规划/pose estimator 中仍存在的非完全确定性。
- **宽松阈值会产生误判风险**：历史 `seed=20260525` 小样本中，3cm/10° 曾在 Cabinet 的 step 3 提前停止并导致成功率下降；原因是相邻两次 bbox pose 变化小不等价于 pose 已足够正确，尤其当估计存在一致性偏差时，连续两帧都可能稳定但仍偏离真实操作位姿。
- **最后一步 candidate 不应算收益**：`seed=20260526` 的宽松阈值在 Cabinet 有 3 次 candidate，但都发生在最大探索步 `step=4`，不会减少视角数；代码已修正统计与 break 条件，避免把这种情况误报为 early stop。
- **失败主因不全在探索模块**：Pot/Mug 多次出现 `Path planner failed`、`Invalid start state`、点云碰撞等日志，说明 manipulation/planner 失败会主导成功率，不能简单归因给 early-stop。

### 改进措施

1. **阈值收紧**：默认从 `3cm/10°` 改为 `2cm/5°`，避免把“粗略稳定”误判成“可操作稳定”。
2. **最近三次 pose 检查**：默认 `recent_window=3`，要求最近两个相邻 pose 差分都小，而不是只看最后一次差分。
3. **更多视角下限**：默认 `min_views=4`，至少初始视角 + 3 次新增视角后才允许提前进入 manipulation。
4. **有效 early-stop 统计修正**：只有真正减少探索步数的触发才记为 early-stop；最后一步触发只作为 candidate 记录。
5. **保守上线策略**：当前默认配置优先保证不降低成功率；如果希望展示速度收益，可在 Cabinet 等任务上单独放宽阈值，但必须同时报告成功率变化。

### 当前结论

在更多固定 seed 实验中，改进后的默认策略没有导致可归因于 early-stop 的成功率退化；它作为安全门控较保守，主要避免误停。若要追求更明显的视角节省，需要继续做任务级阈值搜索，建议以“成功率不低于 baseline”为硬约束，再优化平均探索步数。
