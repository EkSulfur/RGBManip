# Pose Stability Early Stop 汇报稿

## 1. 背景与目标

本次改进的目标是在不重新训练 Global Scheduling Policy（GSP）的前提下，复用已有 RL 视角调度策略，在视角探索过程中加入基于 pose 稳定性的提前停止机制。

原始 RL controller 通常会执行到固定最大探索步数后才进入 Manipulation Module。该流程比较安全，但当物体 6D pose / 3D bbox 估计已经稳定时，继续采集新视角会产生冗余。因此，本次改进希望在保证 manipulation 成功率不下降的前提下，减少不必要的视角探索步数，提高整体执行效率。

核心思路是：每新增一个视角并完成 pose / bbox 估计后，比较最近几次估计结果是否已经稳定；如果平移变化、旋转变化和 SAM mask 质量连续满足阈值，则提前结束探索，直接进入 Manipulation Module。

## 2. 改进思路

Pose Stability Early Stop 由三类稳定性判断和一套安全触发机制组成。

### 2.1 平移稳定性

使用预测 3D bbox 的中心近似物体平移，计算相邻两次估计中心的欧氏距离：

```text
delta_t = ||t_i - t_{i-1}||
```

如果 `delta_t` 小于配置阈值，则认为平移估计在该步稳定。

### 2.2 旋转稳定性

从预测 3D bbox 的三条局部轴恢复旋转矩阵，并计算相邻两次估计之间的旋转角差：

```text
delta_R = arccos((trace(R_i R_{i-1}^T) - 1) / 2)
```

代码中以角度形式记录为 `delta_r_deg`。如果旋转角差小于阈值，则认为旋转估计稳定。

### 2.3 Mask 质量门控

仅靠 bbox pose 稳定可能会误判，因为 pose 估计可能存在一致性偏差。因此额外加入 mask 质量门控，要求最新视角中的 mask bbox：

- mask 可见；
- bbox 不贴近图像边界；
- bbox 中心仍接近图像中心；
- 目标没有明显偏出视野。

只有 pose 稳定且 mask 质量满足要求时，才会累计稳定帧数。

### 2.4 连续稳定帧要求

不是单次满足稳定性阈值就立即停止，而是要求连续若干次满足条件。当前 Cabinet 推荐默认配置为连续 2 次稳定后触发 early-stop。

### 2.5 多环境安全策略

对于单环境评估，只要当前环境满足 early-stop 条件，就可以提前进入 manipulation。

对于向量化环境，为避免部分样本尚未稳定就被截断，代码会记录每个 env 的触发状态；只有所有 env 都稳定时，才整体提前结束探索。

## 3. 当前默认配置

当前默认配置位于 `cfg/controller/rl.yaml:9`：

```yaml
pose_stability_early_stop:
  enabled: True
  translation_threshold: 0.025
  rotation_threshold_deg: 7.5
  mask_edge_margin: 0.03
  mask_center_margin: 0.25
  min_views: 3
  stable_frames: 2
  recent_window: 2
```

该配置是 Cabinet 任务上从 `2cm / 5°` 到 `3cm / 10°` 的参数搜索中得到的折中点：

- 比 `3cm / 10°` 更保守，降低历史 seed 中出现的误停风险；
- 比 `2cm / 5°` 更容易触发有效 early-stop；
- 在 Cabinet 固定 seed 10 回合测试中，成功率不低于 baseline，同时能减少平均探索步数。

## 4. 代码改动位置

### 4.1 RL Pose Controller

主要改动集中在 `models/controller/rl_pose.py`。

| 位置 | 改动内容 |
| --- | --- |
| `models/controller/rl_pose.py:42` | 读取 `pose_stability_early_stop` 配置，并初始化 early-stop 相关状态 |
| `models/controller/rl_pose.py:168` | 新增 `bbox_to_pose()`，从预测 3D bbox 恢复中心和平滑正交化后的旋转矩阵 |
| `models/controller/rl_pose.py:191` | 新增 `pose_delta()`，计算相邻 pose 估计的平移差和旋转角差 |
| `models/controller/rl_pose.py:202` | 新增 `mask_quality()`，判断 mask bbox 是否可见、不贴边且位于视野中心附近 |
| `models/controller/rl_pose.py:222` | 新增 `update_early_stop()`，维护连续稳定计数、更新 early-stop 状态并记录指标 |
| `models/controller/rl_pose.py:547` | 每次 `add_bbox()` 后调用 `update_early_stop()`，将稳定性判定接入探索循环 |
| `models/controller/rl_pose.py:631` | 在 `run()` 中打印 early-stop candidate 日志 |
| `models/controller/rl_pose.py:649` | `run()` 返回 `exploration_steps` 和 `early_stop`，用于测试统计 |

同时修复了 `add_view()` 中多环境 mask bbox 统计问题：改为按 env 单独判断是否存在 mask 像素，避免某个 env 无 mask 时被错误标为 available。

### 4.2 配置文件

| 位置 | 改动内容 |
| --- | --- |
| `cfg/controller/rl.yaml:9` | 新增 `pose_stability_early_stop` 配置块，默认启用 |
| `cfg/config.yaml:10` | 新增 `seed: null`，支持命令行固定全局 seed |

### 4.3 Seed 固定与测试统计

| 位置 | 改动内容 |
| --- | --- |
| `train.py:46` | 新增 `set_global_seed()`，同步设置 Python、NumPy、Torch 和 CUDA seed |
| `train.py:288` | `test()` 中新增 `exploration_steps` 和 `early_stop` 统计 |
| `train.py:313` | 测试日志新增 `Average exploration steps` 和 `Early stop episodes` |
| `train.py:447` | 主入口读取全局 `seed` 并传入 task |
| `env/sapien_envs/base_manipulation.py:110` | 新增环境级 `seed()`，固定 Python `random` 与 NumPy 随机性 |

## 5. 效果对比

### 5.1 单环境功能展示

在 Cabinet 3 回合单环境 demo 中，启用 early-stop 后有 2 回合在达到最大探索视角前满足稳定条件。

| 设置 | 日志路径 | 回合数 | Early-stop 触发 | 成功率 | 平均移动距离 |
| --- | --- | ---: | --- | ---: | ---: |
| 启用 pose stability early stop | `outputs/2026-05-24/23-29-51/train.log` | 3 | 第 2 回合 step 3；第 3 回合 step 4 | 1.000000 | 10.324941 |
| 关闭 early stop | `outputs/2026-05-24/23-30-37/train.log` | 3 | 无 | 1.000000 | 10.428985 |

典型日志如下：

```text
Pose stability candidate at step 3: triggered_envs=[0], delta_t=[0.00753514], delta_r_deg=[2.3638045], stable_count=[2], mask_quality=[ True]
Pose stability candidate at step 4: triggered_envs=[0], delta_t=[0.00297632], delta_r_deg=[2.09506185], stable_count=[2], mask_quality=[ True]
```

该实验说明机制已正确接入调度循环，并能够在 pose 稳定时提前进入 manipulation。

### 5.2 固定 Seed 小样本对比

为避免 early-stop 和 baseline 使用不同初始物体或姿态导致对比不公平，后续实验加入全局 seed。

在 `seed=20260525` 的 Cabinet 3 回合对比中，旧阈值 `3cm / 10°` 能触发 early-stop，但出现成功率下降。

| 任务 | 设置 | 日志路径 | 成功率 | 平均移动距离 | 平均探索步数 | Early-stop 回合 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Cabinet | 旧阈值 3cm/10° | `outputs/2026-05-25/10-21-13/train.log` | 0.666667 | 12.050686 | 3.666667 | 3/3 |
| Cabinet | 关闭 early stop | `outputs/2026-05-25/10-21-55/train.log` | 1.000000 | 11.598612 | 4.000000 | 0/3 |
| Cabinet | 调参后 2cm/5° | `outputs/2026-05-25/10-27-23/train.log` | 1.000000 | 11.405696 | 4.000000 | 0/3 |

结论是：相邻 pose 变化小不等价于 pose 已经足够正确。如果估计存在一致性偏差，过宽阈值可能导致误停。因此后续需要引入更保守的阈值、连续帧要求和 mask 质量门控。

### 5.3 10 回合固定 Seed 对比

在 `seed=20260526`、每任务 10 回合的实验中，保守配置没有产生有效 early-stop，因此 Pot、Mug、Drawer 的成功率基本与 baseline 一致。

| 任务 | 设置 | 日志路径 | 成功率 | 平均移动距离 | 平均探索步数 | 有效 Early-stop |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Cabinet | 默认保守 early stop | `outputs/2026-05-25/10-35-47/train.log` | 0.800000 | 11.059422 | 4.000000 | 0/10 |
| Cabinet | 关闭 early stop | `outputs/2026-05-25/10-37-22/train.log` | 0.900000 | 11.048709 | 4.000000 | 0/10 |
| Drawer | 默认保守 early stop | `outputs/2026-05-25/10-41-03/train.log` | 0.800000 | 8.725354 | 4.000000 | 0/10 |
| Drawer | 关闭 early stop | `outputs/2026-05-25/10-42-34/train.log` | 0.800000 | 8.726346 | 4.000000 | 0/10 |
| Pot | 默认保守 early stop | `outputs/2026-05-25/10-44-23/train.log` | 0.600000 | 5.392665 | 4.000000 | 0/10 |
| Pot | 关闭 early stop | `outputs/2026-05-25/10-45-13/train.log` | 0.600000 | 5.393830 | 4.000000 | 0/10 |
| Mug | 默认保守 early stop | `outputs/2026-05-25/10-46-04/train.log` | 0.400000 | 6.811201 | 4.000000 | 0/10 |
| Mug | 关闭 early stop | `outputs/2026-05-25/10-47-05/train.log` | 0.400000 | 6.805081 | 4.000000 | 0/10 |

该阶段说明：保守配置不会明显造成可归因于 early-stop 的成功率退化，但因为过于保守，节省视角收益也不明显。

### 5.4 Cabinet 参数范围搜索

在 Cabinet 上固定 `seed=20260527`、`task.num_envs=1`、`train.total_round=10`，只调整 pose stability early-stop 参数。

共同设置：

```yaml
min_views: 3
stable_frames: 2
recent_window: 2
mask_edge_margin: 0.03
mask_center_margin: 0.25
```

结果如下：

| 设置 | 日志路径 | 成功率 | 平均移动距离 | 平均探索步数 | 有效 Early-stop | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 关闭 early stop | `outputs/2026-05-25/12-35-57/train.log` | 0.900000 | 11.155858 | 4.000000 | 0/10 | baseline |
| 2cm / 5° | `outputs/2026-05-25/12-38-19/train.log` | 0.900000 | 11.275272 | 3.900000 | 1/10 | 可触发但收益弱 |
| 2.5cm / 7.5° | `outputs/2026-05-25/12-39-24/train.log` | 0.900000 | 11.289485 | 3.600000 | 4/10 | 推荐折中点 |
| 3cm / 10° | `outputs/2026-05-25/12-37-19/train.log` | 0.900000 | 11.313295 | 3.600000 | 4/10 | 本 seed 不衰减，但历史 seed 有误停风险 |

Cabinet 推荐范围为：

```text
translation_threshold = 0.02 ~ 0.03
rotation_threshold_deg = 5 ~ 10
min_views = 3
stable_frames = 2
recent_window = 2
```

推荐默认点为：

```text
translation_threshold = 0.025
rotation_threshold_deg = 7.5
min_views = 3
stable_frames = 2
recent_window = 2
```

该配置在 Cabinet 10 回合固定 seed 测试中，成功率保持 `0.9`，不低于 baseline；平均探索步数从 `4.0` 降到 `3.6`；有效 early-stop 为 `4/10`。

### 5.5 跨任务参数边界复查

在 Drawer、Pot、Mug 上继续复查后发现，不同任务适合的参数并不相同。

| 任务 | 设置 | 日志路径 | 成功率 | 平均探索步数 | 有效 Early-stop | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Drawer | 关闭 early stop | `outputs/2026-05-25/13-01-03/train.log` | 0.800000 | 4.000000 | 0/10 | baseline |
| Drawer | 5cm / 15° / stable_frames=1 | `outputs/2026-05-25/13-04-27/train.log` | 1.000000 | 3.100000 | 5/10 | 可用任务级激活点 |
| Pot | 关闭 early stop | `outputs/2026-05-25/13-05-28/train.log` | 0.700000 | 4.000000 | 0/10 | baseline |
| Pot | 5cm / 15° / stable_frames=1 | `outputs/2026-05-25/13-06-15/train.log` | 0.000000 | 3.100000 | 8/10 | 激活但严重衰减 |
| Pot | 1cm / 5° / stable_frames=1 | `outputs/2026-05-25/13-10-12/train.log` | 0.800000 | 4.000000 | 0/10 | 不衰减但不发挥作用 |
| Mug | 关闭 early stop | `outputs/2026-05-25/13-07-36/train.log` | 0.000000 | 4.000000 | 0/10 | baseline |
| Mug | 5cm / 15° / min_views=2 / stable_frames=1 | `outputs/2026-05-25/13-09-18/train.log` | 0.300000 | 3.800000 | 2/10 | baseline 为 0，证据较弱 |

跨任务结论：

- Cabinet：推荐 `2.5cm / 7.5° / stable_frames=2`，能有效减少探索步数且成功率不低于 baseline。
- Drawer：可以使用更激进的任务级参数，例如 `5cm / 15° / stable_frames=1`。
- Pot：当前未找到安全激活范围，激进 early-stop 会显著降低成功率。
- Mug：baseline 本身较低，现有结果不足以证明 early-stop 安全。
- 全局默认值不应使用 Drawer 的激进参数，否则会导致 Pot 明显退化。

## 6. 总体结论

本次改进实现了一个无需重训 GSP 的后处理式视角探索提前停止机制。该机制通过比较多视角 pose 估计稳定性，并结合 mask 质量门控，在 pose 已经稳定时提前进入 manipulation。

主要结论如下：

- 机制已成功接入 RL 视角调度循环，能够在满足条件时提前进入 manipulation。
- 过宽阈值如 `3cm / 10°` 虽然容易触发，但在历史 seed 中会导致 Cabinet 误停和成功率下降。
- 加入 mask 质量门控、连续稳定帧、最近窗口检查和有效 early-stop 统计后，机制更安全。
- Cabinet 推荐默认点为 `2.5cm / 7.5° / min_views=3 / stable_frames=2 / recent_window=2`。
- 在 Cabinet 固定 seed 10 回合中，推荐配置将平均探索步数从 `4.0` 降到 `3.6`，有效 early-stop 为 `4/10`，成功率保持 `0.9`。
- Drawer 可以按任务使用更激进参数；Pot 当前不建议启用有效 early-stop；Mug 需要更高 baseline 成功率后再判断。
- 该方法适合作为“无需重训、按任务配置、安全优先”的视角探索加速模块。

## 7. 复现命令

### 7.1 语法检查

```bash
.venv/bin/python -m py_compile models/controller/rl_pose.py
```

### 7.2 单环境 Early-stop 功能展示

```bash
timeout 180s .venv/bin/python train.py \
  dataset=cabinet_test task=open_cabinet pose_estimator=adapose_cabinet \
  manipulation=open_cabinet controller=rl train=test headless=True viewerless=False \
  exp_name=early_stop_demo_single train.total_round=3 task.num_envs=1 \
  controller.load=downloads/global_scheduling_policy/Cabinet_0.pt
```

### 7.3 关闭 Early-stop 的对照运行

```bash
timeout 180s .venv/bin/python train.py \
  dataset=cabinet_test task=open_cabinet pose_estimator=adapose_cabinet \
  manipulation=open_cabinet controller=rl train=test headless=True viewerless=False \
  exp_name=no_early_stop_demo_single train.total_round=3 task.num_envs=1 \
  controller.load=downloads/global_scheduling_policy/Cabinet_0.pt \
  controller.controller.pose_stability_early_stop.enabled=False
```

### 7.4 Cabinet 推荐配置复现实验

```bash
python train.py \
  dataset=cabinet_test task=open_cabinet pose_estimator=adapose_cabinet \
  manipulation=open_cabinet controller=rl train=test headless=True viewerless=False \
  train.total_round=10 task.num_envs=1 seed=20260527 \
  controller.load=downloads/global_scheduling_policy/Cabinet_0.pt \
  controller.controller.pose_stability_early_stop.enabled=True \
  controller.controller.pose_stability_early_stop.translation_threshold=0.025 \
  controller.controller.pose_stability_early_stop.rotation_threshold_deg=7.5 \
  controller.controller.pose_stability_early_stop.min_views=3 \
  controller.controller.pose_stability_early_stop.stable_frames=2 \
  controller.controller.pose_stability_early_stop.recent_window=2
```

### 7.5 Cabinet Baseline 对照

```bash
python train.py \
  dataset=cabinet_test task=open_cabinet pose_estimator=adapose_cabinet \
  manipulation=open_cabinet controller=rl train=test headless=True viewerless=False \
  train.total_round=10 task.num_envs=1 seed=20260527 \
  controller.load=downloads/global_scheduling_policy/Cabinet_0.pt \
  controller.controller.pose_stability_early_stop.enabled=False
```

### 7.6 Drawer 任务级激活配置复现

```bash
python train.py \
  dataset=drawer_test task=open_drawer pose_estimator=adapose_drawer \
  manipulation=open_drawer controller=rl train=test headless=True viewerless=False \
  train.total_round=10 task.num_envs=1 seed=20260527 \
  controller.load=downloads/global_scheduling_policy/Drawer_0.pt \
  controller.controller.pose_stability_early_stop.enabled=True \
  controller.controller.pose_stability_early_stop.translation_threshold=0.05 \
  controller.controller.pose_stability_early_stop.rotation_threshold_deg=15.0 \
  controller.controller.pose_stability_early_stop.min_views=3 \
  controller.controller.pose_stability_early_stop.stable_frames=1 \
  controller.controller.pose_stability_early_stop.recent_window=2
```

### 7.7 日志关注指标

复现实验时主要关注以下日志字段：

```text
Success rate
Average distance
Average exploration steps
Early stop episodes
Pose stability candidate at step ...
```

其中：

- `Success rate` 用于判断成功率是否下降；
- `Average distance` 用于观察整体移动距离变化；
- `Average exploration steps` 用于衡量是否减少视角探索；
- `Early stop episodes` 表示真正发生有效提前停止的回合数；
- 最后一步才满足条件的 candidate 不计为有效 early-stop。

## 8. 下一版流程：Observe / Dry-run 统计优先

当前默认流程已改为 `mode: observe`：计算 pose stability early-stop candidate，但不真正提前停止，完整跑到最大探索步后再进入 manipulation。这样可以先统计各任务、各帧之间的 pose / mask 变化，再离线选择任务级阈值。

新增日志包括：

```text
Pose stability candidate at step ...
Pose stability frame stats step ... delta_t_to_final=... delta_r_to_final=... mask_area=...
Pose stability observe mode: enabled
Simulated early stop episodes: ...
```

其中 `delta_t_to_final` 和 `delta_r_to_final` 使用完整探索最后一帧 pose 作为 pseudo reference，用于判断中间帧“看似稳定”时是否已经接近最终估计。`Simulated early stop episodes` 只统计最大探索步之前触发的 candidate，最后一步 candidate 不计为收益。

上线策略调整为：

1. 先在各任务运行 observe 模式，完整采集多 seed、多 episode 的逐帧统计。
2. 按任务、成功/失败 episode、触发 step 分组分析稳定性分布。
3. 离线模拟不同阈值下的触发率、节省探索步数和潜在误停率。
4. 只有经过统计验证的任务级参数才允许将 `mode` 从 `observe` 改为 `stop`。

这比直接手工调整全局阈值更合理，也能避免 Pot 这类“容易提前触发但成功率明显下降”的任务被激进 early-stop 误伤。

## 9. 三任务真实参数标定结果（2026-05-27）

本轮只标定 Cabinet / Drawer / Pot。先用 `mode=observe` 跑完整探索，再用 `scripts/calibrate_pose_stability.py` 离线模拟阈值，最后对 Cabinet / Drawer 做真实 `mode=stop` 验证。

| 任务 | 推荐策略 | 验证日志 | 成功率 | 平均探索步数 | Early-stop | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Cabinet | `0.015 / 10° / min_views=3 / stable_frames=1` | `outputs/2026-05-27/11-07-10/train.log` | 1.000000 | 3.400000 | 4/10 | 可启用任务级 stop |
| Drawer | `0.030 / 12.5° / min_views=3 / stable_frames=1` | `outputs/2026-05-27/11-08-11/train.log` | 1.000000 | 3.100000 | 5/10 | 可启用任务级 stop |
| Pot | 不启用有效 stop，保持 observe/disabled | `outputs/2026-05-27/11-01-05/train.log` | 0.800000 | 4.000000 | 0/10 actual | active 候选 risky，不建议 stop |

离线标定中，Pot 的可触发候选相对完整探索最终 pose 的 `risky_trigger_rate` 为 `1.0`，因此不应为了节省视角启用真实截断。全局默认仍建议保持 `mode=observe`；实际启用时只对 Cabinet / Drawer 做任务级覆盖。

## 10. Drawer 进一步验证（2026-05-27）

阅读论文后，本轮把 Drawer early-stop 的验证重点放在 active perception 的效率-精度权衡上：只有当 pose 已稳定且不损失成功率时，才把减少视角作为有效收益。

新增三 seed 真实 `mode=stop` 对照结果如下：

| 参数 | Seeds | 成功率对比 | 平均探索步数 | Early-stop | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `0.030 / 12.5° / min_views=3 / stable_frames=1` | 20260528-20260530 | baseline `21/30`，stop `23/30` | `4.00 -> 3.07` | 19/30 | 合计有效，但 seed 20260528 从 `0.9` 降到 `0.8`，不适合作为保守默认 |
| `0.020 / 10° / min_views=3 / stable_frames=1` | 20260528-20260530 | baseline `21/30`，stop `27/30` | `4.00 -> 3.17` | 17/30 | 当前推荐 Drawer 任务级参数 |

Drawer 当前推荐任务级覆盖：

```yaml
pose_stability_early_stop:
  enabled: True
  mode: stop
  translation_threshold: 0.020
  rotation_threshold_deg: 10.0
  min_views: 3
  stable_frames: 1
  recent_window: 2
```

结论表述应保持谨慎：在当前 3 seed / 30 回合真实截断验证中，Drawer 保守参数能早停提升效率，并且未观察到成功率损失；但这不是跨所有随机初始化的严格保证。


### 10.1 Drawer 论文式完整测试与 early-stop 子集统计

按论文中 Open Drawer 测试设置，使用 `drawer_test`、`open_drawer`、`adapose_drawer`、`Drawer_0.pt` 做完整测试。论文原始方法没有 pose-stability early-stop，可作为无 early-stop 的公开基准；论文 Table I 中 Ours 在 Open Drawer 上 Train 为 `83.0%`，Test 为 `87.0%`。

Drawer 无 early-stop / early-stop 对照：

| 设置 | 来源/日志路径 | 总轨迹 | 成功率 | 平均距离 | 平均探索步数 | Early-stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 论文 Ours，无 pose-stability early-stop | `RGBManip Monocular Image-based Robotic Manipulation through Active Object Pose Estimation.pdf` Table I | - | 0.870000 | - | 4 views | 0 |
| 本地复现，`num_envs=8` / `total_round=100` / `mode=stop` | `outputs/2026-05-27/12-03-01/train.log` | 800 | 0.868750 | 9.948265 | 4.000000 | 0/800 |
| 本地真实截断，`num_envs=1` / `total_round=100` / `mode=stop` | `outputs/2026-05-27/13-38-43/train.log` | 100 | 0.850000 | 9.572974 | 3.770000 | 18/100 |

说明：论文 Open Drawer 任务要求抽屉打开超过 `15 cm`，并以平均成功率作为主要评价指标。论文中的原始 RGBManip 流程使用固定多视角主动感知，没有本次新增的 pose-stability early-stop，因此 `87.0%` 可作为 Drawer 无 early-stop 的论文级对照。本地 `num_envs=8` 复现结果 `86.875%` 基本贴近论文 `87.0%`；但由于多环境安全策略要求所有并行 env 同时稳定才会真正截断，`num_envs=8` 下没有有效 early-stop，因此主要用于验证成功率口径。

单环境 early-stop 子集统计：

其中 18 条真实提前停止轨迹为：

```text
episode = [1, 4, 7, 13, 22, 27, 31, 37, 38, 42, 50, 67, 71, 75, 82, 90, 95, 99]
```

这些提前停止轨迹的成功率统计如下：

| 子集 | 成功轨迹 | 总轨迹 | 成功率 | 失败 episode |
| --- | ---: | ---: | ---: | --- |
| 有效 early-stop 轨迹 | 15 | 18 | 0.833333 | `[38, 42, 67]` |
| 非 early-stop 轨迹 | 70 | 82 | 0.853659 | - |
| 全部单环境轨迹 | 85 | 100 | 0.850000 | - |

结论：Drawer 单环境完整测试中，early-stop 子集成功率为 `15/18 = 83.33%`，略低于非 early-stop 子集 `70/82 = 85.37%` 和总体 `85.00%`。与论文无 early-stop 的 Open Drawer Test `87.0%` 相比，本地单环境 early-stop 总体为 `85.0%`，仍需谨慎表述：当前结果说明 Drawer 参数能减少探索步数，但不能仅凭这一组随机序列证明相对无 early-stop 严格无损。
