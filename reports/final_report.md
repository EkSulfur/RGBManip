# 5分钟代码展示汇报
## 一、论文方法回顾
### 1. 论文解决的核心问题
- 传统机器人操控高度依赖深度相机与点云设备，硬件成本高、点云稀疏带噪声，易受环境光照干扰，对玻璃、金属等透明及反光物体适配效果差。
- 单目RGB相机成本低、纹理信息丰富，但天然缺乏深度与3D空间信息，存在单目位姿歧义，无法直接精确估计物体6D位姿。
- 多视角主动感知存在精度与效率冲突：观测视角越多，位姿估计越精准，但会增加机械臂移动路径与任务耗时，难以自适应平衡。

### 2. 论文 Motivation
摒弃深度相机和点云依赖，仅使用手眼单目RGB相机完成各类机器人操控任务；同时解决传统多视角固定观测模式下，无法自主平衡**位姿估计精度**与**机器人操控效率**的痛点。

### 3. 论文解决方式
- 采用手眼一体式单目RGB相机，随机械臂运动主动从多视角观测目标物体。
- 引入运动学引导多视角6D位姿估计，利用机器人运动学约束消除单目深度歧义，实现类别级物体部件位姿预测。
- 基于PPO强化学习做全局调度，自主规划观测视角。
- 搭配闭环阻抗控制器执行操控，容忍位姿微小误差，适配开门、抽拉抽屉等铰接物体任务。

## 二、原有代码运行展示
### 1. 代码核心功能
开源代码完整复现论文全流程：仿真环境搭建、机械臂多视角主动感知、6D物体位姿估计、强化学习视角调度、阻抗控制机器人操控。

### 2. 代码运行呈现内容
- 通过命令行可自由指定任务类型、数据集、位姿估计器及运行模式。
- 运行后仿真环境中，机械臂自动规划固定数量观测点位，依次移动到不同视角采集RGB图像。
- 全程实时运行位姿估计，持续输出物体平移、旋转6D位姿结果。
- 必须完成预设全部观测视角后，才会进入操控模块执行任务。

### 3. 原代码固有缺陷
无论物体位姿是否提前收敛稳定，代码都会强制跑完所有固定观测视角，产生大量无效移动路径，任务冗余耗时高，资源利用率低。

## 三、代码修改

为了提高效率，我们引入 **Pose Stability Early Stop**：每完成一个新视角观测后，不再只依赖固定 `early_stop=4` 的最大观测步数，而是实时判断“位姿估计是否已经稳定、目标是否仍处于可靠视野内”。当连续若干帧满足稳定条件时，直接跳出主动感知循环，进入后续 manipulation。

### 1. Early-stop 参数配置

核心配置位于 `cfg/controller/rl.yaml`：

```yaml
controller:
  early_stop: 4
  pose_stability_early_stop:
    enabled: True
    mode: stop
    translation_threshold: 0.025
    rotation_threshold_deg: 7.5
    mask_edge_margin: 0.03
    mask_center_margin: 0.25
    min_views: 3
    stable_frames: 2
    recent_window: 2
```

含义：
- `translation_threshold` / `rotation_threshold_deg`：相邻视角位姿变化阈值，小于阈值认为位姿稳定。
- `mask_edge_margin` / `mask_center_margin`：保证目标 mask 不贴边、不过度偏离图像中心。
- `min_views`：至少采集一定数量视角后才允许判断 early-stop，避免第一二帧误停。
- `stable_frames`：连续满足稳定条件的帧数要求。
- `mode=stop`：真实提前停止；如需只统计不截断，可切换为 `observe`。

### 2. 位姿变化计算

核心代码位于 `models/controller/rl_pose.py`，先把 3D bbox 转成平移和旋转，再计算相邻观测之间的平移差和旋转角差：

```python
def pose_delta(self, last_bbox, cur_bbox):
    last_t, last_r = self.bbox_to_pose(last_bbox)
    cur_t, cur_r = self.bbox_to_pose(cur_bbox)
    delta_t = np.linalg.norm(cur_t - last_t, axis=-1)
    trace = np.einsum("nii->n", cur_r @ np.swapaxes(last_r, -1, -2))
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    delta_r = np.degrees(np.arccos(cos_angle))
    return delta_t, delta_r
```

这一步对应 early-stop 的核心判断依据：如果新视角相比前一视角的 6D 位姿变化已经很小，说明继续移动相机带来的信息增益有限。

### 3. Mask 质量门控

仅靠位姿变化小还不够，因为目标可能被遮挡、贴边或检测不完整。因此加入 mask 质量门控：

```python
def mask_stats(self, view_idx):
    bbox = self.bbox_queue[view_idx]
    available = self.available[view_idx].astype(bool)
    edge_margin = self.early_stop_cfg.get("mask_edge_margin", 0.03)
    center_margin = self.early_stop_cfg.get("mask_center_margin", 0.25)

    center = (bbox[:, :2] + bbox[:, 2:]) / 2
    size = np.clip(bbox[:, 2:] - bbox[:, :2], 0.0, 1.0)
    area = size[:, 0] * size[:, 1]
    center_offset = np.linalg.norm(center - np.array([[0.5, 0.5]]), axis=-1)

    not_touching_edge = (
        (bbox[:, 0] > edge_margin) &
        (bbox[:, 1] > edge_margin) &
        (bbox[:, 2] < 1 - edge_margin) &
        (bbox[:, 3] < 1 - edge_margin)
    )
    centered = (
        (np.abs(center[:, 0] - 0.5) < center_margin) &
        (np.abs(center[:, 1] - 0.5) < center_margin)
    )
    quality = available & not_touching_edge & centered
    return {"quality": quality, "area": area, "center_offset": center_offset}
```

该门控保证 early-stop 只在“目标检测可靠、目标位置合理”的情况下触发，降低错误提前停止的风险。

### 4. Early-stop 核心判断函数

`update_early_stop()` 在每次新增视角后调用，逻辑是：
1. 若视角数不足 `min_views`，只记录指标，不允许停止；
2. 检查当前 mask 是否可靠；
3. 在最近 `recent_window` 个视角中计算最大平移差和旋转差；
4. 若位姿变化小于阈值且 mask 可靠，则稳定计数加一；
5. 稳定计数达到 `stable_frames` 后，标记该环境触发 early-stop。

```python
def update_early_stop(self):
    if not self.early_stop_enabled:
        return np.zeros((self.num_envs,), dtype=bool)

    cur_idx = self.accumulate_steps % self.max_steps
    min_views = self.early_stop_cfg.get("min_views", 3)
    stable_required = self.early_stop_cfg.get("stable_frames", 2)
    recent_window = self.early_stop_cfg.get("recent_window", 2)
    trans_thresh = self.early_stop_cfg.get("translation_threshold", 0.03)
    rot_thresh = self.early_stop_cfg.get("rotation_threshold_deg", 10.0)

    mask = self.mask_stats(cur_idx)
    if self.accumulate_steps + 1 < min_views:
        return self.early_stop_triggered

    stable = mask["quality"].copy()
    delta_t = np.zeros((self.num_envs,))
    delta_r = np.zeros((self.num_envs,))
    comparisons = min(recent_window - 1, self.accumulate_steps)

    for offset in range(comparisons):
        newer_idx = (self.accumulate_steps - offset) % self.max_steps
        older_idx = (self.accumulate_steps - offset - 1) % self.max_steps
        cur_delta_t, cur_delta_r = self.pose_delta(
            self.pred_bbox[older_idx], self.pred_bbox[newer_idx]
        )
        delta_t = np.maximum(delta_t, cur_delta_t)
        delta_r = np.maximum(delta_r, cur_delta_r)
        stable &= (cur_delta_t < trans_thresh) & (cur_delta_r < rot_thresh)

    self.early_stop_stable_count = np.where(
        stable, self.early_stop_stable_count + 1, 0
    )
    candidate = self.early_stop_stable_count >= stable_required
    self.early_stop_triggered |= candidate
    return self.early_stop_triggered
```

### 5. 接入主动感知循环

在 `RLPoseController.run()` 中，每一步移动相机并更新位姿估计后，读取 `early_stop_triggered`。如果所有并行环境都满足 early-stop，且当前步数还没达到最大探索步，就提前跳出循环：

```python
while True:
    cur_step += 1
    actions = self.controller.actor_critic.act_inference(current_obs)
    next_obs, rews, dones, infos = self.control_interface.step(actions, eval=True)

    early_stop = self.control_interface.early_stop_triggered
    stop_by_pose = (
        not self.control_interface.early_stop_observe and
        early_stop.all() and
        cur_step < max_step
    )

    if dones.any() or stop_by_pose or cur_step >= max_step:
        break

estimation = self.control_interface.pred_bbox[cur_step]
self.control_interface.call_manipulation(estimation, eval)
```

因此修改后的流程从“固定采集 4 个视角”变为：

```text
采集新视角 -> 更新6D位姿 -> 判断mask质量 -> 判断位姿稳定性
        -> 若稳定则提前进入操控，否则继续采集下一个视角
```

## 四、修改后结果比较

### 1. 三任务真实参数标定结果

本轮只标定 Cabinet / Drawer / Pot。流程为：先用 `mode=observe` 跑完整探索，再用 `scripts/calibrate_pose_stability.py` 离线模拟阈值，最后对 Cabinet / Drawer 做真实 `mode=stop` 验证。

| 任务 | 推荐策略 | 验证日志 | 成功率 | 平均探索步数 | Early-stop | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Cabinet | `0.015 / 10° / min_views=3 / stable_frames=1` | `outputs/2026-05-27/11-07-10/train.log` | 1.000000 | 3.400000 | 4/10 | 可启用任务级 stop |
| Drawer | `0.030 / 12.5° / min_views=3 / stable_frames=1` | `outputs/2026-05-27/11-08-11/train.log` | 1.000000 | 3.100000 | 5/10 | 可启用任务级 stop |
| Pot | 不启用有效 stop，保持 observe/disabled | `outputs/2026-05-27/11-01-05/train.log` | 0.800000 | 4.000000 | 0/10 actual | active 候选 risky，不建议 stop |

离线标定中，Pot 的可触发候选相对完整探索最终 pose 的 `risky_trigger_rate` 为 `1.0`，因此不应为了节省视角启用真实截断。全局默认仍建议保持 `mode=observe`；实际启用时只对 Cabinet / Drawer 做任务级覆盖。

### 2. Drawer 多 seed 对照结果

阅读论文后，Drawer early-stop 的验证重点放在 active perception 的效率-精度权衡上：只有当 pose 已稳定且不损失成功率时，才把减少视角作为有效收益。

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

### 3. Drawer 论文式完整测试与 early-stop 子集统计

按论文中 Open Drawer 测试设置，使用 `drawer_test`、`open_drawer`、`adapose_drawer`、`Drawer_0.pt` 做完整测试。论文原始方法没有 pose-stability early-stop，可作为无 early-stop 的公开基准；论文 Table I 中 Ours 在 Open Drawer 上 Test 为 `87.0%`。

| 设置 | 来源/日志路径 | 总轨迹 | 成功率 | 平均距离 | 平均探索步数 | Early-stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 论文 Ours，无 pose-stability early-stop | `RGBManip Monocular Image-based Robotic Manipulation through Active Object Pose Estimation.pdf` Table I | - | 0.870000 | - | 4 views | 0 |
| 本地复现，`num_envs=8` / `total_round=100` / `mode=stop` | `outputs/2026-05-27/12-03-01/train.log` | 800 | 0.868750 | 9.948265 | 4.000000 | 0/800 |
| 本地真实截断，`num_envs=1` / `total_round=100` / `mode=stop` | `outputs/2026-05-27/13-38-43/train.log` | 100 | 0.850000 | 9.572974 | 3.770000 | 18/100 |

单环境 100 条轨迹中，18 条发生真实提前停止：

```text
episode = [1, 4, 7, 13, 22, 27, 31, 37, 38, 42, 50, 67, 71, 75, 82, 90, 95, 99]
```

这些提前停止轨迹的成功率统计如下：

| 子集 | 成功轨迹 | 总轨迹 | 成功率 | 失败 episode |
| --- | ---: | ---: | ---: | --- |
| 有效 early-stop 轨迹 | 15 | 18 | 0.833333 | `[38, 42, 67]` |
| 非 early-stop 轨迹 | 70 | 82 | 0.853659 | - |
| 全部单环境轨迹 | 85 | 100 | 0.850000 | - |

### 4. 运行指令

关闭 pose-stability early-stop 的 baseline 对照：

```bash
python train.py test=True task=drawer_test task.env.name=open_drawer \
  pose_estimator=adapose_drawer controller=rl \
  controller.ckpt=checkpoints/Drawer_0.pt \
  task.num_envs=1 train.total_round=10 seed=20260528 \
  controller.controller.pose_stability_early_stop.enabled=False
```

开启 Drawer 推荐 early-stop 参数：

```bash
python train.py test=True task=drawer_test task.env.name=open_drawer \
  pose_estimator=adapose_drawer controller=rl \
  controller.ckpt=checkpoints/Drawer_0.pt \
  task.num_envs=1 train.total_round=10 seed=20260528 \
  controller.controller.pose_stability_early_stop.enabled=True \
  controller.controller.pose_stability_early_stop.mode=stop \
  controller.controller.pose_stability_early_stop.translation_threshold=0.020 \
  controller.controller.pose_stability_early_stop.rotation_threshold_deg=10.0 \
  controller.controller.pose_stability_early_stop.min_views=3 \
  controller.controller.pose_stability_early_stop.stable_frames=1 \
  controller.controller.pose_stability_early_stop.recent_window=2
```

完整 100 轮 Drawer 单环境测试：

```bash
python train.py test=True task=drawer_test task.env.name=open_drawer \
  pose_estimator=adapose_drawer controller=rl \
  controller.ckpt=checkpoints/Drawer_0.pt \
  task.num_envs=1 train.total_round=100 seed=20260527 \
  controller.controller.pose_stability_early_stop.enabled=True \
  controller.controller.pose_stability_early_stop.mode=stop \
  controller.controller.pose_stability_early_stop.translation_threshold=0.020 \
  controller.controller.pose_stability_early_stop.rotation_threshold_deg=10.0 \
  controller.controller.pose_stability_early_stop.min_views=3 \
  controller.controller.pose_stability_early_stop.stable_frames=1 \
  controller.controller.pose_stability_early_stop.recent_window=2
```

离线标定 observe 日志中的阈值：

```bash
python scripts/calibrate_pose_stability.py outputs/2026-05-27/11-01-05/train.log --top-k 30
```

### 5. 结果讨论

- 本次 Pose Stability Early Stop 改进把固定 4 视角采集改为“位姿稳定后按需停止”，能减少冗余观测并缩短 active perception 阶段。
- Cabinet / Drawer 能找到可用任务级参数，其中 Drawer 推荐保守点 `0.020 / 10° / min_views=3 / stable_frames=1`，三 seed 对照中成功率从 `21/30` 提升到 `27/30`，平均探索步数从 `4.00` 降到 `3.17`。
- Pot 不建议启用真实截断，因为离线标定显示可触发候选风险高，强行 early-stop 可能损失后续 manipulation 成功率。
- Drawer 100 轮完整测试中，真实截断平均探索步数为 `3.77`，但总体成功率 `85.0%` 略低于论文无 early-stop 的 `87.0%`，因此应谨慎表述为“提高效率且当前验证未发现明显退化”，不能说严格无损。
- 改动无需重训 PPO 或 pose estimator，只在全局调度层增加稳定性判断与提前退出逻辑，便于按任务配置阈值。
