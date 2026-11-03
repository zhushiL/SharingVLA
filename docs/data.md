# 数据指南

使用 LeRobot v3 数据集。每帧包含 RGB、机器人 state、action 和全局任务；训练子任务语言 loss 时，还需要当前帧的子任务索引。图像仅解码配置选中的相机。

## 配置数据路径

单数据集训练设置：

```yaml
data:
  data_name: own_robot
  train_path: ../data/bi_rebot_09_pick_and_place_1023-annotated
  robot_config_root: configs/robot_configs
  norm_stats_file: assets/norm_stats/own_robot.json
```

相对路径以仓库根目录为基准。`data_name` 对应机器人映射文件名，例如 `configs/robot_configs/own_robot.yaml`。

## own_robot 字段映射

仓库提供一个双臂示例。原生 state 和 action 都是 20 维：

| 原生切片 | 内容 |
| --- | --- |
| `0:9` | 左 TCP：xyz + 6D rotation |
| `9:10` | 左夹爪 |
| `10:19` | 右 TCP：xyz + 6D rotation |
| `19:20` | 右夹爪 |

训练将左右 TCP 拼为 18 维，左右夹爪拼为 2 维。TCP 动作默认转换为相对当前 state 的局部位姿，夹爪保持绝对值。部署时会还原为上表的原生顺序。

| 逻辑相机 | 数据中的原生字段 |
| --- | --- |
| `camera_top` | `observation.images.head` |
| `camera_wrist_left` | `observation.images.left_left_wrist` |
| `camera_wrist_right` | `observation.images.right_right_wrist` |

独立策略只配置左腕、右腕；完整 VLA 按头部、左腕、右腕配置。VLA 自动读取 `t-head_history_offset` 和 `t` 的头部帧，历史不足时使用该 episode 的首帧。默认图像尺寸为 256×256。

更换机器人时，先修改映射中的原生字段、切片和动作语义，再同步调整训练配置的 `joints`、`norm_type`、`action_dim`、`max_action_dim`、`max_state_dim`。`arm.position` 已保留为注释；只有补齐真实关节字段映射后才能启用。最后重新计算统计。

## 任务与子任务

全局任务始终来自 `task`，作为 72 个 token 的任务输入。子任务是独立监督目标，不替换全局任务。

开启 `train.use_subtask_loss` 时，需要：

- 数据 parquet 中每帧有整数列 `subtask_index`。
- `meta/subtasks.parquet` 用文本作为索引，索引名为 `subtask`，并含整数列 `subtask_index`。
- 表中行位置与 ID 一致，从 0 开始连续编号。

例如：

| subtask（索引文本） | subtask_index |
| --- | --- |
| 空字符串 | 0 |
| 抓取物体 | 1 |
| 将物体放入容器 | 2 |

未标注帧使用空文本对应的 ID，只训练动作，不计算语言 loss；不要用 `-1`。有标注帧的目标为当前时刻子任务，最多占 100 个 token，包含 EOS，padding 不计入 loss。

`mid_train.yaml` 和 `multi_mid_train.yaml` 默认开启子任务监督。关闭后不读取子任务标注。独立策略始终不读取任务或子任务。

## 归一化统计

先修改 `configs/norm_compute/own_robot.yaml` 的数据路径和机器人映射，再执行：

```bash
bash train.sh scripts/compute_norm_stats.py configs/norm_compute/own_robot.yaml
```

结果写入 `data.norm_path`。训练配置的 `data.norm_stats_file` 应指向它。计算时的 `train.chunk_size` 必须与训练一致。

| 参数 | 用途 | 建议起点 |
| --- | --- | --- |
| `data.data_ratio_for_norm_compute` | 随机抽取参与统计的样本帧比例，每帧仍读取完整动作 chunk | 正式统计用 `1.0` |
| `data.norm_merge_chunk_dim` | 相对动作是否合并 chunk 时间维，共享每个动作维度的统计 | `true` |

以 `[B,50,18]` 的 TCP 动作为例，`true` 产生 `[18]` 统计；`false` 产生 `[50,18]` 统计。state 始终按特征维统计，绝对动作始终合并时间维。共享统计让各未来步使用同一尺度，适合作为当前策略的默认配置。

更换数据、映射、动作块长度或统计方式后需重新计算。训练与部署必须使用同一套统计；部署包会自动携带训练所用的文件。

## 多数据集训练

编辑 `assets/data_path/multi_pretrain_datasets.txt`，每行一个机器人名和数据集路径：

```text
own_robot /data/task_a
own_robot /data/task_b
```

支持空行和 `#` 整行注释。启用的各数据集需使用一致的动作与状态语义、单位和维度。开启子任务监督时，每份数据都需提供上述子任务字段。

先计算合并统计：

```bash
bash train.sh scripts/compute_norm_stats.py configs/norm_compute/multi_pretrain.yaml
```

再选择训练入口：

```bash
bash train.sh tasks/train_action_policy.py configs/policy_action/multi_pretrain.yaml
bash train.sh tasks/train_vla.py configs/policy_vla/multi_mid_train.yaml
```

两条训练命令按需要分别执行，mid_train 仍需配置已训练的动作权重。后训练也可将 `data_name` 设为 `multi`、`train_path` 指向清单、`norm_stats_file` 指向合并统计。数据按帧拼接并随机采样，不额外设置数据集采样权重。

## 常见问题

- **缺少相机字段**：检查机器人 YAML 的原生字段是否与数据 metadata 一致。
- **找不到子任务**：补齐 `subtask_index` 和 `meta/subtasks.parquet`，或关闭 `train.use_subtask_loss`。
- **视频解码失败**：确认环境中的 FFmpeg / TorchCodec 可用；也可设置 `data.video_backend: pyav`。
- **动作维度不匹配**：同时核对机器人切片、`joints`、模型维度与统计文件。
