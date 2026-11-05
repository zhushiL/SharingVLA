# 数据指南

使用 LeRobot v3 数据集。训练样本包含相机图像、机器人状态和动作；完整 VLA 还需要全局任务文本，启用子任务监督时需要额外标注。

## 数据配置

单数据集训练：

```yaml
data:
  data_name: own_robot
  train_path: /data/task_a
  robot_config_root: configs/robot_configs
  norm_stats_file: assets/norm_stats/own_robot.json
```

`data_name` 对应机器人映射文件 `configs/robot_configs/<data_name>.yaml`。`train_path` 可填写本地 LeRobot 目录或 Hugging Face 数据集 ID；相对路径以仓库根目录为基准。

多数据集训练将 `data_name` 设为 `multi`，`train_path` 指向数据清单：

```yaml
data:
  data_name: multi
  train_path: assets/data_path/multi_robot_datasets.txt
  robot_config_root: configs/robot_configs
  norm_stats_file: assets/norm_stats/multi_robot.json
```

## 机器人字段映射

机器人 YAML 定义原生字段到模型输入的映射。`own_robot.yaml` 的 state/action 均为 20 维：

| 原生切片 | 内容 |
| --- | --- |
| `0:9` | 左 TCP：xyz + 6D rotation |
| `9:10` | 左夹爪 |
| `10:19` | 右 TCP：xyz + 6D rotation |
| `19:20` | 右夹爪 |

映射后，左右 TCP 共 18 维、夹爪共 2 维。TCP 动作转换为相对当前状态的局部位姿，夹爪保持绝对值。

| 逻辑相机 | own_robot 原生字段 |
| --- | --- |
| `camera_top` | `observation.images.head` |
| `camera_wrist_left` | `observation.images.left_wrist` |
| `camera_wrist_right` | `observation.images.right_wrist` |

独立 action policy 使用左右腕相机；完整 VLA 使用头部、左右腕相机，并读取 `t-head_history_offset` 与 `t` 的头部图像。历史不足时使用当前 episode 首帧。

`robot1.yaml`、`robot2.yaml`、`robot3.yaml` 分别适配夹爪位于向量末尾、腕相机带重复前缀、全局相机名为 `top` 等字段差异。各数据集使用哪份映射由清单第一列指定。

接入其他机器人时，核对原生字段、切片、单位和动作语义，并同步调整 `data.joints`、`data.norm_type` 及模型的动作/状态维数。混合训练的数据集应输出一致的逻辑空间。

## 数据清单与采样

每行依次填写机器人配置名、采样比例、数据集路径：

```text
own_robot sampling_ratio=1.00 /data/task_a
robot1 sampling_ratio=0.41 Xense/whiteboard_inspect_1023
```

`sampling_ratio` 取值为 `[0, 1]`：`1.00` 使用全部样本起点，`0` 禁用该行，省略时默认为 `1.0`。支持空行和 `#` 注释；含空格或 `#` 的路径用引号包围。

采样按比例固定、无放回地选择样本起点，训练时打乱顺序。每个起点对应的连续动作块和历史图像仍从原始数据读取。`data.sampling_seed` 默认为 `42`，训练与归一化统计应保持一致。

| 清单 | 用途 |
| --- | --- |
| `assets/data_path/multi_pretrain_datasets.txt` | 通用多数据集配置 |
| `assets/data_path/multi_umi_datasets.txt` | TacVerse 腕视角数据，用于独立 action policy |
| `assets/data_path/multi_robot_datasets.txt` | Xense 机器人数据，包含全局与双腕视角 |

清单中的比例可逐行修改。机器人映射、源数据路径及采样比例以清单和对应 YAML 为准。

## 归一化统计

训练前使用 `scripts/compute_norm_stats.py` 计算统计。单数据集使用 `configs/norm_compute/own_robot.yaml`，多数据集使用 `configs/norm_compute/multi_pretrain.yaml`。

例如，为机器人清单计算合并统计：

```bash
bash train.sh scripts/compute_norm_stats.py configs/norm_compute/multi_pretrain.yaml \
  --data.train_path assets/data_path/multi_robot_datasets.txt \
  --data.robot_name own_robot,robot1,robot2,robot3 \
  --data.norm_path assets/norm_stats/multi_robot.json
```

`data.robot_name` 需包含要参与统计的全部映射名称。统计沿用清单的采样比例；`data.data_ratio_for_norm_compute` 可进一步抽样，正式统计建议为 `1.0`。`data.norm_merge_chunk_dim: true` 让相对动作在不同未来步共享统计。

训练配置的 `data.norm_stats_file` 指向统计输出，`train.chunk_size` 与统计时一致。更换数据、映射、采样比例或统计设置后需重新计算；部署使用训练时的同一份统计。

## 任务与子任务

完整 VLA 的全局任务来自 LeRobot 任务元数据。独立 action policy 不读取任务或子任务文本。

启用 `train.use_subtask_loss` 时，需要：

- 数据 parquet 中每帧有整数列 `subtask_index`。
- `meta/subtasks.parquet` 以子任务文本为索引，索引名为 `subtask`，并含整数列 `subtask_index`。
- 子任务 ID 从 0 连续编号，且与表中行位置一致。未标注帧使用空字符串对应的 ID，不使用 `-1`。

未标注帧只训练动作。没有子任务标注的数据应设置 `train.use_subtask_loss: false`。`configs/policy_vla/mid_train.yaml` 默认开启该项，`configs/policy_vla/post_train.yaml` 默认关闭。
