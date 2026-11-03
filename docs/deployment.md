# WebSocket 部署指南

两种策略共用服务入口。部署需要 CUDA GPU 和已安装本仓库的环境。以下命令均从仓库根目录运行。

## 选择部署包

使用训练导出的整个 `global_step_N/hf_ckpt/`，复制时保持目录完整：

```text
hf_ckpt/
  config.json
  model*.safetensors
  vla_cli.yaml
  vlm_backbone/
  runtime_assets/
    norm_stats.json
    robot_configs/own_robot.yaml
```

独立训练还会导出 `hf_ckpt/action_policy/`，它用于训练迁移，不能作为服务的 `model_path`。服务使用本版本导出的配置与资源，不自动转换其他格式。

## 启动服务

独立 Action Policy：

```bash
python -m deploy.serve_policy --policy action_policy \
  --model_path outputs/action_pretrain/checkpoints/global_step_50000/hf_ckpt
```

完整 VLA：

```bash
python -m deploy.serve_policy --policy vla --port 8007 \
  --model_path outputs/mid_train/checkpoints/global_step_50000/hf_ckpt
```

将步数和目录替换为实际训练输出。后训练模型也使用同一入口。默认端口为 8006，返回完整动作 chunk；多数据集模型需额外指定 `--robot_name own_robot`。

只允许本机连接时加 `--host 127.0.0.1`。每个服务实例支持一个活动客户端；健康检查地址为 `http://127.0.0.1:8006/healthz`。

## 准备观测

发送机器人原生字段，服务负责图像处理、归一化和动作还原。

| 字段 | 格式 | 策略 |
| --- | --- | --- |
| `observation.images.left_left_wrist` | RGB `uint8 [H,W,3]` 或 `[3,H,W]` | 两种 |
| `observation.images.right_right_wrist` | 同上 | 两种 |
| `observation.state` | `float32 [20]` | 两种 |
| `observation.images.head` | 两帧 RGB，`uint8 [2,H,W,3]` 或 `[2,3,H,W]` | 仅 VLA |
| `task` | 全局任务字符串 | 仅 VLA |

state 顺序为左 TCP 9 维、左夹爪 1 维、右 TCP 9 维、右夹爪 1 维。TCP 为 xyz + 6D rotation，单位与坐标系必须和训练一致。OpenCV 采集的 BGR 图像需先转为 RGB。

VLA 的两帧头部图像依次为历史帧、当前帧。历史间隔使用 checkpoint 的 `head_history_offset`，默认 30 个数据帧；缓冲区不足时使用最早帧。建议客户端按相机帧维护缓冲区并显式发送两帧。仅发送当前帧时，服务按请求次数维护历史，无法保证与训练的帧间隔一致。

## 客户端示例

以下函数接收已经采集好的原生 observation，通过 WebSocket 返回一个动作块：

```python
from deploy.websocket_client_policy import WebsocketClientPolicy


def predict_chunk(observation, port=8006):
    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    try:
        client.reset("own_robot")
        return client.infer(observation)["action"]
    finally:
        client.close()
```

连续控制时保持连接，仅在新连接、开始新 episode 或切换任务时调用 `reset`。结束后关闭连接。

独立策略的 observation：

```python
import numpy as np

observation = {
    "observation.images.left_left_wrist": np.asarray(left_rgb, dtype=np.uint8),
    "observation.images.right_right_wrist": np.asarray(right_rgb, dtype=np.uint8),
    "observation.state": np.asarray(state, dtype=np.float32),
}
```

完整 VLA 另外加入头部图像与任务文本，连接上例的 8007 端口：

```python
observation["observation.images.head"] = np.stack([head_history, head_current]).astype(np.uint8)
observation["task"] = "将物体放入容器"
```

示例配置返回 `float32 [50,20]` 的原生动作，已反归一化并恢复绝对位姿，不要再次处理。VLA 推理不接收子任务，也不预测子任务。

需要部分动作块时，可在服务命令增加 `--use_length N`；需要逐步返回时用 `--chunk_ret false`。RTC 前缀测试使用默认完整 chunk 返回方式。

## RTC 与 delay 测试

RTC 能力由 checkpoint 的 `rtc_training.enabled` 决定。两种后训练示例默认开启；服务 metadata 的 `prefix_conditioned` 和 `rtc_training_max_delay_steps` 可用于检查。

控制端将从当前观测时刻开始、在推理延迟期间已承诺执行的动作作为前缀发送：

```python
prefix = np.asarray(committed_actions, dtype=np.float32)
request = {
    **observation,
    "rtc_action_prefix": prefix,
    "rtc_prefix_length": int(prefix.shape[0]),
}
actions = client.infer(request)["action"]
```

前缀形状为 `[delay,20]`，delay 不得超过训练上限。返回块保留这些原生前缀动作，控制器应从尚未执行的位置继续。服务不负责估计 delay 或维护机器人执行队列。

启动服务后，可用录制数据验证前缀并测量延迟。以下 30 Hz 应改为实际动作执行频率：

```bash
python scripts/test_delay.py \
  --data-path ../data/bi_rebot_09_pick_and_place_1023-annotated \
  --control-hz 30 --output outputs/delay/action_policy.json
```

测试 VLA 时增加 `--port 8007`。默认逐档测试 delay=0 到训练最大值，每档预热 2 次、测量 20 次；可用 `--delays 0 2 8`、`--repeats`、`--sample-index` 调整。非 RTC 模型只测试 delay=0。

也可使用包含上述观测字段的 NPZ 文件：

```bash
python scripts/test_delay.py --observation recordings/observation.npz \
  --control-hz 30 --output outputs/delay/recorded.json
```

NPZ 保存 RGB、state，以及 VLA 所需的头部帧和字符串 task，不含 action 标签、RTC 字段或 pickle 对象数组。

报告包含前缀校验、端到端与服务端延迟的 P50/P95/P99，以及 P95 对应的控制步数。端到端时间包含序列化、网络、预处理、GPU 推理和响应解析。测试不会向机器人驱动发送动作。

## 离线轨迹评估

使用录制轨迹比较预测动作与标签，生成误差和曲线：

```bash
python scripts/open_loop_eval.py \
  --model_path outputs/post_train/checkpoints/global_step_4000/hf_ckpt \
  --robo_name own_robot \
  --data_path ../data/bi_rebot_09_pick_and_place_1023-annotated \
  --traj_ids 0 1 --use_bf16 --video-backend pyav \
  --save_plot_path outputs/open_loop
```

独立策略使用相同命令，替换模型路径即可。`--max_infer_time` 控制每个 episode 的块推理次数，`--use_length` 控制每块使用的步数。离线误差、delay 测试均不能替代闭环任务成功率评估。
