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

连续控制时保持连接，在新连接或开始新 episode 时调用 `reset`。同一 episode 中更新任务不需要 reset，保留头部历史和 RTC 已承诺动作。结束后关闭连接。

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

## 离线网页控制台

先启动页面，再在浏览器中选择权重；启动阶段不导入模型、不占用模型显存。使用已安装本仓库依赖的 Python 环境，无需 npm 或外网：

```bash
python -m deploy.serve_console
```

- 页面：`http://127.0.0.1:8080`。
- 机器人连接：`ws://<server局域网IP>:8006`，**仅在模型加载成功后开放**。
- 页面右上角显示本机局域网 IP，展开后列出各网卡的网页和 client 地址，支持复制。
- 默认绑定 `0.0.0.0`，用于可信局域网。只在本机访问时加 `--host 127.0.0.1`。
- 可用 `--web-port 8080 --port 8006 --preview-fps 5` 调整端口和预览编码频率。
- 默认可浏览和加载服务器本机任意有权限读取的目录（含隐藏目录）。`--checkpoint-root` 可重复指定选择器的快捷目录，省略时从当前工作目录开始。若需要限定范围，可加 `--restrict-checkpoints`，此时仅允许这些根目录内的路径。

权重位于其他目录时，直接在页面选择或输入绝对路径即可，无需修改启动命令。若希望添加常用目录快捷入口，可选用：

```bash
python -m deploy.serve_console \
  --checkpoint-root outputs \
  --checkpoint-root /mnt/checkpoints
```

页面依次操作：

1. 在服务器桌面通过 `http://127.0.0.1:8080` 访问时，点击文件夹图标打开系统目录选择器，选好后自动检查权重（Linux 需已安装 `zenity` 且服务能访问桌面会话）。远程访问、无桌面或选择器不可用时自动使用网页目录选择；也可点击“路径选择”或直接输入绝对路径（支持 `~`）。选择完整 `hf_ckpt`，不会上传权重文件。
2. 检查配置。页面自动识别 Action Policy / VLA、可选机器人、chunk 大小和 RTC 训练能力。不能选择 `hf_ckpt/action_policy` 训练迁移包。
3. 选择机器人、BF16/FP32，按需开启 `torch.compile`，点击加载。首个真实请求可能触发编译，首次推理延迟不代表稳定运行延迟。
4. VLA 在页面填写并应用 task；Action Policy 不读取文本，页面禁用任务编辑。Task 默认显示“待应用”“等待下一轮推理”等状态；展开“版本详情”可查看提交、推理和返回版本。“新任务已用于推理”仅确认生成动作块所用的任务，不代表机器人已执行完成。
5. Client 连接并调用 `reset(metadata["robot_name"])`，随后连续发送观测。页面中间横排显示三路图像，最左侧显示 TCP XYZ 及最近 100 次预览采样的小曲线，最右侧为模型设置，任务编辑位于实时观测下方的中间主栏。头部当前/历史帧用下拉框切换。
6. 需要更换权重时，先停止并断开 client，然后在页面卸载、重新加载。加载失败可以修正目录或设置后重试。

控制台强制 `chunk_ret=True, use_length=-1`，每个正常推理请求生成完整动作块。模型加载、reset 与推理通过同一个 worker 串行执行；预览编码运行在独立 worker，以最新快照覆盖旧快照，不积压视频帧。关闭浏览器不会停止模型服务。

### 统一观测与任务来源

两种 policy 可以接收同样的完整观测：头部历史/当前两帧、左右腕部 RGB、state，以及可选 RTC 前缀。**以页面/metadata 中的 `observation_schema` 为准使用相机与 state 的原生字段名**，不要从其他权重包复制字段名。模型继续使用 checkpoint 的预处理；Action Policy 只消费其腕部图像和 state，额外头部帧仍可用于页面展示。

在控制台模式下 `metadata["task_source"] == "server"`，client 的 `task` 被忽略。VLA 每次推理开始时固定任务文本及版本，页面更新 task 只影响下一次开始的推理，**不 reset、不重载模型、不清除头部历史**。正在计算的旧任务响应正常返回，用其版本识别即可；不要因页面更新就无条件清空 RTC 队列或丢掉在途结果。

VLA 尚未设置 task 时，服务仍接收并展示观测，但返回：

```python
{"action": None, "status": "task_required", "console": {...}}
```

此时没有可执行的新动作，client 应在控制器定义的等待状态下重试。该响应本身不是停止机器人的指令。设置 task 后，后续请求正常推理。正常响应保留原来的动作字段，并附加 `status="ok"`、`console` 元数据和 `server_timing`，原来的 MessagePack/WebSocket 编码不变。

```python
from deploy.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host="192.168.1.100", port=8006)
metadata = client.get_server_metadata()
client.reset(metadata["robot_name"])
try:
    # observation: 按 metadata["observation_schema"] 采集的完整观测
    request = {
        **observation,
        "request_id": "req-42",          # 可选；建议每个请求唯一
        "observation_tick": 1200,         # 可选；观测对应的 client 控制步
        "observation_timestamp": 12.34,  # 可选；client 自己的时间基准
    }
    # 非空 RTC 前缀沿用现有协议；零长度时同时省略这两个字段。
    if len(committed_actions):
        request["rtc_action_prefix"] = committed_actions  # float32 [delay, action_dim]
        request["rtc_prefix_length"] = len(committed_actions)
    response = client.infer(request)
    if response.get("status") == "task_required":
        print("等待页面设置任务")
    else:
        actions = response["action"]
        context = response["console"]
        # context: session_id, observation_epoch, sequence, task_version, task, received_at，
        # 以及回传的 request_id / observation_tick / observation_timestamp。
        # 交给控制器按 observation_tick 与执行进度接续，不能从动作块第 0 步重放。
finally:
    client.close()
```

RTC 的已承诺前缀在切换任务时保持，新的 task 条件化生成后续动作。Client 仍负责控制时钟、动作队列、延迟估计、结果过期处理及超时保持；本控制台不会驱动机器人。若请求等待期间控制器执行超过有效覆盖范围，不能重放过期动作，需要控制器按自己的策略重新请求或保持。建议每次只保留一个在途请求，避免排队使观测和 RTC 前缀过期。

页面显示的是 **模型收到的观测**，不是独立相机视频流。图片和 state 带同一个观测序号；页面仅按 checkpoint 中的 TCP 映射展示 XYZ，模型仍使用完整 state；超过 2 秒未刷新标记过期。已提交、当前推理、最近返回任务版本分开显示，它们不等同于实际已执行版本；当前 client 没有回传执行确认。

### 验证控制台

以下测试使用 CPU 假模型和临时 checkpoint，覆盖 HTTP、真实 WebSocket 往返、推理期间更新任务、RTC 前缀传递、两种 policy、加载失败恢复及目录边界，不向机器人发送动作：

```bash
python -m unittest discover -s tests -v
```

真实权重的控制台端到端测试使用：

```bash
python scripts/test_console_gpu.py \
  --vla /path/to/vla/hf_ckpt \
  --action /path/to/action_posttrain/hf_ckpt \
  --data-path /path/to/recorded/lerobot_dataset \
  --report outputs/console_gpu_report.json
```

测试在本机临时端口启动独立控制台，通过 HTTP 加载、实际 `WebsocketClientPolicy` 连接和 CUDA 推理验证两种策略。两份权重需使用同一机器人映射并支持 RTC，录制数据的原生字段需与权重内配置一致。两者发送同一份完整观测；测试覆盖完整 chunk、0 到训练上限的 RTC 前缀、真实推理期间更新 task、下一轮文本 token 变化、头部历史保留、图像/state 预览、reset 和卸载。

默认每档 delay 预热 3 次、测量 20 次，可以通过 `--warmup` 和 `--repeats` 调整。报告分别记录加载耗时、首次请求耗时，以及预热后的服务端/本机 WebSocket 往返延迟 P50/P95 和原始样本。首次请求不计入稳定延迟。

对比编译开关时，用相同权重、数据和参数运行上述命令两次，分别保存报告；第二次加 `--use-compile`。默认 `torch.compile` 在第一次推理时才真正编译，因此“加载成功”不代表编译已完成。测试的单请求超时为 900 秒（可用 `--request-timeout` 修改），报告还记录切换 task 后是否捕获新计算图，以及首次新任务的延迟。编译日志中可能出现 graph break；报告中的 `compiler_counters` 可用于区分实际编译与完全 eager 执行。

可将 `OMP_NUM_THREADS=4` 同时用于两次测量，并在编译测试设置 `TORCHINDUCTOR_COMPILE_THREADS=4` 控制 CPU 编译并行度。使用独立 `TORCHINDUCTOR_CACHE_DIR` 和 `TRITON_CACHE_DIR` 可测量首次缓存构建成本；重复使用该目录测得的是缓存启动时间。报告中的延迟仅代表当前机器、固定录制观测和测试负载，不包含机器人控制器、真实局域网或任务完成时间。

缺少权重时，可通过真实训练入口生成验证包：

```bash
python scripts/test_training_entries_gpu.py \
  --output-dir outputs/console_validation \
  --parameter-dtype bfloat16 \
  --data-path /path/to/recorded/lerobot_dataset \
  --robot-config-root /path/to/matching/robot_configs
```

四个阶段各运行一个优化步，生成 `action_posttrain/checkpoints/global_step_1/hf_ckpt` 和 `post_train/checkpoints/global_step_1/hf_ckpt` 等目录。**这些是验证加载和协议的权重，不是可用于真机任务的训练成果。** `--robot-config-root` 可为录制数据提供独立映射，避免修改当前机器人的源配置。

所有运行资源来自本地；控制台强制 Hugging Face 离线模式，不会在缺失资源时自动下载。完整 VLA 的图像处理配置既支持独立 `preprocessor_config.json`，也支持嵌入 `processor_config.json` 的导出格式。
