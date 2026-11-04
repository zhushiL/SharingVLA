# VLA

基于 Qwen3.5-2B 的机器人动作模型，提供两种可部署的策略：

| 策略 | 输入 | 适用场景 |
| --- | --- | --- |
| Action Policy | 左右腕部 RGB、机器人 state | 不需要语言指令的独立控制 |
| VLA | 历史与当前头部 RGB、左右腕部 RGB、state、任务文本 | 需要语言指令的控制 |

两种策略均输出动作 chunk。VLA 可在训练时监督当前子任务，推理时只生成动作。不使用触觉或未来图像预测。

## 快速开始

以下命令均在仓库根目录执行。需要 Linux、Conda 和支持 BF16 的 NVIDIA GPU。

### 1. 安装环境与下载权重

```bash
bash create_env.sh
conda activate shilinvla
python scripts/download_hf_model.py
```

权重默认下载到 `pretrained_models/Qwen3.5-2B`。安装中断后可用 `bash create_env.sh --resume` 继续；自定义环境名使用 `--env-name NAME`。

### 2. 准备数据

示例使用 `own_robot` 的双臂 TCP、夹爪映射。两种预训练配置当前使用多数据集清单，两种后训练配置使用 `../data/bi_rebot_09_pick_and_place_1023-annotated` 单数据集。使用自己的数据时，先检查 [数据指南](docs/data.md)，修改数据路径；按下方预训练流程，编辑 `assets/data_path/multi_pretrain_datasets.txt` 后计算合并统计：

```bash
bash train.sh scripts/compute_norm_stats.py configs/norm_compute/multi_pretrain.yaml
```

单数据集统计使用 `configs/norm_compute/own_robot.yaml`。统计配置和训练配置必须指向同一份数据，并使用相同的机器人映射、动作组成和 chunk 长度。

### 3. 训练独立策略

```bash
bash train.sh tasks/train_action_policy.py configs/policy_action/pretrain.yaml
```

未设置 `CUDA_VISIBLE_DEVICES` 和 `NPROC_PER_NODE` 时，启动脚本自动使用本机全部 GPU。指定 GPU 可在命令前加 `CUDA_VISIBLE_DEVICES=0,1`；首次检查可使用 `CUDA_VISIBLE_DEVICES=0` 并在命令末尾加 `--train.micro_batch_size 1`。完成独立策略训练后，可直接部署，也可用它初始化 VLA。四种训练流程和权重路径见 [训练指南](docs/training.md)。

### 4. 启动 WebSocket 服务

将路径中的步数替换为实际保存的 checkpoint：

```bash
python -m deploy.serve_policy --policy action_policy \
  --model_path outputs/action_pretrain/checkpoints/global_step_50000/hf_ckpt
```

服务默认使用端口 8006。观测字段、完整 VLA 部署、RTC 和客户端示例见 [部署指南](docs/deployment.md)。

也可以先启动离线网页控制台，再在页面选择并加载权重：

```bash
python -m deploy.serve_console
```

打开 `http://127.0.0.1:8080`。支持 Action Policy / VLA、图像与 state 预览、网卡 IP、VLA 任务更新和完整 chunk / RTC。操作步骤和客户端响应协议见[离线控制台说明](docs/deployment.md#离线网页控制台)。

## 训练选择

| 目标 | 入口 | 配置 | 子任务 loss | RTC |
| --- | --- | --- | --- | --- |
| 独立策略预训练 | `tasks/train_action_policy.py` | `configs/policy_action/pretrain.yaml` | 无 | 关 |
| 独立策略后训练 | `tasks/train_action_policy.py` | `configs/policy_action/post_train.yaml` | 无 | 开 |
| 完整 VLA 预训练 | `tasks/train_vla.py` | `configs/policy_vla/mid_train.yaml` | **默认开，可关闭** | 关 |
| 完整 VLA 后训练 | `tasks/train_vla.py` | `configs/policy_vla/post_train.yaml` | 默认关，可开启 | 开 |

训练配置仅保留上表四份 YAML。RTC、子任务 loss 和冻结设置由各自参数控制；每份 YAML 可独立编辑。单数据集与多数据集共用这四份配置，通过 `data` 字段切换；当前 `pretrain.yaml` 和 `mid_train.yaml` 使用多数据集清单，两份 `post_train.yaml` 使用 `own_robot` 单数据集。设置方式见 [数据指南](docs/data.md#多数据集训练)。

## 模型结构

左右腕部特征各为 1024 维，按左、右顺序拼为 2048 维。图像 self-attention 后，与投影到 2048 维的 state 做 cross-attention，再投影回 1024 维，作为动作噪声前方的环境 token。

动作网络使用 18 层全注意力，时间调制采用共享投影和每层偏移表。完整 VLA 使用两张头部图像和 72 个任务 token；开启子任务监督时，训练额外追加 100 个当前子任务 token。MoT 只连接图像和任务的 KV。推理仅运行一次 VLM，随后去噪复用 KV。

## 目录与文档

```text
configs/
  policy_action/       # 独立策略配置
  policy_vla/          # 完整 VLA 配置
  norm_compute/        # 共用归一化统计配置
  robot_configs/       # 原生字段映射
xense0vla/
  models/              # 模型、环境融合、MoT
  data/                # 数据读取与变换
  training/            # 训练逻辑与公共循环
  inference/           # 策略推理
  checkpoint/          # 保存与恢复
  distributed/         # 分布式训练
  optim/               # 优化器与调度器
  utils/               # 配置、统计、日志
tasks/                # 两个训练入口
scripts/              # 下载、统计、评估、测试
deploy/               # WebSocket 服务与客户端
docs/                 # 使用指南
tests/                # 回归测试
```

继续阅读：[数据指南](docs/data.md) · [训练指南](docs/training.md) · [配置参考](docs/configuration.md) · [部署指南](docs/deployment.md)。
