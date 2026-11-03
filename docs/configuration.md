# 配置参考

从 `configs/policy_action/` 或 `configs/policy_vla/` 选择完整 YAML。配置分为 `model`、`data`、`train`，可直接编辑，也可用命令行覆盖单个字段：

```bash
bash train.sh tasks/train_vla.py configs/policy_vla/mid_train.yaml \
  --train.micro_batch_size 1 --train.vlm_lr 1e-6
```

## 权重来源

| 参数 | 含义 |
| --- | --- |
| `model.vlm_path` | 本地 Qwen3.5-2B 目录；独立策略必填，VLA 从 Qwen 初始化时使用 |
| `model.action_policy_path` | 独立策略导出的 `hf_ckpt/action_policy/`；独立预训练可不填，独立后训练和 mid_train 需填写 |
| `model.model_path` | 完整 VLA 的 `hf_ckpt/`；与 `vlm_path`、`action_policy_path` 互斥 |

模型类型由训练入口决定。完整模型部署包必须包含 `config.json`、模型权重、`vla_cli.yaml`、`vlm_backbone/` 和 `runtime_assets/`；动作迁移包必须包含 `vision_encoder/`。

## 数据与动作

| 参数 | 含义 |
| --- | --- |
| `data.train_path` | 单数据集目录，或多数据集清单 |
| `data.data_name` | 机器人映射名，或 `multi` |
| `data.norm_stats_file` | 训练使用的归一化统计 |
| `data.cameras` | 独立策略为左腕、右腕；VLA 为头部、左腕、右腕 |
| `data.img_size` | 当前固定为 256 |
| `data.video_backend` | `torchcodec` 或 `pyav` |
| `train.action_dim` | 实际动作维度 |
| `train.max_action_dim` / `train.max_state_dim` | 模型动作 / 状态维度，包含预留 padding |
| `train.chunk_size` | 每次生成的动作步数，示例为 50 |
| `train.num_steps` | 动作去噪步数，示例为 5 |

相机与原生数据字段映射、统计计算见 [数据指南](data.md)。

## 学习率、冻结与监督

| 参数 | 含义 |
| --- | --- |
| `train.lr` | 基础学习率，也是模块学习率的回退值 |
| `train.action_policy_lr` | 动作网络学习率 |
| `train.vlm_lr` | 完整 VLA 的语言模型及解冻后的视觉编码器学习率 |
| `train.freeze_vision_encoder` | 默认 true；所有训练配置都可设为 false |
| `train.freeze_action_policy` | 完整 VLA 可设为 true，默认 false |
| `train.use_subtask_loss` | 仅完整 VLA；**mid_train 示例默认 true**，后训练示例默认 false |
| `train.language_loss_weight` / `train.action_loss_weight` | 两个损失的权重，示例均为 1.0 |
| `train.head_history_offset` | 完整 VLA 的头部历史帧偏移，默认 30，不得为负 |
| `train.tokenizer_max_length` / `train.subtask_max_length` | 当前固定为 72 / 100 |

独立策略解冻后的视觉编码器使用 `train.lr`。独立入口不接受语言学习率、头部历史或子任务监督参数。

## 训练资源与保存

| 参数 | 含义 |
| --- | --- |
| `train.micro_batch_size` | 每张 GPU 每次前向的样本数 |
| `train.gradient_accumulation_steps` | 梯度累积次数 |
| `train.parameter_dtype` | 默认 `float32`；可显式选 `bfloat16` 以减少存储显存 |
| `train.enable_gradient_checkpointing` | 通过重计算减少激活显存，示例开启 |
| `data.num_workers` | 每进程的数据加载 worker 数 |
| `train.output_dir` | 训练输出目录 |
| `train.max_steps` / `train.save_steps` | 总优化步数 / checkpoint 保存间隔 |
| `train.enable_resume` | 从输出目录恢复 DCP 训练状态 |
| `train.use_wandb` | 是否上传训练日志，默认关闭 |

## RTC

`train.rtc_training.enabled` 显式控制 RTC。两种后训练示例设为 true，预训练和 mid_train 示例设为 false。`max_delay_steps` 默认为 8 的后训练示例会均匀采样整数 delay=0～8；上限必须小于 `chunk_size`。

RTC、子任务监督和冻结参数各自独立。权重路径不会自动改变这些开关。训练命令和保存恢复方式见 [训练指南](training.md)。
