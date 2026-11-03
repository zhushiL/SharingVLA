# 训练指南

先完成 [环境安装](../README.md#快速开始) 和 [数据准备](data.md)。以下命令从仓库根目录执行。根据显存调整 `train.micro_batch_size`，并把权重路径中的步数替换为实际输出。

## 1. 独立 Action Policy 预训练

只使用左右腕部图像和 state，不构建语言 backbone，也不读取任务或子任务。腕部图像经过 Qwen 视觉编码器，默认冻结视觉编码器，只训练 action policy。

```bash
bash train.sh tasks/train_action_policy.py configs/policy_action/pretrain.yaml
```

需要修改的主要字段：

- `model.vlm_path`：Qwen3.5-2B 目录。
- `data.train_path`、`data.norm_stats_file`：数据及归一化统计。
- `train.output_dir`、`train.max_steps`：保存目录与训练步数。

训练完成后，整个 `hf_ckpt` 可用于部署；其中的 `action_policy/` 用于后续训练初始化。

## 2. 独立 Action Policy 后训练

加载 Qwen 视觉配置与独立策略迁移包，默认冻结视觉编码器并开启 RTC。

```bash
bash train.sh tasks/train_action_policy.py configs/policy_action/post_train.yaml \
  --model.action_policy_path outputs/action_pretrain/checkpoints/global_step_50000/hf_ckpt/action_policy
```

迁移包必须包含动作网络和 `vision_encoder/`。两部分权重都会恢复，确保继续使用预训练时的视觉特征。

## 3. 完整 VLA 预训练：mid_train

加载完整 Qwen 权重和已训练的 action policy，联合学习任务条件下的动作与当前子任务。

```bash
bash train.sh tasks/train_vla.py configs/policy_vla/mid_train.yaml \
  --model.action_policy_path outputs/action_pretrain/checkpoints/global_step_50000/hf_ckpt/action_policy
```

**mid_train 默认开启子任务语言 loss**，包括多数据集示例 `multi_mid_train.yaml`。训练数据需有子任务标注；若只训练动作，可显式关闭：

```bash
bash train.sh tasks/train_vla.py configs/policy_vla/mid_train.yaml \
  --train.use_subtask_loss false
```

默认冻结视觉编码器。`train.vlm_lr` 控制语言模型及解冻后的视觉编码器，`train.action_policy_lr` 控制动作网络。

## 4. 完整 VLA 后训练

从 mid_train 导出的整个模型初始化，默认开启 RTC、关闭子任务语言 loss：

```bash
bash train.sh tasks/train_vla.py configs/policy_vla/post_train.yaml \
  --model.model_path outputs/mid_train/checkpoints/global_step_50000/hf_ckpt
```

需要子任务监督时加 `--train.use_subtask_loss true`。只希望训练 VLM 时，设置 `--train.freeze_action_policy true`；动作 loss 仍可通过 MoT 传播到 VLM。

## 常用调整

| 需求 | 配置 |
| --- | --- |
| 解冻视觉编码器 | 所有训练配置都可设 `train.freeze_vision_encoder: false` |
| 调整模块学习率 | `train.action_policy_lr`；完整 VLA 另有 `train.vlm_lr` |
| 增加有效 batch | 增加 `train.gradient_accumulation_steps` |
| 减少显存占用 | 先减小 `train.micro_batch_size`，保留梯度检查点 |
| 调整头部历史 | 完整 VLA 的 `train.head_history_offset`，单位为数据帧 |
| 多数据集训练 | 使用 `multi_pretrain.yaml` 或 `multi_mid_train.yaml`，见 [数据指南](data.md#多数据集训练) |

模块学习率未设置时使用 `train.lr`。独立策略解冻后的视觉编码器也使用 `train.lr`。

默认训练使用 FP32 参数与 AdamW 状态、BF16 autocast。`train.parameter_dtype: bfloat16` 可进一步减少显存，但会改变参数和优化器状态的精度。部署权重统一导出为 BF16。

指定可见 GPU 后，`train.sh` 会按可见数量启动进程：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash train.sh tasks/train_vla.py configs/policy_vla/mid_train.yaml
```

## RTC 设置

后训练示例使用以下设置：

```yaml
rtc_training:
  enabled: true
  max_delay_steps: 8
  delay_sampling: uniform
```

训练时对每个样本从整数区间 `[0,8]` 等概率采样 delay，以前缀动作作为条件，仅计算剩余动作的 loss。不设置采样权重。`max_delay_steps` 必须小于 `chunk_size`。部署侧需要自行提供已承诺执行的动作前缀，见 [部署指南](deployment.md#rtc-与-delay-测试)。

## 保存与恢复

训练输出位于 `train.output_dir/checkpoints/global_step_N/`：

| 内容 | 用途 |
| --- | --- |
| DCP 训练状态 | 恢复模型、优化器、调度器、随机数和 DataLoader 进度 |
| `hf_ckpt/` | 完整部署包，可整体复制 |
| `hf_ckpt/action_policy/` | 独立训练导出的迁移包，用于独立后训练和 mid_train |

在同一输出目录断点续训：

```bash
bash train.sh tasks/train_vla.py configs/policy_vla/mid_train.yaml \
  --train.enable_resume true
```

也可用 `train.load_checkpoint_path` 指定 DCP checkpoint。若刻意更换数据或采样顺序，设置 `train.resume_dataloader_state: false`。后训练的权重初始化会创建新的优化器，与断点续训用途不同。

## 验证训练入口

运行回归测试：

```bash
python -m pip install pytest
python -m pytest tests -q
```

用真实 Qwen3.5-2B 和示例数据执行四种训练流程，每种做一步优化、导出和重载推理：

```bash
python scripts/test_training_entries_gpu.py \
  --parameter-dtype bfloat16 --output-dir outputs/training_entries_gpu
```

使用新的输出目录。复查已有产物时加 `--verify-only`。这是接口与数值检查，不用于判断训练是否收敛。
