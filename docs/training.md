# 训练指南

先完成 [环境安装](../README.md#快速开始) 和 [数据准备](data.md)。训练配置仅保留下述四份，单数据集和多数据集通过各配置的 `data` 字段切换。以下命令从仓库根目录执行。根据显存调整 `train.micro_batch_size`，并把权重路径中的步数替换为实际输出。

## 1. 独立 Action Policy 预训练

独立策略以 `spatial_encoder` 从双腕当前帧提取的空间 token 为条件。预训练时，用 DA3-LARGE 权重初始化其中的 Transformer 主干，并以 `train.spatial_encoder_lr` 微调。编码器中的下采样卷积、通道投影以及动作网络其余参数均从头训练，使用 `train.lr`。

```bash
bash train.sh tasks/train_action_policy.py configs/policy_action/pretrain.yaml
```

当前 `pretrain.yaml` 使用多数据集清单和合并统计；准备方式见 [数据清单与采样](data.md#数据清单与采样)。单数据集训练可直接修改同一份配置的 `data` 字段。

需要修改的主要字段：

- `model.spatial_encoder_path` 指向用于初始化 Transformer 主干的 DA3-LARGE checkpoint。
- `data.train_path`、`data.norm_stats_file`：数据及归一化统计。
- `train.output_dir`、`train.max_steps`：保存目录与训练步数。

训练完成后，整个 `hf_ckpt` 可用于部署；其中的 `action_policy/` 用于后续训练初始化。

## 2. 独立 Action Policy 后训练

加载预训练导出的完整动作策略，使用一个学习率更新所有参数，默认开启 RTC。

```bash
bash train.sh tasks/train_action_policy.py configs/policy_action/post_train.yaml \
  --model.action_policy_path outputs/action_pretrain/checkpoints/global_step_50000/hf_ckpt/action_policy
```

动作包包含训练后的完整 `spatial_encoder`、18 个动作层及动作网络的输入输出投影，可以直接部署。

## 3. 完整 VLA 预训练：mid_train

加载完整 Qwen 权重和已训练的 action policy，联合学习任务条件下的动作与当前子任务。

```bash
bash train.sh tasks/train_vla.py configs/policy_vla/mid_train.yaml \
  --model.action_policy_path outputs/action_pretrain/checkpoints/global_step_50000/hf_ckpt/action_policy
```

当前 `mid_train.yaml` 的 `data_name` 为 `multi`，使用 `assets/data_path/multi_pretrain_datasets.txt` 和对应合并统计。单数据集训练时，按 [数据配置](data.md#数据配置) 修改 `data` 字段。

**mid_train 默认开启子任务语言 loss**。训练数据需有子任务标注；若只训练动作，可显式关闭：

```bash
bash train.sh tasks/train_vla.py configs/policy_vla/mid_train.yaml \
  --train.use_subtask_loss false
```

`mid_train.yaml` 默认冻结预训练动作策略的大部分参数（`train.freeze_action_policy: true`），但开启 `train.unfreeze_mot_layers: true`，让动作侧接收 Qwen KV 的 6 个 MoT 耦合层（第 3、6、9、12、15、18 个完整 block）保持可训练并使用 `train.lr`；Qwen 语言模型与头部视觉塔共用 `train.vlm_lr`。如需完全冻结全部动作层，可显式传 `--train.unfreeze_mot_layers false`。

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
| 只解冻接收 MoT KV 的动作层 | VLA 设置 `train.freeze_action_policy: true` 和 `train.unfreeze_mot_layers: true` |
| 调整模块学习率 | 独立预训练用 `train.spatial_encoder_lr`，完整 VLA 用 `train.vlm_lr` |
| 增加有效 batch | 增加 `train.gradient_accumulation_steps` |
| 减少显存占用 | 先减小 `train.micro_batch_size`，保留梯度检查点 |
| 调整头部历史 | `train.history_memory` 设历史张数，`train.history_stride` 设间隔帧数 |
| 多数据集训练 | 在对应阶段配置中设置 `data_name: multi`、数据集清单和合并统计，见 [数据清单与采样](data.md#数据清单与采样) |

预训练样例使用 `1e-5` 微调 `spatial_encoder` 的 Transformer 主干，使用 `1e-4` 训练其余参数。独立后训练样例统一使用 `1e-5`。

默认训练使用 FP32 参数与 AdamW 状态、BF16 autocast。`train.parameter_dtype: bfloat16` 可进一步减少显存，但会改变参数和优化器状态的精度。部署权重统一导出为 BF16。

未设置 `CUDA_VISIBLE_DEVICES` 和 `NPROC_PER_NODE` 时，`train.sh` 自动使用本机全部 GPU。指定可见 GPU 后，默认按可见数量启动进程；显式设置 `NPROC_PER_NODE` 时以该值为准：

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
| `hf_ckpt/action_policy/` | 独立策略训练时导出的完整动作包，可单独部署，也可用于动作后训练和 VLA mid_train 初始化 |

在同一输出目录断点续训：

```bash
bash train.sh tasks/train_vla.py configs/policy_vla/mid_train.yaml \
  --train.enable_resume true
```

也可用 `train.load_checkpoint_path` 指定 DCP checkpoint。续训从本次训练保存的模型配置和 DCP 恢复权重，原始 DA3 和 Qwen 权重目录可以移走。若更换数据或采样顺序，可设置 `train.resume_dataloader_state: false`。从部署包开始后训练时，训练器会创建新的优化器。

## 验证训练入口

运行回归测试：

```bash
python -m unittest discover -s tests -v
```

以下命令使用 DA3-LARGE 和 Qwen3.5-2B 的预训练权重与示例数据，依次验证四种训练流程的单步优化、导出和重载推理。

```bash
python scripts/test_training_entries_gpu.py \
  --parameter-dtype bfloat16 --output-dir outputs/training_entries_gpu
```

使用新的输出目录。复查已有产物时加 `--verify-only`。这是接口与数值检查，不用于判断训练是否收敛。
