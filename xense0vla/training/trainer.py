import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import wandb
from tqdm import trange

from xense0vla.checkpoint import build_checkpointer
from xense0vla.data import (
    PolicyDataCollator,
    build_dataloader,
    build_vla_dataset,
)
from xense0vla.distributed.offloading import build_activation_offloading_context
from xense0vla.distributed.parallel_state import get_parallel_state, init_parallel_state
from xense0vla.distributed.torch_parallelize import build_parallelize_model
from xense0vla.models import save_model_assets
from xense0vla.optim import build_lr_scheduler, build_muon_optimizer, build_optimizer
from xense0vla.training.checkpoint import TrainingCheckpoint
from xense0vla.training.runtime import (
    _cuda_stage_end,
    _cuda_stage_start,
    _elapsed_cuda_stages,
)
from xense0vla.utils import helper
from xense0vla.utils.arguments import parse_args, save_args
from xense0vla.utils.async_tb_writer import AsyncTBWriter
from xense0vla.utils.dist_utils import all_reduce


logger = helper.create_logger(__name__)


def train(recipe):
    """Run either recipe with the same distributed optimizer/checkpoint loop."""
    args = parse_args(recipe.arguments)
    if args.train.performance_log_steps < 0:
        raise ValueError("train.performance_log_steps must be non-negative.")
    recipe.validate(args)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))

    if not args.data.norm_stats_file:
        raise ValueError(
            "VLA training requires data.norm_stats_file; robot YAML "
            "normalization fallbacks are not supported."
        )
    if not os.path.isfile(os.path.expanduser(args.data.norm_stats_file)):
        raise FileNotFoundError(
            f"data.norm_stats_file does not exist: {args.data.norm_stats_file!r}."
        )

    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(backend="nccl")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare model")
    model = recipe.build_model(args)
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    logger.info_rank0("Prepare data")
    processor = recipe.build_processor(args)

    if args.train.rmpad or args.train.rmpad_with_pos_ids:
        raise ValueError(
            "VLA requires regular padded batches; set both rmpad and "
            "rmpad_with_pos_ids to false."
        )

    data_collate_fn = [
        PolicyDataCollator()
    ]
    if args.data.dataloader_type == "native":
        args.data.chunk_size = args.train.chunk_size
        train_dataset = build_vla_dataset(
            dataset_config=args.data,
            model_config=args.model,
            config=model.config,
            processor=processor,
            return_raw_images=False,
        )
        args.train.compute_train_steps(
            args.data.max_seq_len,
            args.data.train_size,
            len(train_dataset),
        )

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            collate_fn=data_collate_fn,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor if args.data.num_workers > 0 else None,
            persistent_workers=args.data.persistent_workers,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    model = build_parallelize_model(
        model,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=True,
        parameter_dtype=args.train.parameter_dtype,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        fsdp_kwargs={},
        basic_modules=model._no_split_modules if args.train.module_fsdp_enable else None,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        fsdp_llm_blocks=False,
        ignore_norm=False,
        vlm_fsdp=getattr(args.train, "vlm_fsdp", False),
    )
    logger.info_rank0(model)
    if args.train.use_compile:
        model = torch.compile(model)

    if args.train.optimizer == "muon":
        optimizer = build_muon_optimizer(
            model,
            args.train,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
        )
        muon_groups, adamw_groups = (
            optimizer.optimizers[0].param_groups,
            optimizer.optimizers[1].param_groups if len(optimizer.optimizers) > 1 else [],
        )
        muon_summary = {
            f"{g.get('group_name', 'default')}@lr={g['lr']:.2e}": len(g["params"])
            for g in muon_groups
        }
        adamw_summary = {
            f"{g.get('group_name', 'default')}@lr={g['lr']:.2e}": len(g["params"])
            for g in adamw_groups
        }
        logger.info_rank0(
            f"Muon enabled. Muon groups: {muon_summary}; AdamW fallback groups: {adamw_summary}"
        )
    else:
        optimizer = build_optimizer(
            model,
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
            fused=args.train.fused_optimizer,
            optimizer_type=args.train.optimizer,
            vlm_lr=getattr(args.train, "vlm_lr", None),
            action_policy_lr=args.train.action_policy_lr,
        )

    total_train_steps = args.train.train_steps * args.train.num_train_epochs
    if args.train.max_steps is not None:
        total_train_steps = min(total_train_steps, args.train.max_steps)
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=total_train_steps,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    model_assets = [model_config, processor]
    writer = None
    if args.train.global_rank == 0:
        log_dir=f"{args.train.output_dir}/runs/"
        writer = AsyncTBWriter(log_dir=log_dir)
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        if args.train.enable_profiling:
            profiler = helper.create_profiler(
                start_step=args.train.profile_start_step,
                end_step=args.train.profile_end_step,
                trace_dir=args.train.profile_trace_dir,
                record_shapes=args.train.profile_record_shapes,
                profile_memory=args.train.profile_profile_memory,
                with_stack=args.train.profile_with_stack,
            )
            profiler.start()

        save_model_assets(args.train.model_assets_dir, model_assets)

    start_epoch, start_step, global_step = 0, 0, 0
    last_saved_step = -1

    checkpoints = TrainingCheckpoint(
        args=args, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        model_assets=model_assets, writer=writer, checkpointer=Checkpointer,
    )
    start_epoch, start_step, global_step = checkpoints.restore()

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    logger.info_rank0(f"Training precision: {args.train.parameter_dtype} parameters with BF16 autocast compute.")
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    if args.train.global_rank == 0:
        os.makedirs(args.train.save_checkpoint_path, exist_ok=True)
    reached_max_steps = False
    max_steps_driven = (
        args.train.max_steps is not None
        and args.train.max_steps < args.train.train_steps * args.train.num_train_epochs
    )
    if max_steps_driven:
        data_loader_tqdm = trange(
            args.train.max_steps,
            bar_format="Step: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
            initial=global_step,
            disable=args.train.local_rank != 0,
        )
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        if not max_steps_driven:
            data_loader_tqdm = trange(
                args.train.train_steps,
                desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
                total=args.train.train_steps,
                initial=start_step,
                disable=args.train.local_rank != 0,
            )
        data_iterator = iter(train_dataloader)
        for epoch_step in range(start_step, args.train.train_steps):
            global_step += 1
            iteration_start_time = time.perf_counter()
            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break
            data_wait_time = time.perf_counter() - iteration_start_time

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0.0
            total_action_loss = 0.0
            total_language_loss = 0.0
            total_metrics: Dict[str, float] = {}
            avg_lang_length = 0.0
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            should_log_performance = (
                args.train.performance_log_steps > 0
                and global_step % args.train.performance_log_steps == 0
            )
            stage_names = [
                "h2d",
                "model_forward",
                "backward",
                "grad_clip",
                "optimizer",
                "metric_reduce",
            ]
            stage_events: Dict[
                str, List[tuple[torch.cuda.Event, torch.cuda.Event]]
            ] = {}
            num_micro_batches = len(micro_batches)
            for micro_batch in micro_batches:
                micro_batch.pop("rep_id", None)
                _cuda_stage_start(stage_events, "h2d", should_log_performance)
                micro_batch = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                _cuda_stage_end(stage_events, "h2d", should_log_performance)

                # FSDP applies module-level mixed precision when sharding is
                # active; autocast also covers unwrapped MoT paths and the
                # single-GPU case where FSDP is intentionally disabled.
                _cuda_stage_start(
                    stage_events, "model_forward", should_log_performance
                )
                with model_fwd_context, torch.autocast("cuda", dtype=torch.bfloat16):
                    (
                        loss,
                        action_loss,
                        language_loss,
                        loss_dict,
                        _,
                    ) = recipe.forward(model, micro_batch)
                    loss = loss / num_micro_batches
                    action_loss = action_loss / num_micro_batches
                    language_loss = language_loss / num_micro_batches
                    if "lang_masks" in micro_batch:
                        avg_lang_length += (
                            micro_batch["lang_masks"].sum(dim=-1).float().mean().item()
                            / num_micro_batches
                        )
                _cuda_stage_end(
                    stage_events, "model_forward", should_log_performance
                )

                _cuda_stage_start(stage_events, "backward", should_log_performance)
                with model_bwd_context:
                    loss.backward()
                _cuda_stage_end(stage_events, "backward", should_log_performance)

                total_loss += loss.item()
                total_action_loss += action_loss.item()
                total_language_loss += language_loss.item()
                for key, value in loss_dict.items():
                    if torch.is_tensor(value):
                        if value.numel() != 1:
                            continue
                        scalar = value.detach().float().item()
                    elif isinstance(value, (int, float)):
                        scalar = float(value)
                    else:
                        continue
                    total_metrics[key] = (
                        total_metrics.get(key, 0.0) + scalar / num_micro_batches
                    )

                del micro_batch

            if global_step > args.train.stable_train_steps:
                max_grad_norm = args.train.decayed_max_grad_norm
            else:
                max_grad_norm = args.train.max_grad_norm
            _cuda_stage_start(stage_events, "grad_clip", should_log_performance)
            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, foreach=True)
            _cuda_stage_end(stage_events, "grad_clip", should_log_performance)

            _cuda_stage_start(stage_events, "optimizer", should_log_performance)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            _cuda_stage_end(stage_events, "optimizer", should_log_performance)
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()
            elif torch.is_tensor(grad_norm):
                grad_norm = grad_norm.item()

            metric_keys = sorted(total_metrics)
            _cuda_stage_start(stage_events, "metric_reduce", should_log_performance)
            reduced_values = all_reduce(
                [
                    total_loss,
                    total_action_loss,
                    total_language_loss,
                    avg_lang_length,
                    grad_norm,
                    *(total_metrics[key] for key in metric_keys),
                ],
                group=get_parallel_state().fsdp_group,
            )
            _cuda_stage_end(stage_events, "metric_reduce", should_log_performance)
            (
                total_loss,
                total_action_loss,
                total_language_loss,
                avg_lang_length,
                grad_norm,
                *metric_values,
            ) = reduced_values
            total_metrics = dict(zip(metric_keys, metric_values))

            torch.cuda.synchronize()
            delta_time = time.time() - start_time
            iteration_time = time.perf_counter() - iteration_start_time
            peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
            peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
            lr = max(lr_scheduler.get_last_lr())
            data_loader_tqdm.update()
            metrics_str = "".join(
                f"{key} {value:.4f}, " for key, value in total_metrics.items()
            )
            performance_message = None
            if should_log_performance:
                local_performance = torch.tensor(
                    [
                        data_wait_time,
                        iteration_time,
                        *_elapsed_cuda_stages(stage_events, stage_names),
                    ],
                    dtype=torch.float32,
                    device=f"cuda:{torch.cuda.current_device()}",
                )
                gathered_performance = [
                    torch.empty_like(local_performance)
                    for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_performance, local_performance)
                performance_by_rank = torch.stack(gathered_performance).cpu()
                performance_labels = ["data_wait", "iteration", *stage_names]
                summaries = []
                for index, label in enumerate(performance_labels):
                    values = performance_by_rank[:, index]
                    slow_rank = int(values.argmax().item())
                    summaries.append(
                        f"{label}={values.min().item():.3f}/"
                        f"{values.mean().item():.3f}/"
                        f"{values.max().item():.3f}s(r{slow_rank})"
                    )
                performance_message = "Perf min/mean/max: " + ", ".join(summaries)
            logger.info_rank0(
                f"Step {global_step}/{args.train.train_steps}, "
                f"Epoch {epoch+1}, "
                f"Total_Loss {total_loss:.4f}, "
                f"Action_Loss {total_action_loss:.4f}, "
                f"Language_Loss {total_language_loss:.4f}, "
                f"{metrics_str}"
                f"GradNorm {grad_norm:.4f}, "
                f"LR {lr:.2e}, "
                f"PeakAllocated {peak_allocated_gib:.2f}GiB, "
                f"PeakReserved {peak_reserved_gib:.2f}GiB, "
                f"StepTime {delta_time:.3f}s, "
                f"WallTime {iteration_time:.3f}s, "
                f"DataWait {data_wait_time:.3f}s"
            )
            if performance_message is not None:
                logger.info_rank0(performance_message)

            if args.train.global_rank == 0:
                writer.add_scalar("training/total_loss", total_loss, global_step)
                writer.add_scalar("training/action_loss", total_action_loss, global_step)
                writer.add_scalar(
                    "training/language_loss",
                    total_language_loss,
                    global_step,
                )
                for key, value in total_metrics.items():
                    writer.add_scalar(key, value, global_step)
                writer.add_scalar("training/grad_norm", grad_norm, global_step)
                writer.add_scalar("training/lr", lr, global_step)
                writer.add_scalar("training/avg_lang_length", avg_lang_length, global_step)
                writer.add_scalar("memory/peak_allocated_gib", peak_allocated_gib, global_step)
                writer.add_scalar("memory/peak_reserved_gib", peak_reserved_gib, global_step)
                writer.add_scalar("steptime", delta_time, global_step)
                writer.add_scalar("walltime", iteration_time, global_step)
                writer.add_scalar("data_wait", data_wait_time, global_step)

                if args.train.use_wandb:
                    wandb_metrics = {
                        "training/total_loss": total_loss,
                        "training/action_loss": total_action_loss,
                        "training/language_loss": total_language_loss,
                        "training/grad_norm": grad_norm,
                        "training/lr": lr,
                        "training/avg_lang_length": avg_lang_length,
                        "memory/peak_allocated_gib": peak_allocated_gib,
                        "memory/peak_reserved_gib": peak_reserved_gib,
                        "steptime": delta_time,
                        "walltime": iteration_time,
                        "data_wait": data_wait_time,
                        "epoch": epoch + 1,
                        **total_metrics,
                    }
                    wandb.log(wandb_metrics, step=global_step)

                if args.train.enable_profiling and global_step <= args.train.profile_end_step:
                    profiler.step()
                    if global_step == args.train.profile_end_step:
                        profiler.stop()
                        helper.upload_trace(
                            args.train.wandb_project, args.train.wandb_name, args.train.profile_trace_dir
                        )

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                checkpoints.save(global_step)
                last_saved_step = global_step

            if args.train.max_steps is not None and global_step >= args.train.max_steps:
                logger.info_rank0(f"Reached max_steps={args.train.max_steps}, stopping training.")
                reached_max_steps = True
                break

        if not max_steps_driven:
            data_loader_tqdm.close()
        if args.train.global_rank == 0:
            writer.flush()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if reached_max_steps:
            if last_saved_step != global_step:
                checkpoints.save(global_step)
                last_saved_step = global_step
            break
        if (
            args.train.save_epochs
            and (epoch + 1) % args.train.save_epochs == 0
            and last_saved_step != global_step
        ):
            checkpoints.save(global_step)
            last_saved_step = global_step

    if last_saved_step != global_step and global_step > 0:
        checkpoints.save(global_step)
    if max_steps_driven:
        data_loader_tqdm.close()
    if args.train.global_rank == 0:
        writer.close()
        if args.train.use_wandb:
            wandb.finish()
    torch.cuda.synchronize()
    # release memory
    helper.empty_cache()

    dist.barrier()
    dist.destroy_process_group()

