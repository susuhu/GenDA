"""Training entry point: python -m genda.training.cli --help."""
import argparse
from datetime import datetime
import os
from pathlib import Path
from genda.config import DEFAULT_CONFIG, load_config, checkpoint_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train GenDA on the human genome")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", help="Reusable dataset cache directory; defaults inside this run")
    parser.add_argument("--resume-from")
    parser.add_argument("--continue-training", action="store_true", help="Restore optimizer, scheduler and step as well as weights")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--attention", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args(argv)
    if args.continue_training and not args.resume_from:
        parser.error("--continue-training requires --resume-from")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    import torch.distributed as dist
    import yaml
    from torch.nn.parallel import DistributedDataParallel
    from transformers import get_cosine_schedule_with_warmup
    from genda.checkpoints import load_checkpoint
    from genda.runtime import seed_everything
    from genda.data.tokenizer import GenomicTokenizer
    from genda.data.loaders import build_loaders
    from genda.model.factory import build_model
    from .trainer import GenomicTrainer
    from .tracking import wandb

    config = load_config(args.config)
    if args.resume_from:
        previous = checkpoint_config(args.resume_from)
        if args.continue_training:
            previous.update({k: config[k] for k in ("total_steps", "patience")})
        config = previous
    if args.attention:
        config["attn_implementation"] = args.attention
    for key in ("batch_size", "total_steps", "accumulation_steps", "val_interval", "patience"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    rank, local_rank, world = (int(os.environ.get(k, default)) for k, default in
                               (("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if config["attn_implementation"] == "flash_attention_2" and device.type != "cuda":
        raise ValueError("Flash Attention requires CUDA; use --attention sdpa for CPU checks")
    if world > 1:
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    try:
        seed_everything(config.get("seed", 2026) + rank)
        torch.set_float32_matmul_precision("high")
        run_name = [datetime.now().strftime("genda_%Y%m%d_%H%M%S_%f") if rank == 0 else None]
        if world > 1:
            dist.broadcast_object_list(run_name, src=0)
        output = Path(args.output_dir).resolve() / run_name[0]
        if rank == 0:
            output.mkdir(parents=True)
            with (output / "run_config.yaml").open("w") as handle:
                yaml.safe_dump(config, handle)
            with (output / "run_arguments.yaml").open("w") as handle:
                yaml.safe_dump(vars(args), handle)
        if world > 1:
            dist.barrier()
        cache = Path(args.cache_dir) if args.cache_dir else output / "data_cache"
        tokenizer = GenomicTokenizer(config["kmer_size"])
        train_loader, val_loader, test_loader = build_loaders(config, tokenizer, cache, workers=args.workers)
        model = build_model(config, tokenizer)
        model.enable_gradient_checkpointing()
        model.to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                      weight_decay=float(config["weight_decay"]))
        scheduler = get_cosine_schedule_with_warmup(optimizer, config["warmup_steps"], config["total_steps"])
        step, best = 0, float("inf")
        if args.resume_from:
            loaded = load_checkpoint(args.resume_from, model,
                        optimizer if args.continue_training else None,
                        scheduler if args.continue_training else None)
            if args.continue_training:
                step, best = loaded
        if world > 1:
            model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
        if args.wandb and rank == 0:
            wandb.enable(project="GenDA", config=config, name=run_name[0], dir=str(output))
        trainer = GenomicTrainer(model, tokenizer, device,
            accumulation_steps=config["accumulation_steps"], use_amp=device.type == "cuda",
            use_span_weighting=config.get("use_span_weighting", False), span_lambda=config["span_lambda"],
            max_span_weight=config.get("max_span_weight", 1.0), entropy_exponent=config.get("entropy_exponent", 1.0),
            curriculum_stage=config.get("curriculum_stage", 0))
        trainer.best_val_loss = best
        if config.get("clinvar_eval", False) and rank == 0:
            from .clinvar import ClinVarInLoopEvaluator
            trainer.clinvar_eval = ClinVarInLoopEvaluator(config["clinvar_eval_parquet"], tokenizer, device,
                n=config.get("clinvar_eval_n"), win=config.get("clinvar_eval_win", 4096),
                batch_size=config.get("clinvar_eval_bs", 64))
            trainer.clinvar_select = config.get("clinvar_select_region", "overall")
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        trainer.train(train_loader, val_loader, optimizer, scheduler, criterion,
                      config["total_steps"], str(output), config["val_interval"], config["patience"], step)
        trainer.test(test_loader, criterion, str(output))
        print(f"Run saved to {output}")
    finally:
        if wandb.run is not None:
            wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
