"""Construct the original human-genome data pipeline with explicit cache paths."""
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from .genome import get_train_test_data, FunctionalGenomicDataset, worker_init_fn
from .masking import DiffusionCollateFn, set_curriculum_stage


def build_loaders(config, tokenizer, cache_dir, workers=3):
    if config.get("dataset") != "HUMAN_GENOME":
        raise ValueError("Only HUMAN_GENOME is supported")
    for key in ("fasta_path", "gtf_path"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"{key}: {config[key]}")
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    def prepare():
        common = dict(gtf_path=config["gtf_path"], fasta_path=config["fasta_path"],
                      seed=config["seed"], output_dir=str(cache_dir))
        return get_train_test_data(**common, repeatmasker_path=config.get("repeatmasker_path"),
                                   ccre_bed_path=config.get("ccre_bed_path"))
    if rank == 0:
        splits = prepare()
    if distributed:
        dist.barrier()
    if rank != 0:
        splits = prepare()
    *frames, excluded = splits
    set_curriculum_stage(config.get("curriculum_stage", 0))
    loaders = []
    for index, frame in enumerate(frames):
        if config.get("debug", False):
            frame = frame.iloc[:200 if index == 0 else 50]
        dataset = FunctionalGenomicDataset(config["fasta_path"], frame, tokenizer,
                    is_train=index == 0, config=config, excluded_intervals=excluded)
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if distributed and index == 0 else None
        options = dict(num_workers=workers, pin_memory=True, worker_init_fn=worker_init_fn)
        if workers > 0:
            options.update(persistent_workers=True, prefetch_factor=2)
        loader = DataLoader(dataset, batch_size=config["batch_size"],
                            shuffle=index == 0 and sampler is None, sampler=sampler,
                            collate_fn=DiffusionCollateFn(tokenizer, config), **options)
        if not len(loader):
            raise ValueError("Empty data loader; check splits, batch size, and world size")
        loaders.append(loader)
    return loaders
