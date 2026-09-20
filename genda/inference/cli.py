"""Generation entry point: python -m genda.inference.cli --help."""
import argparse
import csv
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate DNA from a GenDA checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Defaults to run_config.yaml beside checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="[HUMAN][PROMOTER]")
    parser.add_argument("--negative-prompt", default="[HUMAN][BACKGROUND]")
    parser.add_argument("--length", type=int, default=256, help="Number of generated tokens")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device")
    parser.add_argument("--attention", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--allow-eos", action="store_true")
    parser.add_argument("--bio-check", action="store_true")
    parser.add_argument("--remasking-strategy", choices=["single_token", "poisson_span"], default="single_token")
    args = parser.parse_args(argv)
    if min(args.samples, args.batch_size, args.length, args.steps) <= 0 or args.temperature <= 0:
        parser.error("samples, batch size, length, steps and temperature must be positive")
    from genda.runtime import seed_everything
    from .api import Predictor
    seed_everything(args.seed)
    predictor = Predictor(args.checkpoint, args.config, args.device, args.attention)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sequences = []
    for offset in range(0, args.samples, args.batch_size):
        prompts = [args.prompt] * min(args.batch_size, args.samples - offset)
        sequences.extend(predictor.generate(prompts, args.length, args.steps, args.temperature,
            args.cfg_scale, args.negative_prompt, args.allow_eos, args.remasking_strategy, args.bio_check))
    with (output / "sequences.fasta").open("w") as handle:
        for i, sequence in enumerate(sequences):
            handle.write(f">sequence_{i}\n{sequence}\n")
    with (output / "sequences.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "prompt", "sequence"])
        writer.writerows((i, args.prompt, sequence) for i, sequence in enumerate(sequences))
    (output / "generation.json").write_text(json.dumps(dict(vars(args), checkpoint_step=predictor.step), indent=2))
    print(f"Saved {len(sequences)} sequences to {output}")


if __name__ == "__main__":
    main()
