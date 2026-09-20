"""Prepare genome anchors and cached splits without starting training."""
import argparse
from pathlib import Path
from genda.config import DEFAULT_CONFIG, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True, help="Directory for parsed anchors and train/val/test CSVs")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.get("dataset") != "HUMAN_GENOME":
        parser.error("Only HUMAN_GENOME is supported")
    keys = ["fasta_path", "gtf_path"]
    keys.extend(key for key in ("repeatmasker_path", "ccre_bed_path") if config.get(key))
    for key in keys:
        if not config.get(key) or not Path(config[key]).is_file():
            parser.error(f"Missing {key}: {config.get(key)}; download the data and update --config")
    from .genome import get_train_test_data
    options = dict(gtf_path=config["gtf_path"], fasta_path=config["fasta_path"],
                   seed=config["seed"], output_dir=args.output_dir)
    splits = get_train_test_data(**options,
                repeatmasker_path=config.get("repeatmasker_path"),
                ccre_bed_path=config.get("ccre_bed_path"))[:3]
    for name, frame in zip(("train", "validation", "test"), splits):
        print(f"{name}: {len(frame):,} anchors")
    print(f"Prepared data in {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
