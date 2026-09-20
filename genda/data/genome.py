import random
import os
import yaml
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from pyfaidx import Fasta
from genda.runtime import seed_everything
from genda.data.masking import GenomicTokenizer, DiffusionCollateFn


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# ANCHOR PARSING FUNCTIONS
# ============================================================

def parse_and_cache_gtf(gtf_path, fasta_path, cache_path):
    """
    Parses GENCODE GTF to extract functional anchors:
    1. Promoters: Defined as Transcription Start Site (TSS) +/- 2000bp derived from 'gene' entries.
    2. Exons: Explicit 'exon' entries.
    Also validates against the FASTA file to remove sequences with >5% Ns.
    """
    if os.path.exists(cache_path):
        print(f"✅ Found cached anchors at {cache_path}. Loading...")
        return pd.read_csv(cache_path)

    print(f"⏳ Parsing GTF from {gtf_path}... This takes about 1-2 mins...")

    cols = [
        "chrom", "source", "feature", "start", "end",
        "score", "strand", "frame", "attribute",
    ]

    df = pd.read_csv(
        gtf_path, sep="\t", comment="#", names=cols,
        usecols=["chrom", "feature", "start", "end", "strand", "attribute"],
    )

    df["gene_id"] = df["attribute"].str.extract(r'gene_id "([^"]+)"')
    df = df[df["feature"].isin(["gene", "exon"])]

    print(f"🔍 Validating sequences against {fasta_path} (Checking for 'N's)...")
    fasta_ref = Fasta(fasta_path, sequence_always_upper=True)
    fasta_keys = set(fasta_ref.keys())

    anchors = []
    n_dropped_bounds = 0
    n_dropped_ns = 0
    n_dropped_empty = 0

    print(f"⚡ Validating sequences...")

    for row in df.itertuples():
        chrom = row.chrom
        if chrom not in fasta_keys:
            continue

        chrom_len = len(fasta_ref[chrom])

        if row.feature == "gene":
            center = (row.start - 1) if row.strand == "+" else (row.end - 1)
            start = max(0, center - 2000)
            end = center + 2000
            a_type = "promoter"
        else:
            start = row.start - 1
            end = row.end
            a_type = "exon"

        if end > chrom_len:
            n_dropped_bounds += 1
            continue
        if end <= start:
            n_dropped_empty += 1
            continue

        seq = str(fasta_ref[chrom][start:end])
        seq_len = end - start
        if seq_len == 0:
            n_dropped_empty += 1
            continue
        if (seq.count("N") / seq_len) > 0.05:
            n_dropped_ns += 1
            continue

        anchors.append({
            "chrom": chrom, "start": start, "end": end,
            "type": a_type, "gene_id": row.gene_id, "strand": row.strand,
        })

    fasta_ref.close()

    anchors_df = pd.DataFrame(anchors)
    print(f"   Raw valid anchors: {len(anchors_df)}")
    anchors_df = anchors_df.drop_duplicates(subset=["chrom", "start", "end", "type"])
    print(f"   Unique anchors: {len(anchors_df)}")
    print(f"   🧹 Cleaned up: {n_dropped_bounds} OOB, {n_dropped_empty} empty, {n_dropped_ns} N-rich.")
    print(f"   ✅ Final Unique Anchors: {len(anchors_df)}")

    print(f"💾 Saving {len(anchors_df)} functional anchors to cache...")
    anchors_df.to_csv(cache_path, index=False)
    return anchors_df


def parse_intron_anchors(gtf_path, fasta_path, cache_path, min_intron_len=200):
    """
    Derives intron intervals from GENCODE GTF.

    For each transcript: introns = gaps between consecutive sorted exons.
    Stores the full intron interval (start, end). During training, __getitem__
    samples a random seq_len window from within these bounds — not center-cropped,
    since introns can be 100kb+.

    min_intron_len: skip introns shorter than this (avoids tiny frameshifting introns
    that are indistinguishable from exon boundaries at seq_len resolution).
    """
    if os.path.exists(cache_path):
        print(f"✅ Found cached intron anchors at {cache_path}. Loading...")
        return pd.read_csv(cache_path)

    print(f"⏳ Parsing intron anchors from {gtf_path}...")

    cols = [
        "chrom", "source", "feature", "start", "end",
        "score", "strand", "frame", "attribute",
    ]
    df = pd.read_csv(
        gtf_path, sep="\t", comment="#", names=cols,
        usecols=["chrom", "feature", "start", "end", "strand", "attribute"],
    )
    df["gene_id"]       = df["attribute"].str.extract(r'gene_id "([^"]+)"')
    df["transcript_id"] = df["attribute"].str.extract(r'transcript_id "([^"]+)"')
    df = df[df["feature"].isin(["transcript", "exon"])]

    fasta_ref = Fasta(fasta_path, sequence_always_upper=True)
    fasta_keys = set(fasta_ref.keys())
    chrom_lens = {k: len(fasta_ref[k]) for k in fasta_keys}

    exons_df = df[df["feature"] == "exon"].copy()
    exons_df["start_0"] = exons_df["start"] - 1  # GTF 1-based → 0-based half-open
    exons_by_tx = exons_df.groupby("transcript_id")

    transcripts = df[df["feature"] == "transcript"]

    anchors = []
    n_short = 0
    n_bounds = 0
    n_ns = 0

    for tx_row in transcripts.itertuples():
        chrom = tx_row.chrom
        if chrom not in fasta_keys:
            continue

        tx_id = tx_row.transcript_id
        if tx_id not in exons_by_tx.groups:
            continue

        tx_exons = (
            exons_by_tx.get_group(tx_id)[["start_0", "end"]]
            .sort_values("start_0")
            .reset_index(drop=True)
        )

        for i in range(len(tx_exons) - 1):
            intron_start = int(tx_exons.loc[i,   "end"])      # 0-based: exon end = intron start
            intron_end   = int(tx_exons.loc[i+1, "start_0"])  # 0-based: next exon start = intron end
            intron_len   = intron_end - intron_start

            if intron_len < min_intron_len:
                n_short += 1
                continue
            if intron_end > chrom_lens[chrom]:
                n_bounds += 1
                continue

            # N-check on a 200 bp probe around the intron center
            probe_s = max(intron_start, (intron_start + intron_end) // 2 - 100)
            probe_e = min(intron_end,   probe_s + 200)
            seq = str(fasta_ref[chrom][probe_s:probe_e])
            if len(seq) > 0 and seq.count("N") / len(seq) > 0.05:
                n_ns += 1
                continue

            anchors.append({
                "chrom":   chrom,
                "start":   intron_start,
                "end":     intron_end,
                "type":    "intron",
                "gene_id": tx_row.gene_id,
                "strand":  tx_row.strand,
            })

    fasta_ref.close()

    df_out = pd.DataFrame(anchors).drop_duplicates(subset=["chrom", "start", "end"])
    print(
        f"   ✅ {len(df_out):,} intron intervals parsed "
        f"(skipped: {n_short} short, {n_bounds} OOB, {n_ns} N-rich)."
    )
    df_out.to_csv(cache_path, index=False)
    return df_out


def parse_repeat_anchors(repeatmasker_path, fasta_path, cache_path,
                         min_len=500, max_anchors=None):
    """
    Parses a RepeatMasker .out file (hg38.fa.out from UCSC genome browser).

    Format (after 3-line header):
        SW score | div% | del% | ins% | chrom | begin(1-based) | end | ... | class/family | ...

    min_len: skip repeats shorter than this (filters noisy short SINEs at <500 bp).
    max_anchors: optional cap — repeat regions are dense; cap to avoid overwhelming
                 the dataset. If None, uses all passing repeats.

    The full interval is stored. __getitem__ samples a random seq_len window from
    within these bounds (same as intron handling).
    """
    if os.path.exists(cache_path):
        print(f"✅ Found cached repeat anchors at {cache_path}. Loading...")
        return pd.read_csv(cache_path)

    print(f"⏳ Parsing RepeatMasker anchors from {repeatmasker_path}...")

    fasta_ref = Fasta(fasta_path, sequence_always_upper=True)
    fasta_keys = set(fasta_ref.keys())
    chrom_lens = {k: len(fasta_ref[k]) for k in fasta_keys}

    anchors = []
    n_short = 0
    n_bounds = 0
    n_ns = 0

    with open(repeatmasker_path, "r") as f:
        for line_no, line in enumerate(f):
            if line_no < 3:      # Skip 3-line header
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue

            try:
                chrom = parts[4]
                start = int(parts[5]) - 1   # 1-based → 0-based
                end   = int(parts[6])
            except (ValueError, IndexError):
                continue

            if chrom not in fasta_keys:
                continue
            if (end - start) < min_len:
                n_short += 1
                continue
            if end > chrom_lens[chrom]:
                n_bounds += 1
                continue

            # N-check on 200 bp probe around center
            probe_s = max(start, (start + end) // 2 - 100)
            probe_e = min(end,   probe_s + 200)
            seq = str(fasta_ref[chrom][probe_s:probe_e])
            if len(seq) > 0 and seq.count("N") / len(seq) > 0.05:
                n_ns += 1
                continue

            anchors.append({
                "chrom":   chrom,
                "start":   start,
                "end":     end,
                "type":    "repeat",
                "gene_id": None,
                "strand":  ".",
            })

    fasta_ref.close()

    df_out = pd.DataFrame(anchors).drop_duplicates(subset=["chrom", "start", "end"])
    if max_anchors and len(df_out) > max_anchors:
        df_out = df_out.sample(n=max_anchors, random_state=42).reset_index(drop=True)

    print(
        f"   ✅ {len(df_out):,} repeat anchors "
        f"(skipped: {n_short} short, {n_bounds} OOB, {n_ns} N-rich)."
    )
    df_out.to_csv(cache_path, index=False)
    return df_out


def parse_enhancer_anchors(ccre_bed_path, fasta_path, cache_path):
    """
    Parses an ENCODE cCRE or EPDnew enhancer BED file.

    Expected format: chrom \\t start(0-based) \\t end \\t [name ...]
    The function is permissive — only the first 3 columns are required.

    Enhancers from ENCODE cCREs are typically 150–500 bp, similar to exons,
    so __getitem__ uses standard center-crop sampling (same as exon/promoter).
    """
    if os.path.exists(cache_path):
        print(f"✅ Found cached enhancer anchors at {cache_path}. Loading...")
        return pd.read_csv(cache_path)

    print(f"⏳ Parsing enhancer anchors from {ccre_bed_path}...")

    fasta_ref = Fasta(fasta_path, sequence_always_upper=True)
    fasta_keys = set(fasta_ref.keys())
    chrom_lens = {k: len(fasta_ref[k]) for k in fasta_keys}

    anchors = []
    n_dropped = 0

    with open(ccre_bed_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            chrom = parts[0]
            try:
                start = int(parts[1])
                end   = int(parts[2])
            except ValueError:
                continue

            if chrom not in fasta_keys:
                continue
            if end > chrom_lens[chrom] or end <= start:
                n_dropped += 1
                continue

            seq_len = end - start
            seq = str(fasta_ref[chrom][start:end])
            if seq_len > 0 and seq.count("N") / seq_len > 0.05:
                n_dropped += 1
                continue

            anchors.append({
                "chrom":   chrom,
                "start":   start,
                "end":     end,
                "type":    "enhancer",
                "gene_id": None,
                "strand":  ".",
            })

    fasta_ref.close()
    df_out = pd.DataFrame(anchors).drop_duplicates(subset=["chrom", "start", "end"])
    print(f"   ✅ {len(df_out):,} enhancer anchors parsed ({n_dropped} dropped).")
    df_out.to_csv(cache_path, index=False)
    return df_out


# ============================================================
# INTERVAL EXCLUSION UTILITIES (for clean background sampling)
# ============================================================

def build_excluded_intervals(dfs):
    """
    Merge anchor DataFrames into a per-chromosome sorted list of (start, end) intervals.
    Overlapping intervals are merged so the structure stays compact.

    Used by FunctionalGenomicDataset._get_random_background to guarantee that
    [BACKGROUND] windows do not overlap any annotated region (exon, promoter,
    intron, repeat, enhancer).

    Returns:
        dict[str, list[[start, end]]]: merged, sorted intervals per chromosome.
    """
    combined = pd.concat(dfs, ignore_index=True)[["chrom", "start", "end"]]
    excluded = {}
    for chrom, grp in combined.groupby("chrom"):
        intervals = sorted(zip(grp["start"].tolist(), grp["end"].tolist()))
        merged = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        excluded[chrom] = merged
    return excluded


def _overlaps_excluded(chrom, start, end, excluded_intervals):
    """
    O(log n) overlap check: returns True if [start, end) overlaps any excluded interval.

    Binary searches for the first interval whose end > start, then checks if it
    starts before end. Works correctly because excluded_intervals are merged +
    sorted by start.
    """
    if chrom not in excluded_intervals:
        return False
    intervals = excluded_intervals[chrom]
    # Binary search: find first interval with end > start
    lo, hi = 0, len(intervals)
    while lo < hi:
        mid = (lo + hi) // 2
        if intervals[mid][1] <= start:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(intervals) and intervals[lo][0] < end


# ============================================================
# DATA SPLIT FUNCTIONS
# ============================================================

def get_train_test_data(
    gtf_path,
    fasta_path,
    seed,
    repeatmasker_path=None,
    ccre_bed_path=None,
    output_dir=None,
    repeat_cap=None,
    OVERWRITE=False,
):
    """
    GenDA pipeline: promoter + exon + intron + repeat + enhancer.

    Splits:
    - Types with gene_id (promoter, exon, intron): gene-aware GroupShuffleSplit
      to prevent gene leakage across splits.
    - Types without gene_id (repeat, enhancer): random 80/10/10 row split.

    Splits are saved to output_dir (separate from the original data dir so
    existing splits for the trained model are untouched).

    Returns excluded_intervals in addition to the three DataFrames, so the
    caller can pass it directly to FunctionalGenomicDataset to clean up
    background sampling.
    """
    if output_dir is None:
        raise ValueError("An explicit output_dir is required for dataset caches")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_set.csv")
    val_path   = os.path.join(output_dir, "val_set_fixed.csv")
    test_path  = os.path.join(output_dir, "test_set_fixed.csv")

    if not OVERWRITE:
        if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
            print("✅ Found ready-to-use Train/Val/Test splits. Loading...")
            train_df = pd.read_csv(train_path)
            val_df   = pd.read_csv(val_path)
            test_df  = pd.read_csv(test_path)
            # Rebuild excluded_intervals from the combined anchor pool
            all_dfs  = [train_df, val_df, test_df]
            excluded = build_excluded_intervals(all_dfs)
            return train_df, val_df, test_df, excluded

    # ---- Parse all anchor types ----
    all_dfs_to_merge = []

    # 1. Promoter + exon (gene-aware split)
    prom_exon_path = os.path.join(output_dir, "all_anchors_prom_exon.csv")
    prom_exon_df   = parse_and_cache_gtf(gtf_path, fasta_path, prom_exon_path)
    all_dfs_to_merge.append(("gene_aware", prom_exon_df))

    # 2. Introns (gene-aware split — share gene_id with GTF)
    intron_path = os.path.join(output_dir, "all_anchors_intron.csv")
    intron_df   = parse_intron_anchors(gtf_path, fasta_path, intron_path)
    all_dfs_to_merge.append(("gene_aware", intron_df))

    # 3. Repeats (optional, random split — no gene_id)
    if repeatmasker_path:
        repeat_path = os.path.join(output_dir, "all_anchors_repeat.csv")
        repeat_df   = parse_repeat_anchors(
            repeatmasker_path, fasta_path, repeat_path, max_anchors=repeat_cap
        )
        all_dfs_to_merge.append(("random", repeat_df))
    else:
        print("⚠️  No repeatmasker_path provided — skipping [REPEAT] anchors.")

    # 4. Enhancers (optional, random split — no gene_id)
    if ccre_bed_path:
        enhancer_path = os.path.join(output_dir, "all_anchors_enhancer.csv")
        enhancer_df   = parse_enhancer_anchors(ccre_bed_path, fasta_path, enhancer_path)
        all_dfs_to_merge.append(("random", enhancer_df))
    else:
        print("⚠️  No ccre_bed_path provided — skipping [ENHANCER] anchors.")

    # ---- Split each group and collect ----
    train_parts, val_parts, test_parts = [], [], []

    for split_strategy, df in all_dfs_to_merge:
        if df is None or len(df) == 0:
            continue

        if split_strategy == "gene_aware":
            splitter_test = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
            train_val_idx, test_idx = next(
                splitter_test.split(df, groups=df["gene_id"])
            )
            tv_df    = df.iloc[train_val_idx].copy()
            test_p   = df.iloc[test_idx].copy()

            splitter_val = GroupShuffleSplit(n_splits=1, test_size=1/9, random_state=seed)
            train_idx, val_idx = next(
                splitter_val.split(tv_df, groups=tv_df["gene_id"])
            )
            train_p = tv_df.iloc[train_idx].copy()
            val_p   = tv_df.iloc[val_idx].copy()

        else:  # random split for repeat/enhancer (no gene_id)
            n = len(df)
            idx = np.random.default_rng(seed).permutation(n)
            test_cut  = int(n * 0.1)
            val_cut   = int(n * 0.2)
            test_p    = df.iloc[idx[:test_cut]].copy()
            val_p     = df.iloc[idx[test_cut:val_cut]].copy()
            train_p   = df.iloc[idx[val_cut:]].copy()

        train_parts.append(train_p)
        val_parts.append(val_p)
        test_parts.append(test_p)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df   = pd.concat(val_parts,   ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True)

    print(f"\n💾 Saving splits to {output_dir}")
    print(f"   Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")
    for split_df, path in [(train_df, train_path), (val_df, val_path), (test_df, test_path)]:
        split_df.to_csv(path, index=False)

    # Build excluded_intervals from the full combined anchor pool
    excluded = build_excluded_intervals([train_df, val_df, test_df])
    return train_df, val_df, test_df, excluded


# ============================================================
# DATASET CLASS
# ============================================================

# Types that store full interval bounds and need random-window sampling
# (vs. small types like exon/promoter/enhancer that use center-crop).
_LARGE_INTERVAL_TYPES = {"intron", "repeat"}


class FunctionalGenomicDataset(Dataset):
    def __init__(self, fasta_path, anchors_df, tokenizer, is_train, config,
                 excluded_intervals=None):
        """
        Parameters
        ----------
        excluded_intervals : dict or None
            Output of build_excluded_intervals(). When provided, _get_random_background
            rejects windows overlapping any annotated region, restricting [BACKGROUND]
            to truly intergenic non-repetitive sequence.
            Pass None (default) for backward compatibility with existing callers.
        """
        self.fasta_path           = fasta_path
        self.tokenizer            = tokenizer
        self.is_train             = is_train
        self.config               = config
        self.dropout_rate         = self.config.get("cond_dropout_rate", 0.0)
        self.excluded_intervals   = excluded_intervals  # None → old behavior

        # ---- Condition token map ----
        # [INTRON] and [ENHANCER] are already in the tokenizer vocab (dataset_utils.py:20).
        # [REPEAT] and [NULL] are not yet — they map to <UNK> until the tokenizer is updated.
        # The code below detects their absence and falls back gracefully so no UNK tokens
        # are injected into the input sequence.
        self.type_to_token = {
            "promoter":   "[PROMOTER]",
            "exon":       "[EXON]",
            "enhancer":   "[ENHANCER]",
            "intron":     "[INTRON]",
            "background": "[BACKGROUND]",
        }

        # [REPEAT]: add only if the token is actually in the tokenizer vocab
        if "[REPEAT]" in self.tokenizer.token_to_id:
            self.type_to_token["repeat"] = "[REPEAT]"
        else:
            # Fall back to [BACKGROUND] until tokenizer is updated.
            # This preserves correct behavior for old checkpoints.
            self.type_to_token["repeat"] = "[BACKGROUND]"
            if is_train:
                print("⚠️  [REPEAT] not in tokenizer vocab — repeat anchors will use [BACKGROUND] tag. "
                      "Update tokenizer to enable separate [REPEAT] conditioning.")

        # [NULL] for CFG dropout: use only if in vocab (needs tokenizer update).
        # Detected once here and stored as a flag for __getitem__.
        self._null_token_available = "[NULL]" in self.tokenizer.token_to_id
        if self._null_token_available:
            self.type_to_token["null"] = "[NULL]"
        elif is_train and self.dropout_rate > 0:
            print("⚠️  [NULL] not in tokenizer vocab — CFG dropout will silently omit the condition tag. "
                  "Update tokenizer to enable explicit [NULL] unconditional token.")

        # ---- Class balancing (training only) ----
        if self.is_train:
            print(f"⚖️  Checking Class Balance...")
            anchors_df = self._balance_anchors(anchors_df, config)

            if self.dropout_rate > 0:
                print(f"🎲 CFG dropout: {self.dropout_rate * 100:.0f}% of sequences will use "
                      f"{'[NULL] token' if self._null_token_available else 'silent condition drop'}.")

        self.anchors = anchors_df.reset_index(drop=True)

        # ---- Load genome into RAM ----
        print("⏳ Loading Genome into RAM (Approx 3.2GB)... This eliminates Disk I/O lag.")
        self.genome_cache = {}
        temp_fasta = Fasta(self.fasta_path, sequence_always_upper=True)
        for chrom in temp_fasta.keys():
            self.genome_cache[chrom] = str(temp_fasta[chrom])
        temp_fasta.close()

        print(f"✅ Genome loaded! Cached {len(self.genome_cache)} chromosomes.")
        self.chrom_list = list(self.genome_cache.keys())
        self.chrom_lens = {k: len(v) for k, v in self.genome_cache.items()}

    @staticmethod
    def _balance_anchors(anchors_df, config):
        """
        Optionally balance anchor counts across condition types.

        When config["balance_classes"] is True (default: False):
        - Exon is the dominant class and acts as the reference count.
        - Promoter and enhancer (typically < exon count) are upsampled to match exons.
        - Intron and repeat (potentially >> exon count) are downsampled to
          config["intron_cap"] / config["repeat_cap"] if those keys exist,
          otherwise capped at exon count.

        When False: just shuffles and returns natural ratios.
        """
        if not config.get("balance_classes", False):
            shuffled = anchors_df.sample(frac=1, random_state=config.get("seed", 2026)).reset_index(drop=True)
            print(f"   ✅ Dataset (natural ratios): {len(shuffled):,} anchors total.")
            return shuffled

        df_exons = anchors_df[anchors_df["type"] == "exon"]
        n_exon   = len(df_exons)

        if n_exon == 0:
            return anchors_df

        parts = [df_exons]

        for a_type in ["promoter", "enhancer"]:
            df_t = anchors_df[anchors_df["type"] == a_type]
            if len(df_t) == 0:
                continue
            if len(df_t) < n_exon:
                multiplier = n_exon // len(df_t)
                print(f"   Upsampling {a_type} {multiplier}x ({len(df_t):,} → ~{len(df_t)*multiplier:,})...")
                df_t = pd.concat([df_t] * multiplier, ignore_index=True)
            parts.append(df_t)

        for a_type, cfg_key in [("intron", "intron_cap"), ("repeat", "repeat_cap")]:
            df_t = anchors_df[anchors_df["type"] == a_type]
            if len(df_t) == 0:
                continue
            cap = config.get(cfg_key, n_exon)
            if len(df_t) > cap:
                print(f"   Downsampling {a_type}: {len(df_t):,} → {cap:,} (cap={cfg_key})...")
                df_t = df_t.sample(n=cap, random_state=config.get("seed", 42)).reset_index(drop=True)
            parts.append(df_t)

        combined = pd.concat(parts, ignore_index=True)
        combined = combined.sample(frac=1, random_state=config.get("seed", 2026)).reset_index(drop=True)
        print(f"   ✅ Balanced dataset: {len(combined):,} anchors total.")
        return combined

    def __len__(self):
        return len(self.anchors)

    def _get_random_background(self):
        """
        Sample a random genomic window for the [BACKGROUND] condition.

        When excluded_intervals is provided, rejects windows overlapping any
        annotated region (exon, promoter, intron, repeat, enhancer), restricting
        [BACKGROUND] to intergenic non-repetitive neutral sequence — the coherent
        null prior that CFG can meaningfully steer away from.

        Without excluded_intervals (backward compat), behavior is unchanged:
        any window with ≤5% N is accepted.
        """
        max_retries = 50
        seq_len = self.config["seq_len"]

        for _ in range(max_retries):
            chrom = random.choice(self.chrom_list)
            limit = self.chrom_lens[chrom]
            if limit <= seq_len:
                continue

            start = random.randint(0, limit - seq_len)
            end   = start + seq_len

            seq_str = self.genome_cache[chrom][start:end]
            if seq_str.count("N") > seq_len * 0.05:
                continue

            if self.excluded_intervals is not None:
                if _overlaps_excluded(chrom, start, end, self.excluded_intervals):
                    continue  # Landed on an annotated region — try again

            return chrom, start

        print("⚠️ Warning: Reached max retries for background sampling. Returning last checked.")
        return chrom, start

    def _sample_window_from_interval(self, row):
        """
        For large-interval types (intron, repeat): sample a random seq_len window
        from within [row.start, row.end] instead of center-cropping.

        This is necessary because introns can be 100kb+ — always centering would
        sample only the midpoint of the intron, never the splice-proximal regions
        or the deep intronic landscape.
        """
        seq_len    = self.config["seq_len"]
        chrom      = row["chrom"]
        ivl_start  = int(row["start"])
        ivl_end    = int(row["end"])
        chrom_len  = self.chrom_lens[chrom]

        max_start = min(ivl_end - seq_len, chrom_len - seq_len)
        min_start = ivl_start

        if max_start <= min_start:
            # Interval shorter than seq_len: center on midpoint
            center = (ivl_start + ivl_end) // 2
            start  = max(0, center - seq_len // 2)
        else:
            start = random.randint(min_start, max_start)

        return chrom, start

    def __getitem__(self, idx):
        use_anchor     = True
        condition_type = None

        # Determine anchor vs. background
        if self.is_train and random.random() > self.config["functional_ratio"]:
            use_anchor = False

        if use_anchor:
            row            = self.anchors.iloc[idx]
            condition_type = row["type"]

            if condition_type in _LARGE_INTERVAL_TYPES:
                # Large intervals: sample a random window from within the interval
                chrom, start = self._sample_window_from_interval(row)
                if self.is_train:
                    jitter = random.randint(-self.config["jitter_bp"], self.config["jitter_bp"])
                    start += jitter
            else:
                # Small anchors (exon, promoter, enhancer): center-crop with jitter
                chrom  = row["chrom"]
                center = (row["start"] + row["end"]) // 2
                if self.is_train:
                    jitter = random.randint(-self.config["jitter_bp"], self.config["jitter_bp"])
                    start  = center - (self.config["seq_len"] // 2) + jitter
                else:
                    start  = center - (self.config["seq_len"] // 2)
        else:
            chrom, start   = self._get_random_background()
            condition_type = "background"

        # Bounds check
        seq_len   = self.config["seq_len"]
        chrom_len = self.chrom_lens[chrom]
        start     = max(0, start)
        end       = start + seq_len
        if end > chrom_len:
            end   = chrom_len
            start = max(0, end - seq_len)

        # Strict cache check
        if chrom not in self.genome_cache:
            available_keys = list(self.genome_cache.keys())[:5]
            raise KeyError(
                f"🚨 GTF requested '{chrom}', but it is NOT in the Genome Cache.\n"
                f"   Available keys look like: {available_keys}...\n"
                f"   Did you forget to normalize the FASTA keys in __init__?"
            )

        seq_str = self.genome_cache[chrom][start:end]

        # Pad if chromosome is shorter than seq_len (extremely rare)
        if len(seq_str) < seq_len:
            seq_str = seq_str + "N" * (seq_len - len(seq_str))

        # CFG dropout (training only)
        # If [NULL] is in the tokenizer vocab, insert it explicitly so the model
        # learns a clean unconditional distribution p(x|[NULL]) distinct from
        # p(x|[BACKGROUND]).  Until the tokenizer is updated, fall back to silent
        # drop (existing behaviour) to avoid injecting <UNK> into inputs.
        if self.is_train and random.random() < self.dropout_rate:
            final_condition = "null" if self._null_token_available else "dropped"
        else:
            final_condition = condition_type

        # Build prefix
        prefix_tags = ["[HUMAN]"]

        if final_condition != "dropped":
            func_token = self.type_to_token.get(final_condition)
            if func_token:
                prefix_tags.append(func_token)

        tag_ids = [
            self.tokenizer.token_to_id.get(t, self.tokenizer.unk_id)
            for t in prefix_tags
        ]

        dna_ids = self.tokenizer.encode(seq_str)

        bos = [self.tokenizer.bos_id]
        eos = [self.tokenizer.eos_id]

        # "Omission as Signal": only append <EOS> when the full biological
        # element fits within max_token_len. If truncated, the *absence* of
        # <EOS> tells the model "this sequence continues beyond my window."
        # This prevents poisoning the <EOS> prior with artificial stops in the
        # middle of continuous motifs.
        #
        # We check the anchor's actual size (bp), NOT the extracted window length,
        # because center-crop always extracts exactly seq_len bp regardless of
        # whether the underlying element is 200 bp or 100 kb.
        overhead_with_eos = len(bos) + len(eos) + len(tag_ids)
        max_dna_with_eos  = self.config["max_token_len"] - overhead_with_eos
        max_bp_with_eos   = max_dna_with_eos * self.config.get("kmer_size", 1)

        if use_anchor:
            anchor_bp = int(row["end"]) - int(row["start"])
            fits_in_window = anchor_bp <= max_bp_with_eos
        else:
            # Background: no biological anchor → always truncated
            fits_in_window = False

        if fits_in_window:
            # ✓ Full element fits → keep EOS, truncate DNA to leave room
            dna_ids = dna_ids[:max_dna_with_eos]
            full_ids = np.concatenate([bos, tag_ids, dna_ids, eos]).astype(np.int64)
        else:
            # ✗ Truncated → omit <EOS>; +1 extra DNA token in its place
            max_dna_no_eos = self.config["max_token_len"] - len(bos) - len(tag_ids)
            dna_ids = dna_ids[:max_dna_no_eos]
            full_ids = np.concatenate([bos, tag_ids, dna_ids]).astype(np.int64)

        if "N" * 100 in seq_str:
            print(f"🚨 ALERT: Found Zombie Sequence! {chrom}:{start}-{end}")

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "condition": condition_type,
            "metadata":  f"{chrom}:{start}-{end}",
        }


def print_distribution(df, set_name):
    """Print human genome distribution + per-category sequence length stats."""
    print(f"\n🔹 {set_name} Set Distribution")
    print(f"{'Category':<15} | {'Count':<10} | {'Ratio':<6}")
    print("-" * 36)

    counts = df["type"].value_counts()
    total  = len(df)
    for category, count in counts.items():
        ratio = (count / total) * 100
        print(f"{category:<15} | {count:<10,} | {ratio:>5.1f}%")

    print("-" * 36)
    print(f"📏 Feature Length Stats (bp) — Per Category")
    print(f"   {'Category':<15} {'Count':>8} {'Min':>8} {'P25':>8} {'Median':>8} {'P75':>8} {'Max':>10} {'Mean':>8}  {'Fits?':>6}")
    print(f"   {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}  {'-'*6}")

    # EOS threshold: max_bp_with_eos for 1-mer with [HUMAN]+[TYPE] tags
    # overhead = BOS(1) + EOS(1) + tags(2) = 4 → 4096 - 4 = 4092 bp
    max_bp_with_eos = 4092

    for category in ["promoter", "exon", "enhancer", "intron", "repeat"]:
        sub = df[df["type"] == category]
        if len(sub) == 0:
            continue
        lens = sub["end"] - sub["start"]
        pct_fit = (lens <= max_bp_with_eos).mean() * 100
        print(f"   {category:<15} {len(sub):>8,} {lens.min():>8,} {lens.quantile(0.25):>8,.0f} "
              f"{lens.median():>8,.0f} {lens.quantile(0.75):>8,.0f} {lens.max():>10,} "
              f"{lens.mean():>8,.0f}  {pct_fit:>5.1f}%")

    # Global summary
    lengths = df["end"] - df["start"]
    pct_fit_total = (lengths <= max_bp_with_eos).mean() * 100
    print(f"   {'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*8}  {'─'*6}")
    print(f"   {'ALL':<15} {len(df):>8,} {lengths.min():>8,} {lengths.quantile(0.25):>8,.0f} "
          f"{lengths.median():>8,.0f} {lengths.quantile(0.75):>8,.0f} {lengths.max():>10,} "
          f"{lengths.mean():>8,.0f}  {pct_fit_total:>5.1f}%")
    print(f"   EOS threshold: ≤{max_bp_with_eos} bp → biological element fits in window + gets <EOS>")


