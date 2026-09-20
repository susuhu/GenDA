import os
import re
import random
import numpy as np
import math

import torch
import torch.nn.functional as F


def find_premature_stop_codons(seq_str, gap_start):
    """
    Validates a 300-bp exon core centred in the canvas — NOT the full 4,096 bp.

    A continuous 4,096-bp ORF without a single stop codon is biologically
    implausible (median human exon ≈ 175 bp).  We only enforce the open
    reading frame check inside a realistic 300-bp sub-window at the centre
    of the generated canvas, letting the remaining 3,800 bp behave as
    natural non-coding flanking context.

    Returns a list of tensor indices to re-mask.
    """
    stop_codons = {"TAA", "TAG", "TGA"}
    L = len(seq_str)
    if L < 600:
        return []

    # ── Define a realistic 300-bp Exon Core centred in the canvas ──
    exon_window = 300
    exon_start = max(0, (L // 2) - (exon_window // 2))
    exon_end   = min(L, exon_start + exon_window)
    exon_str   = seq_str[exon_start:exon_end]
    exon_len   = len(exon_str)

    # Count stops in each of the 3 frames inside the exon sub-window only
    frame_stops = {0: [], 1: [], 2: []}
    for frame in range(3):
        for i in range(frame, exon_len - 2, 3):
            codon = exon_str[i:i+3]
            if codon in stop_codons:
                frame_stops[frame].append(i)

    # Pick the best frame (fewest stops)
    best_frame = min(frame_stops, key=lambda f: len(frame_stops[f]))
    bad_positions = frame_stops[best_frame]

    # At least one clean frame → exon core is valid
    if not bad_positions:
        return []

    bad_indices = []
    for i in bad_positions:
        # Map local exon-string index → global canvas coordinate
        global_idx = gap_start + exon_start + i
        bad_indices.extend([global_idx, global_idx + 1, global_idx + 2])
        if i > 0:
            bad_indices.append(global_idx - 1)
        if i < exon_len - 4:
            bad_indices.append(global_idx + 3)

    return bad_indices


def check_promoter_motifs(seq_str, gap_start):
    """
    Validates promoter architecture with the TSS centred in the canvas.

    Placing the TSS at the far-right edge (e.g. position 4046 of 4096)
    starves AlphaGenome of downstream gene-body context, muting predicted
    transcription peaks.  We force the TSS dead-centre (L//2) so the model
    sees ~2,000 bp of generated upstream promoter AND downstream gene body.

    Human promoters fall into two classes:
      - TATA-box promoters (~30%): sharp TSS, TATA at -31 to -26
      - CpG-island promoters (~70%): broad TSS, no TATA, high CpG density

    We flag only if NEITHER architecture is present.
    """
    bad_indices = []
    L = len(seq_str)
    if L < 2000:
        return bad_indices

    # 🎯 TSS dead-centre of the canvas (position 2048 for a 4096-bp window)
    tss_in_gap = L // 2

    # ── 1. TATA box check: -35 to -25 relative to TSS ──
    tata_start = max(0, tss_in_gap - 35)
    tata_end   = tss_in_gap - 20
    tata_region = seq_str[tata_start:tata_end]
    # Canonical TATA box: TATAWAWR (W=A/T, R=A/G)
    has_tata = bool(re.search(r'TATA[AT]{2}[AG]', tata_region))

    # ── 2. Inr (Initiator) check: -3 to +5 relative to TSS ──
    inr_start = max(0, tss_in_gap - 3)
    inr_end   = min(L, tss_in_gap + 6)
    inr_region = seq_str[inr_start:inr_end]
    # Inr consensus: YYANWYY (Y=C/T, W=A/T, N=any)
    has_inr = bool(re.search(r'[CT]{2}A[ACGT][AT][CT]{2}', inr_region))

    # ── 3. CpG island check: elevated CpG density near TSS ──
    cpg_start = max(0, tss_in_gap - 200)
    cpg_end   = min(L, tss_in_gap + 100)
    cpg_region = seq_str[cpg_start:cpg_end]
    c_count = cpg_region.count('C')
    g_count = cpg_region.count('G')
    cpg_count = cpg_region.count('CG')
    region_len = len(cpg_region)
    if region_len > 50 and c_count + g_count > 0:
        obs_over_exp = (cpg_count * region_len) / max(1, c_count * g_count)
        has_cpg_island = obs_over_exp > 0.6 and (c_count + g_count) / region_len > 0.5
    else:
        has_cpg_island = False

    # ── Decision ──
    if not has_tata and not has_cpg_island:
        # No recognizable promoter architecture — nuke the TATA-expected region
        target_start = gap_start + tata_start
        target_end   = gap_start + tata_end
        bad_indices.extend(list(range(target_start, target_end)))

    if not has_inr and not has_tata:
        # Also missing Inr — re-mask around TSS
        target_start = gap_start + inr_start
        target_end   = gap_start + inr_end
        bad_indices.extend(list(range(target_start, target_end)))

    return bad_indices


def check_gc_content(seq_str, gap_start, condition=None):
    """
    Forces GC content to stay within biologically appropriate ranges per condition.

    Typical human genomic GC ranges:
      - Exon:  50–65%  (coding regions are GC-rich)
      - Intron: 35–48%  (introns are AT-rich)
      - Promoter: 45–70% (CpG island promoters are very GC-rich)
      - Enhancer: 40–60% (variable, but avoid extremes)
      - Default: 35–65% (genome-wide safe range)

    Targets overrepresented nucleotides with span-aware re-masking.
    """
    L = max(1, len(seq_str))
    c_count = seq_str.count("C")
    g_count = seq_str.count("G")
    gc_pct = (c_count + g_count) / L

    # Condition-specific thresholds
    thresholds = {
        "[EXON]":      (0.48, 0.68),
        "[INTRON]":    (0.33, 0.50),
        "[PROMOTER]":  (0.42, 0.72),
        "[ENHANCER]":  (0.38, 0.62),
    }
    lo, hi = thresholds.get(condition, (0.35, 0.65))

    bad_indices = []

    if gc_pct < lo:
        # AT-rich trap: find AT runs and re-mask in spans around them
        at_positions = [i for i, nuc in enumerate(seq_str) if nuc in "AT"]
        if at_positions:
            # Target ~20% of the AT positions, clustered into spans
            num_to_mask = max(5, int(len(at_positions) * 0.20))
            sampled = sorted(random.sample(at_positions, min(num_to_mask, len(at_positions))))
            # Convert to small spans (\u00b13 bp around each sampled position)
            for pos in sampled:
                for offset in range(-3, 4):
                    idx = pos + offset
                    if 0 <= idx < L:
                        bad_indices.append(gap_start + idx)

    elif gc_pct > hi:
        # GC-rich trap: find GC runs and re-mask
        cg_positions = [i for i, nuc in enumerate(seq_str) if nuc in "CG"]
        if cg_positions:
            num_to_mask = max(5, int(len(cg_positions) * 0.20))
            sampled = sorted(random.sample(cg_positions, min(num_to_mask, len(cg_positions))))
            for pos in sampled:
                for offset in range(-3, 4):
                    idx = pos + offset
                    if 0 <= idx < L:
                        bad_indices.append(gap_start + idx)

    # Deduplicate
    return list(set(bad_indices))


def check_splice_sites(seq_str, gap_start):
    """
    Validates an Exon-Intron-Exon sandwich centred in the canvas.

    Splicing machinery (and AlphaGenome's convolutional layers) cannot detect
    an intron unless it is immediately flanked by coding exons.  Placing bare
    GT/AG at the canvas edges (positions 0 and 4095) with no flanking exon
    context produces flat splice-usage tracks.

    Architecture (4,096-bp canvas):
      [0 … 999]       → left pseudo-exon  (1,000 bp of generated coding sequence)
      [1000 … 3095]   → intron core       (2,096 bp with GT donor at 1000,
                                            AG acceptor at 3095, branch point +
                                            polypyrimidine tract in tail)
      [3096 … 4095]   → right pseudo-exon (1,000 bp of generated coding sequence)
    """
    bad_indices = []
    L = len(seq_str)
    if L < 2000:
        return bad_indices

    # ── Sandwich boundaries ──
    left_exon_end   = L // 4                # ~1024
    right_exon_start = L - (L // 4)          # ~3072
    intron_str = seq_str[left_exon_end:right_exon_start]
    intron_len = len(intron_str)
    if intron_len < 50:
        return bad_indices

    # ── 1. 5' Splice Donor (first 2 bp of intron core) ──
    donor = intron_str[:2].upper()
    if donor not in ("GT", "GC"):
        # Nuke donor region inside the intron core
        for offset in range(min(8, intron_len)):
            bad_indices.append(gap_start + left_exon_end + offset)

    # ── 2. 3' Splice Acceptor (last 2 bp of intron core) ──
    acceptor = intron_str[-2:].upper()
    if acceptor != "AG":
        # Nuke acceptor + polypyrimidine tract region
        for offset in range(max(0, intron_len - 20), intron_len):
            bad_indices.append(gap_start + left_exon_end + offset)

    # ── 3. Branch point + Polypyrimidine tract (last 50 bp of intron) ──
    tail_start = max(0, intron_len - 50)
    tail = intron_str[tail_start:]
    poly_yy_match = re.search(r'[CT]{8,}', tail[:-2])  # before AG
    branch_match = re.search(r'[CT][ACGT][CT][AT][AG]A[CT]', tail)

    if not poly_yy_match and not branch_match:
        for offset in range(max(0, intron_len - 25), intron_len):
            bad_indices.append(gap_start + left_exon_end + offset)

    # ── 4. Left pseudo-exon: no TATA box (should not be a promoter) ──
    left_exon_str = seq_str[:left_exon_end]
    if len(left_exon_str) > 100 and re.search(r'TATA[AT]{2}[AG]', left_exon_str):
        tata_pos = re.search(r'TATA[AT]{2}[AG]', left_exon_str).start()
        for offset in range(-3, 10):
            idx = tata_pos + offset
            if 0 <= idx < left_exon_end:
                bad_indices.append(gap_start + idx)

    # ── 5. Right pseudo-exon: no TATA box ──
    right_exon_str = seq_str[right_exon_start:]
    if len(right_exon_str) > 100 and re.search(r'TATA[AT]{2}[AG]', right_exon_str):
        tata_pos = re.search(r'TATA[AT]{2}[AG]', right_exon_str).start()
        for offset in range(-3, 10):
            idx = tata_pos + offset
            if 0 <= idx < len(right_exon_str):
                bad_indices.append(gap_start + right_exon_start + idx)

    return list(set(bad_indices))


def check_enhancer_elements(seq_str, gap_start):
    """
    Validates enhancer sequence with centred TF motif cluster architecture.

    Enhancer hallmarks:
      - Dense clusters of TF binding motifs near the centre
      - NO TATA box anywhere (distinguishes from promoter)
      - NO splice sites GT...AG spanning >=30 bp (distinguishes from intron)

    Contamination checks scan the full canvas; motif-density check focuses
    on the central 2,000 bp where the enhancer core should be.
    """
    bad_indices = []
    L = len(seq_str)
    if L < 2000:
        return bad_indices

    # ── 1. Contamination: TATA box (should NOT be anywhere in an enhancer) ──
    tata_hit = re.search(r'TATA[AT]{2}[AG]', seq_str)
    if tata_hit:
        tata_pos = tata_hit.start()
        for offset in range(-5, 13):
            idx = tata_pos + offset
            if 0 <= idx < L:
                bad_indices.append(gap_start + idx)

    # ── 2. Contamination: splice sites (should NOT be in an enhancer) ──
    gt_positions = [i for i in range(L - 1) if seq_str[i:i+2] == "GT"]
    ag_positions = [i for i in range(L - 1) if seq_str[i:i+2] == "AG"]
    for gt in gt_positions[:5]:
        for ag in ag_positions:
            if ag - gt >= 30:
                for offset in range(-3, 8):
                    idx = gt + offset
                    if 0 <= idx < L:
                        bad_indices.append(gap_start + idx)
                break

    # ── 3. TF motif density in the central enhancer core ──
    core_half = min(1000, L // 4)
    core_start = max(0, (L // 2) - core_half)
    core_end   = min(L, (L // 2) + core_half)
    core_str   = seq_str[core_start:core_end]

    enhancer_motifs = [
        r'TGACT[CT]A',                # AP-1 (TRE)
        r'GGG[AG][ACGT]{3}[CT]CC',   # NF-kB
        r'TTC[ACGT]{3}GAA',           # STAT
        r'[AC]GGA[AT]',               # ETS (core)
        r'TG[AT][CT]{2}',             # TCF/LEF
        r'[AG]C[ACGT]GG[ACGT]{2}',   # SMAD
    ]
    motif_hits = 0
    for motif in enhancer_motifs:
        motif_hits += len(re.findall(motif, core_str))

    # If too few TF motifs in the core AND no contamination flagged,
    # nuke the central region to encourage motif emergence
    if motif_hits < 3 and not bad_indices:
        for offset in range(-30, 31):
            idx = (L // 2) + offset
            if 0 <= idx < L:
                bad_indices.append(gap_start + idx)

    return list(set(bad_indices))


## Generator Class
class GenomicGenerator:
    def __init__(self, model, tokenizer, device, remasking_strategy="single_token", span_lambda=100):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model.eval()  # Always set to eval mode on init

        valid_remasking_strategies = {"single_token", "poisson_span"}
        if remasking_strategy not in valid_remasking_strategies:
            raise ValueError(
                f"Unknown remasking strategy: {remasking_strategy!r}. "
                f"Expected one of {sorted(valid_remasking_strategies)}."
            )

        # The generation-time re-masking policy is independent of the masking
        # strategy used to train the checkpoint.
        self.remasking_strategy = remasking_strategy
        self.span_lambda = span_lambda
        print(
            f"🧬 Generation using remasking strategy {self.remasking_strategy} "
            f"with span_lambda {self.span_lambda}."
        )

        self.forbidden_ids = list(self.tokenizer.all_special_ids)
        n_id = self.tokenizer.token_to_id.get("N")
        if n_id is not None:
            self.forbidden_ids.append(n_id)
        print(f"🚫 Generator suppressing {len(self.forbidden_ids)} tokens (including N).")

    def generate(self, prompt_context, gap_length=32, steps=20, temperature=1.0, cfg_scale=1.0,
                 negative_prompt="[HUMAN][BACKGROUND]", use_bio_check=False,
                 return_confidence=False, allow_eos=False, capture_phases=False):
        """
        Main entry point.
        - If prompt_context is a STRING: Generates 1 sequence.
        - If prompt_context is a LIST: Generates a batch in parallel (FAST).

        allow_eos : bool
            False (default) — inpainting mode. EOS is blocked inside the gap; it already
            sits at a fixed position in the template outside the gap, so no change needed.
            True  — de novo generation mode (prompt is only condition tokens, no flanks).
            The model may emit EOS at any gap position; everything after EOS is frozen
            as PAD, giving variable-length outputs up to gap_length.

        use_bio_check : bool
            True — activate biological oracles during a confidence window (middle ~50%
            of steps).  Early steps are left unconstrained so the model can build broad
            scaffolding; late steps are left unconstrained for fine-grained resolution.

        capture_phases : bool
            True — record per-step canvas snapshots and bio-check activity for
            phase visualization (implies return_confidence=True).
        """
        return self._generate_masked_diffusion(
            prompt_context, gap_length, steps, temperature, cfg_scale,
            negative_prompt, use_bio_check,
            return_confidence=return_confidence or capture_phases,
            allow_eos=allow_eos, capture_phases=capture_phases,
        )

    def _generate_masked_diffusion(
        self, prompt_context, gap_length, steps, temperature, cfg_scale,
        negative_prompt, use_bio_check, return_confidence=False, allow_eos=False,
        capture_phases=False,
    ):
        # 1. Standardize prompt_context to a list so we can index it later for bio checks
        if isinstance(prompt_context, str):
            prompt_context_list = [prompt_context]
        else:
            prompt_context_list = prompt_context

        def prepare_sequence_batch(prompts_input):
            batch_tensors = []
            for p_str in prompts_input:
                tokens = p_str.split()
                ids = []
                for t in tokens:
                    if t in self.tokenizer.token_to_id:
                        ids.append(self.tokenizer.token_to_id[t])
                    else:
                        encoded = self.tokenizer.encode(t)
                        if hasattr(encoded, "tolist"): encoded = encoded.tolist()
                        ids.extend(encoded)

                bos  = [self.tokenizer.bos_id]
                eos  = [self.tokenizer.eos_id]
                mask = [self.tokenizer.mask_id]
                full_seq = bos + ids + (mask * gap_length) + eos
                batch_tensors.append(full_seq)

            return torch.tensor(batch_tensors, device=self.device), len(bos) + len(ids)

        # Build Batches
        input_pos, gap_start = prepare_sequence_batch(prompt_context_list)
        batch_size = input_pos.shape[0]
        neg_prompts_list = [negative_prompt] * batch_size
        input_neg, _ = prepare_sequence_batch(neg_prompts_list)

        gap_end = gap_start + gap_length

        # Inpainting (allow_eos=False): EOS is blocked in the gap — it sits at a fixed
        # position outside the gap in the template.
        # De novo (allow_eos=True): EOS is allowed anywhere in the gap so the model can
        # emit a stop signal.  Positions after the first EOS are frozen as PAD.
        effective_forbidden = (
            [pid for pid in self.forbidden_ids if pid != self.tokenizer.eos_id]
            if allow_eos else self.forbidden_ids
        )

        # frozen[b, pos] = True once sequence b has committed EOS at or before pos.
        # Frozen positions are never written back by predictions and never re-masked.
        frozen = torch.zeros_like(input_pos, dtype=torch.bool)

        batch_input = torch.cat([input_pos, input_neg], dim=0)

        # for plotting
        confidence_history = []
        prediction_history = []  # track chosen token IDs at gap positions (batch 0 only)

        # ── Bio-check confidence window ──
        # Inpainting (allow_eos=False): flanks provide context, so bio-check can
        # span a wider window whose range scales with gap_ratio.
        # De novo (allow_eos=True): no flanks → conservative middle ~50% of steps.
        if use_bio_check:
            if not allow_eos:
                total_len = input_pos.shape[1]
                gap_ratio = gap_length / total_len
                if gap_ratio > 0.5:
                    # Macroscopic gaps (e.g. 3500bp out of 4096): behaves like de novo
                    bio_start_pct = 0.25
                    bio_end_pct   = 0.75
                elif gap_ratio > 0.1:
                    # Medium gaps (e.g. 500bp–1000bp): flanks provide good stability
                    bio_start_pct = 0.15
                    bio_end_pct   = 0.85
                else:
                    # Micro gaps (e.g. 50bp): flanks dominate, model converges fast
                    bio_start_pct = 0.05
                    bio_end_pct   = 0.90
                bio_check_start = max(1, int(steps * bio_start_pct))
                bio_check_end   = max(bio_check_start + 1, int(steps * bio_end_pct))
            else:
                bio_check_start = max(1, steps * 1 // 4)   # e.g. step 8 of 32
                bio_check_end   = max(bio_check_start + 1, steps * 3 // 4)  # e.g. step 24 of 32

        # ── Phase capture data structures ──
        phase_snapshots = {}     # {batch_idx: {step: dna_string}} — post-re-mask
        pre_mask_snapshots = {}  # {batch_idx: {step: dna_string}} — pre-re-mask
        bio_check_activity = []  # [(step, count_re_masked), ...]

        if capture_phases:
            # Compute the 7 key snapshot steps
            _snap_steps = set()
            _snap_steps.add(0)                                         # initial MASK
            if use_bio_check:
                _snap_steps.add(max(0, bio_check_start - 1))           # last scaffolding
                _snap_steps.add(bio_check_start)                       # first architecture
                _snap_steps.add((bio_check_start + bio_check_end) // 2)  # mid-architecture
                _snap_steps.add(bio_check_end - 1)                     # last architecture
                _snap_steps.add(bio_check_end)                         # first polishing
            _snap_steps.add(steps - 1)                                  # final
            snapshot_steps = sorted(_snap_steps)
            # Remove duplicates (e.g. if bio_check_start == 0)
            snapshot_steps = sorted(set(s for s in snapshot_steps if 0 <= s < steps))
        else:
            snapshot_steps = set()

        # ------------------------------------------------------------------
        # GENERATION LOOP
        # ------------------------------------------------------------------
        for step in range(steps):
            # #linear schedule for unmasking
            # ratio_to_keep = (step + 1) / steps

            # Cosine schedule for unmasking
            step_ratio = (step + 1) / steps
            ratio_to_keep = math.sin(step_ratio * (math.pi / 2))

            num_to_keep = int(gap_length * ratio_to_keep)

            with torch.no_grad():
                out_batch = self.model(batch_input)
                logits_batch = out_batch.logits if hasattr(out_batch, "logits") else out_batch

                logits_batch[:, :, effective_forbidden] = -float("inf")

                # ── EOS penalty: early steps → strong suppression; decays to 0 ──
                # The model was never trained to predict EOS from MASK tokens (EOS
                # was always protected), so EOS logits are uncalibrated in the
                # all-MASK starting regime.  A step-dependent penalty prevents
                # spurious early EOS while still allowing genuine stop signals late.
                if allow_eos:
                    eos_penalty = float(steps - 1 - step) / float(max(steps - 1, 1)) * 8.0
                    logits_batch[:, :, self.tokenizer.eos_id] -= eos_penalty

                logits_pos, logits_neg = logits_batch.chunk(2, dim=0)

                if cfg_scale > 1.0:
                    pos_gap = logits_pos[:, gap_start:gap_end, :]
                    neg_gap = logits_neg[:, gap_start:gap_end, :]
                    guided_gap = neg_gap + cfg_scale * (pos_gap - neg_gap)
                    logits_pos[:, gap_start:gap_end, :] = guided_gap

            logits_pos = torch.clamp(logits_pos, min=-30.0, max=30.0)
            if torch.isnan(logits_pos).any():
                logits_pos = torch.nan_to_num(logits_pos, nan=-30.0)

            logits = logits_pos / temperature
            # # temperature annealing: start more aggressive to prevent early EOS, end at user-specified temp for final sampling
            # current_temp = max(0.8, temperature + (1.0 - (step / steps)))
            # logits = logits_pos / current_temp

            probs  = F.softmax(logits, dim=-1)
            preds  = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), probs.size(1))

            # Write predictions only into currently-MASK and non-frozen gap positions.
            currently_masked = (input_pos == self.tokenizer.mask_id)
            active_write = currently_masked & ~frozen
            next_input_pos = input_pos.clone()
            next_input_pos[active_write] = preds[active_write]

            # ── EOS freezing (de novo only) ────────────────────────────────
            if allow_eos:
                for b in range(batch_size):
                    if frozen[b, gap_start]:  # entire gap already frozen → skip
                        continue
                    gap_vals = next_input_pos[b, gap_start:gap_end]
                    eos_hits = (gap_vals == self.tokenizer.eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_hits) > 0:
                        first_eos_abs = gap_start + eos_hits[0].item()
                        # Require a meaningful fraction of committed tokens before
                        # allowing EOS to freeze.  The threshold scales with gap_length
                        # so short gaps aren't blocked entirely and long gaps aren't
                        # truncated after a handful of tokens.
                        min_committed = max(50, gap_length // 8)
                        committed = (
                            (next_input_pos[b, gap_start:first_eos_abs] != self.tokenizer.mask_id) &
                            (~frozen[b, gap_start:first_eos_abs])
                        ).sum().item()
                        # Also require we are past the early diffusion steps before
                        # EOS can freeze.  Confidence-based re-masking accelerates
                        # convergence, so spurious EOS tokens that slip past the logit
                        # penalty can accumulate committed neighbors too quickly.
                        min_step_for_eos = max(1, steps * 3 // 4)
                        if committed < min_committed or step < min_step_for_eos:
                            continue
                        # EOS position stays as EOS; everything after → PAD
                        if first_eos_abs + 1 < gap_end:
                            next_input_pos[b, first_eos_abs + 1 : gap_end] = self.tokenizer.pad_id
                        # Freeze from EOS onward so they are never re-masked
                        frozen[b, first_eos_abs : gap_end] = True

            next_input_neg = input_neg.clone()
            next_input_neg[:, gap_start:gap_end] = next_input_pos[:, gap_start:gap_end]

            # Confidence of chosen tokens
            chosen_probs = probs.gather(2, preds.unsqueeze(-1)).squeeze(-1)
            gap_probs    = chosen_probs[:, gap_start:gap_end]

            if return_confidence:
                full_vocab_probs = probs[0, gap_start:gap_end, :]
                confidence_history.append(full_vocab_probs.float().cpu().numpy())
                # Track which token was chosen at each gap position (for ATCG-only filtering)
                pred_token_ids = preds[0, gap_start:gap_end].clone().cpu()
                prediction_history.append(pred_token_ids)

            # ── Pre-re-mask capture (right after predictions, before re-masking) ──
            if capture_phases and step in snapshot_steps:
                for b in range(batch_size):
                    pre_ids = next_input_pos[b, gap_start:gap_end].tolist()
                    _chars = []
                    for t in pre_ids:
                        if t == self.tokenizer.mask_id:
                            _chars.append("M")
                        else:
                            tok = self.tokenizer.id_to_token.get(t, "N")
                            _chars.append(tok if len(tok) == 1 and tok in "ACGT" else "N")
                    pre_mask_snapshots.setdefault(b, {})[step] = "".join(_chars)

            # ----------------------------------------------------------
            # RE-MASKING (Confidence + Optional Biological Check)
            # ----------------------------------------------------------
            if step < steps - 1:
                num_to_mask = gap_length - num_to_keep

                # Frozen gap positions get max confidence → topk never picks them
                # for re-masking, preserving their committed EOS / PAD values.
                if allow_eos:
                    frozen_gap = frozen[:, gap_start:gap_end]
                    gap_probs  = gap_probs.clone()
                    gap_probs[frozen_gap] = float("inf")

                if self.remasking_strategy == "single_token":
                    _, low_conf_idx = torch.topk(gap_probs, k=num_to_mask, largest=False, dim=1)
                else:
                    # Confidence-based span re-masking: spans are centered on the
                    # lowest-confidence positions, not random positions.
                    _, sorted_low_conf = torch.topk(
                        gap_probs, k=gap_length, largest=False, dim=1
                    )
                    block_mask_idx = []
                    for b in range(batch_size):
                        mask_indices_for_seq = set()
                        candidate_ptr = 0
                        while len(mask_indices_for_seq) < num_to_mask:
                            span_len = np.random.poisson(self.span_lambda)
                            current_span = int(np.clip(span_len, 1, 300))
                            center = sorted_low_conf[b, candidate_ptr].item()
                            candidate_ptr += 1
                            half = current_span // 2
                            start_idx = max(0, center - half)
                            end_idx   = min(start_idx + current_span, gap_length)
                            for idx in range(start_idx, end_idx):
                                if len(mask_indices_for_seq) < num_to_mask:
                                    mask_indices_for_seq.add(idx)
                                else:
                                    break
                        block_mask_idx.append(list(mask_indices_for_seq))
                    low_conf_idx = torch.tensor(block_mask_idx, device=self.device)

                # 🧬 OPTIONAL BIOLOGY CHECK (gated: middle ~50% of steps only)
                biological_mask = torch.zeros_like(next_input_pos, dtype=torch.bool)
                if use_bio_check and (bio_check_start <= step < bio_check_end):
                    for i in range(batch_size):
                        gen_ids = next_input_pos[i, gap_start:gap_end].tolist()
                        gen_seq = "".join([self.tokenizer.id_to_token.get(t, "N") for t in gen_ids])
                        condition = prompt_context_list[i]
                        bad_idx = []
                        # GC check: always on, condition-aware
                        bad_idx.extend(check_gc_content(gen_seq, gap_start, condition=condition))
                        # Condition-specific structural checks
                        if "[EXON]" in condition:
                            bad_idx.extend(find_premature_stop_codons(gen_seq, gap_start))
                        elif "[PROMOTER]" in condition:
                            bad_idx.extend(check_promoter_motifs(gen_seq, gap_start))
                        elif "[INTRON]" in condition:
                            bad_idx.extend(check_splice_sites(gen_seq, gap_start))
                        elif "[ENHANCER]" in condition:
                            bad_idx.extend(check_enhancer_elements(gen_seq, gap_start))
                        valid_bad_idx = [idx for idx in bad_idx if gap_start <= idx < gap_end]
                        if valid_bad_idx:
                            biological_mask[i, valid_bad_idx] = True

                # Apply Masks — skip frozen positions
                for i in range(batch_size):
                    mask_locs = low_conf_idx[i] + gap_start
                    if allow_eos:
                        mask_locs = mask_locs[~frozen[i, mask_locs]]
                    next_input_pos[i, mask_locs] = self.tokenizer.mask_id
                    next_input_neg[i, mask_locs] = self.tokenizer.mask_id
                    if use_bio_check:
                        bio_locs = biological_mask[i].nonzero(as_tuple=True)[0]
                        if allow_eos:
                            bio_locs = bio_locs[~frozen[i, bio_locs]]
                        next_input_pos[i, bio_locs] = self.tokenizer.mask_id
                        next_input_neg[i, bio_locs] = self.tokenizer.mask_id

            input_pos  = next_input_pos
            input_neg  = next_input_neg
            batch_input = torch.cat([input_pos, input_neg], dim=0)

            # ── Phase capture (after re-masking applied) ──
            if capture_phases:
                # Record bio-check re-mask count for this step (per-sequence average)
                bio_count = int(biological_mask.sum().item()) / max(batch_size, 1) if use_bio_check else 0
                bio_check_activity.append((step, bio_count))

                # Record canvas state at snapshot steps
                if step in snapshot_steps:
                    for b in range(batch_size):
                        gen_ids = input_pos[b, gap_start:gap_end].tolist()
                        # Single-character encoding: M=MASK, A/C/G/T=bases, N=other
                        _chars = []
                        for t in gen_ids:
                            if t == self.tokenizer.mask_id:
                                _chars.append("M")
                            else:
                                tok = self.tokenizer.id_to_token.get(t, "N")
                                _chars.append(tok if len(tok) == 1 and tok in "ACGT" else "N")
                        gen_seq = "".join(_chars)
                        phase_snapshots.setdefault(b, {})[step] = gen_seq

        # ------------------------------------------------------------------
        # DECODE
        # ------------------------------------------------------------------
        decoded_seqs = []
        final_ids    = input_pos.tolist()

        for i in range(batch_size):
            generated_part = final_ids[i][gap_start:gap_end]
            # De novo: truncate at EOS (strip EOS + trailing PAD)
            if allow_eos and self.tokenizer.eos_id in generated_part:
                generated_part = generated_part[:generated_part.index(self.tokenizer.eos_id)]
            tokens = [
                self.tokenizer.id_to_token.get(t, "N") for t in generated_part
                if t != self.tokenizer.pad_id
            ]
            decoded_seqs.append("".join(tokens))

        if capture_phases:
            result = decoded_seqs[0] if isinstance(prompt_context, str) else decoded_seqs
            return result, confidence_history, prediction_history, phase_snapshots, pre_mask_snapshots, bio_check_activity
        elif return_confidence:
            return decoded_seqs[0] if isinstance(prompt_context, str) else decoded_seqs, confidence_history, prediction_history
        else:
            return decoded_seqs[0] if isinstance(prompt_context, str) else decoded_seqs

