import itertools
import random
import torch
import numpy as np
import re
import ctypes
import multiprocessing as mp


# ── Inverse Curriculum Masking Ratio Scheduler ──────────────────────────────────
# Shared across DataLoader worker processes via multiprocessing.Value.
_curriculum_stage = mp.Value(ctypes.c_int, 0)   # 0 = disabled, 1/2/3 = phases

def set_curriculum_stage(stage: int):
    """Called by the trainer at phase boundaries."""
    _curriculum_stage.value = stage

def get_curriculum_stage() -> int:
    """Called inside DiffusionCollateFn.__call__."""
    return _curriculum_stage.value


def sample_inverse_curriculum_mask(stage: int,
                                   mixture_random_fraction: float = 0.0) -> float:
    """
    Sample a masking ratio according to the inverse-curriculum schedule.

    Phase 1 — Global Scaffolding (mostly very high masking, 60–100%):
        Practises the extreme masking regime the model sees at the start of
        de novo generation.  A fraction of batches get uniform random for calibration.

    Phase 2 — Structural Grammar (moderate masking, 20–70 %):
        Trains local motif grammar with a realistic spread of mask density.

    Phase 3 — High-Res Refinement + De Novo:
        Random mix: 10 % full-mask (100 %) de novo practice, 90 % low mask (5–30 %).

    mixture_random_fraction: config value controlling calibration fraction
        (e.g. 0.2 → 20 % of batches use uniform [0,1] ratios).

    Returns a float in [0.0, 1.0].
    """
    r = random.random()

    if stage == 1:
        # mixture_random_fraction uniform, rest high masking (0.60–1.00)
        if r < mixture_random_fraction:
            return random.uniform(0.0, 1.0)
        else:
            return random.uniform(0.60, 1.0)

    elif stage == 2:
        # mixture_random_fraction uniform, rest moderate masking (0.20–0.70)
        if r < mixture_random_fraction:
            return random.uniform(0.0, 1.0)
        else:
            return random.uniform(0.20, 0.70)

    elif stage == 3:
        # Random mixed schedule: 10 % full-mask de novo, rest low mask (5–30 %)
        if r < 0.10:
            return 1.0
        else:
            return random.uniform(0.05, 0.30)

    else:
        # Disabled → uniform random (original behaviour)
        return random.uniform(0.0, 1.0)

from .tokenizer import GenomicTokenizer

# --- Collate Fn ---
class DiffusionCollateFn:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_id
        self.mask_id = tokenizer.mask_id
        self.multi_mask_factor = config.get("multi_mask_factor", 1)
        
        # --- ABLATION FLAGS ---
        self.masking_strategy = config.get("masking_strategy", "random")
        self.span_lambda = config.get("span_lambda", 100)
        self.adaptive_sliding_window = config.get("adaptive_sliding_window", 9)
        self.mixture_random_fraction = config.get("mixture_random_fraction", 0.0)

        # --- Inverse Curriculum Masking Ratio ---
        self.curriculum_enabled = config.get("curriculum_stage", 0) > 0
        
        # --- Identify Protected Tokens (including N — never mask or predict N) ---
        self.protected_ids = set(self.tokenizer.all_special_ids)
        n_id = self.tokenizer.token_to_id.get("N")
        if n_id is not None:
            self.protected_ids.add(n_id)

        # --- Build Detailed Mode String ---
        if self.mixture_random_fraction > 0:
            mode_str = f"Context-Adaptive Span (W={self.adaptive_sliding_window}, λ={self.span_lambda}) + {self.mixture_random_fraction:.0%} Random"
        elif self.masking_strategy == "span":
            mode_str = f"Poisson Span (λ={self.span_lambda})"
        elif self.masking_strategy == "adaptive_span":
            mode_str = f"Context-Adaptive Span Masking (W={self.adaptive_sliding_window}, λ={self.span_lambda})"
        elif self.masking_strategy == "adaptive":
            mode_str = f"Entropy-Weighted Token Masking (W={self.adaptive_sliding_window}, no spans)"
        else:
            mode_str = "Uniform Random Token"

        # --- Print Initialization Log ---
        curriculum_str = f"\n   ├── Curriculum     : Inverse 3-phase (scaffolding→grammar→refinement)" if self.curriculum_enabled else ""
        print(
            f"🛡️  LLaDA CollateFn Initialized\n"
            f"   ├── Masking Mode   : {mode_str}\n"
            f"   ├── Protected IDs  : {len(self.protected_ids)} tokens (BOS, EOS, Prompts){curriculum_str}\n"
            f"   └── Multi-masking  : {self.multi_mask_factor}x"
        )

    def __call__(self, batch):
        sequences = [item["input_ids"] for item in batch]
        metadata = [item.get("metadata", "unknown") for item in batch]
        conditions = [item.get("condition", "unknown") for item in batch]

        if self.multi_mask_factor > 1:
            sequences = sequences * self.multi_mask_factor
            metadata = metadata * self.multi_mask_factor
            conditions = conditions * self.multi_mask_factor

        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.pad_id
        )
        original_input_ids = input_ids.clone() # Keep a pristine copy

        attention_mask = (input_ids != self.pad_id).long()
        batch_size, seq_len = input_ids.shape
        labels = input_ids.clone()

        # 1. Identify Safe Zones
        special_mask = input_ids == self.pad_id
        for pid in self.protected_ids:
            special_mask = special_mask | (input_ids == pid)
        
        # 2. Sample Masking Ratios (curriculum or uniform)
        if self.curriculum_enabled:
            stage = get_curriculum_stage()
            ratios_float = [sample_inverse_curriculum_mask(
                stage, self.mixture_random_fraction) for _ in range(batch_size)]
            ratios = torch.tensor(ratios_float, device=input_ids.device, dtype=torch.float32)
        else:
            ratios = torch.rand(batch_size, device=input_ids.device)

        # --- Mixture: occasionally use random masking for calibration ---
        if self.mixture_random_fraction > 0 and random.random() < self.mixture_random_fraction:
            effective_strategy = "random"
        else:
            effective_strategy = self.masking_strategy

        # =====================================================
        # BRANCH: SPAN MASKING vs. ADAPTIVE SPAN MASKING vs. RANDOM MASKING 
        # =====================================================
        if effective_strategy == "span":
            # --- POISSON SPAN MASKING ---
            final_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)
            valid_mask_locations = ~special_mask
            
            for i in range(batch_size):
                num_valid = valid_mask_locations[i].sum().item()
                target_count = int(num_valid * ratios[i].item())
                if target_count == 0: continue

                # Fast path: ratio=1.0 → mask everything, no span picking needed
                if ratios[i].item() >= 1.0:
                    final_mask[i] = valid_mask_locations[i]
                    continue

                current_count = 0
                valid_indices = torch.nonzero(valid_mask_locations[i], as_tuple=True)[0]
                if len(valid_indices) == 0: continue

                attempts = 0
                while current_count < target_count and attempts < target_count * 2:
                    attempts += 1
                    span_len = max(1, np.random.poisson(self.span_lambda))
                    
                    rand_idx = torch.randint(0, len(valid_indices), (1,)).item()
                    start_pos = valid_indices[rand_idx].item()
                    end_pos = min(start_pos + span_len, seq_len)
                    
                    span_indices = torch.arange(start_pos, end_pos, device=input_ids.device)
                    
                    # Check collision with protected tokens
                    if special_mask[i, span_indices].any():
                        continue
                        
                    # Apply
                    new_masks = (~final_mask[i, span_indices]).sum().item()
                    final_mask[i, span_indices] = True
                    current_count += new_masks
            
            mask = final_mask

        elif effective_strategy == "adaptive_span":
            # ---------------------------------------------------------
            #  ADAPTIVE SPAN MASKING (Density-Optimized)
            # ---------------------------------------------------------
            W = self.adaptive_sliding_window
            vocab_size = len(self.tokenizer.vocab)
            
            # 1. Identify "Real" Nucleotide IDs (A, C, G, T)
            # We filter out N, MASK, PAD, and context tokens
            nucleotide_bases = ["A", "C", "G", "T"]
            nuc_ids = [self.tokenizer.token_to_id[b] for b in nucleotide_bases if b in self.tokenizer.token_to_id]
            nuc_ids_tensor = torch.tensor(nuc_ids, device=input_ids.device)
            
            # 2. Calculate Local Counts for ALL tokens
            one_hot = torch.nn.functional.one_hot(input_ids, num_classes=vocab_size).float()
            one_hot_t = one_hot.transpose(1, 2)
            kernel = torch.ones(vocab_size, 1, W, device=input_ids.device)
            local_counts = torch.nn.functional.conv1d(one_hot_t, kernel, padding=W//2, groups=vocab_size)
            
            # 3. Filter counts to ONLY include A, C, G, T for entropy
            # Shape: (batch, 4, seq_len)
            nuc_counts = local_counts[:, nuc_ids_tensor, :]
            
            # Re-normalize: what is the distribution of ACGT relative to EACH OTHER?
            nuc_sums = nuc_counts.sum(dim=1, keepdim=True) + 1e-6
            P_nuc = nuc_counts / nuc_sums
            
            # Calculate Shannon Entropy on the 4-base distribution
            # If a window is all 'N's, P_nuc will be flat/zeros, and entropy will be low.
            local_entropy = -torch.sum(P_nuc * torch.log(P_nuc + 1e-6), dim=1) 
            
            # 4. Penalty for non-DNA regions
            # If the window is mostly Ns or PADs, we want the weight to be near zero.
            dna_density = nuc_sums.squeeze(1) / W # Ratio of ACGT in the window
            weight_dist = (local_entropy ** 2) * dna_density 
            
            # 5. Span-Density Smoothing
            expected_span_len = int(self.span_lambda)
            if expected_span_len % 2 == 0:
                expected_span_len += 1  # Force odd number for padding='same'

            smoothing_kernel = torch.ones(1, 1, expected_span_len, device=input_ids.device) / expected_span_len
            
            smoothed_weights = torch.nn.functional.conv1d(
                weight_dist.unsqueeze(1), 
                smoothing_kernel, 
                padding="same"
            ).squeeze(1)
            
            final_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)
            valid_mask_locations = ~special_mask
            
            # 4. Drop the Spans
            for i in range(batch_size):
                num_valid = valid_mask_locations[i].sum().item()
                target_count = int(num_valid * ratios[i].item())
                if target_count == 0: continue

                # Fast path: ratio=1.0 → mask everything, no entropy/span needed
                if ratios[i].item() >= 1.0:
                    final_mask[i] = valid_mask_locations[i]
                    continue

                # Sample from the SMOOTHED weights instead of the raw weights!
                seq_weights = smoothed_weights[i].clone()
                seq_weights[~valid_mask_locations[i]] = 0.0 
                
                if seq_weights.sum() <= 1e-6:
                    seq_weights[valid_mask_locations[i]] = 1.0
                
                current_count = 0
                attempts = 0
                
                while current_count < target_count and attempts < target_count * 2:
                    attempts += 1
                    
                    # 🎯 Target Acquired: Picks a start_pos that guarantees the highest AVERAGE entropy in the coming span!
                    start_pos = torch.multinomial(seq_weights, 1).item()
                    
                    # Change to variable Span Length
                    # Using Poisson(lambda) provides natural variation. 
                    # We clamp it to stay within a biologically meaningful range.
                    span_len = np.random.poisson(self.span_lambda)
                    span_len = np.clip(span_len, 1, 300) # Ensure spans aren't tiny or massive
                    end_pos = min(start_pos + span_len, seq_len)
                    
                    span_indices = torch.arange(start_pos, end_pos, device=input_ids.device)
                    
                    if special_mask[i, span_indices].any():
                        continue
                        
                    new_masks = (~final_mask[i, span_indices]).sum().item()
                    final_mask[i, span_indices] = True
                    current_count += new_masks
            
            mask = final_mask

        elif effective_strategy == "adaptive":
            # ---------------------------------------------------------
            # ENTROPY-WEIGHTED TOKEN MASKING (no spans)
            # Same entropy + dna_density weights as adaptive_span,
            # but tokens are sampled individually by weight — no smoothing,
            # no span extension.
            # ---------------------------------------------------------
            W = self.adaptive_sliding_window
            vocab_size = len(self.tokenizer.vocab)

            nucleotide_bases = ["A", "C", "G", "T"]
            nuc_ids = [self.tokenizer.token_to_id[b] for b in nucleotide_bases if b in self.tokenizer.token_to_id]
            nuc_ids_tensor = torch.tensor(nuc_ids, device=input_ids.device)

            one_hot = torch.nn.functional.one_hot(input_ids, num_classes=vocab_size).float()
            one_hot_t = one_hot.transpose(1, 2)
            kernel = torch.ones(vocab_size, 1, W, device=input_ids.device)
            local_counts = torch.nn.functional.conv1d(one_hot_t, kernel, padding=W//2, groups=vocab_size)

            nuc_counts = local_counts[:, nuc_ids_tensor, :]
            nuc_sums   = nuc_counts.sum(dim=1, keepdim=True) + 1e-6
            P_nuc      = nuc_counts / nuc_sums

            local_entropy = -torch.sum(P_nuc * torch.log(P_nuc + 1e-6), dim=1)
            dna_density   = nuc_sums.squeeze(1) / W
            weight_dist   = (local_entropy ** 2) * dna_density

            final_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)
            valid_mask_locations = ~special_mask

            for i in range(batch_size):
                num_valid    = valid_mask_locations[i].sum().item()
                target_count = int(num_valid * ratios[i].item())
                if target_count == 0:
                    continue

                # Fast path: ratio=1.0 → mask everything, no entropy sampling needed
                if ratios[i].item() >= 1.0:
                    final_mask[i] = valid_mask_locations[i]
                    continue

                seq_weights = weight_dist[i].clone()
                seq_weights[~valid_mask_locations[i]] = 0.0
                if seq_weights.sum() <= 1e-6:
                    seq_weights[valid_mask_locations[i]] = 1.0

                # Sample exactly target_count distinct tokens, no replacement
                target_count = min(target_count, int(seq_weights.gt(0).sum().item()))
                chosen = torch.multinomial(seq_weights, target_count, replacement=False)
                final_mask[i, chosen] = True

            mask = final_mask

        elif effective_strategy == "random":
            # ---RANDOM TOKEN MASKING (Original/Fast) ---
            mask_probs = torch.rand(batch_size, seq_len, device=input_ids.device)
            mask = mask_probs < ratios.unsqueeze(-1)
            mask = mask & ~special_mask

        else:
            raise ValueError(
                "Unknown masking strategy! Available dataset options are: span, adaptive_span, adaptive, random"
            )

        # =====================================================

        # Apply Mask
        input_ids[mask] = self.mask_id
        labels[~mask] = -100 # Ignore unmasked in loss

        # Diagnostic: Prevent Zero-Loss crash
        active_tokens = (labels != -100).sum().item()
        if active_tokens == 0:
            valid_indices = (~special_mask).nonzero(as_tuple=False)
            if len(valid_indices) > 0:
                b_idx, s_idx = valid_indices[0]
                input_ids[b_idx, s_idx] = self.mask_id
                labels[b_idx, s_idx] = original_input_ids[b_idx, s_idx] # 100% safe tensor assignment

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "avg_mask_ratio": ratios.mean().item(),
            "std_mask_ratio": ratios.std().item() if ratios.numel() > 1 else 0.0,
            "metadata": metadata,
            "condition": conditions,
        }

