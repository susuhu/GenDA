#!/usr/bin/env python3
"""
In-loop ClinVar variant-effect validation for the GenDA masked-diffusion DNA model.

================================================================================================
INTEGRATION — the only changes needed elsewhere (NOT applied here; add it in the repo).

(1) Build it once on the main rank.  The parquet path is REQUIRED — error if `clinvar_eval` is on
    but no path is given (in train.py or GenomicTrainer.__init__):

    from clinvar_inloop_eval import ClinVarInLoopEvaluator
    self.clinvar_eval = None
    if config.get("clinvar_eval", False):
        parquet = config.get("clinvar_eval_parquet")
        if not parquet:
            raise ValueError("config 'clinvar_eval: true' requires 'clinvar_eval_parquet' to be set.")
        self.clinvar_eval = ClinVarInLoopEvaluator(
            parquet_path=parquet, tokenizer=tokenizer, device=self.device,
            n=config.get("clinvar_eval_n", None), win=config.get("clinvar_eval_win", 4096),
            batch_size=config.get("clinvar_eval_bs", 64))
        self.clinvar_select = config.get("clinvar_select_region", "overall")  # which metric early-stops
        self.best_clinvar = 0.0

(2) In utils/train_utils.py, inside the `if global_step % val_interval == 0:` / `if is_main_process:`
    block (after val_loss), score + log + DRIVE EARLY STOPPING by ClinVar AUROC instead of loss:

    if self.clinvar_eval is not None:
        cv = self.clinvar_eval.evaluate(self.model)
        by = "  ".join(f"{r}={a:.3f}" for r,(a,_) in cv["clinvar_auroc_by_region"].items())
        print(f"   ClinVar AUROC: {cv['clinvar_auroc']:.4f} | {by}")
        if self.use_wandb:
            wandb.log({"clinvar_auroc": cv["clinvar_auroc"], "clinvar_auprc": cv["clinvar_auprc"],
                       **{f"clinvar_auroc_{r}": a for r,(a,_) in cv["clinvar_auroc_by_region"].items()},
                       "global_step": global_step})
        sel = cv["clinvar_auroc"] if self.clinvar_select == "overall" \
              else cv["clinvar_auroc_by_region"][self.clinvar_select][0]
        if sel > self.best_clinvar:                       # <-- early-stop on the FUNCTIONAL metric
            self.best_clinvar = sel; patience_counter = 0
            save_checkpoint(..., os.path.join(log_dir, "best_clinvar_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping on ClinVar AUROC plateau"); break

    (Replace the existing val_loss-based `best_val_loss`/`patience_counter` block, or keep loss-based
     checkpointing in parallel and let ClinVar own `patience_counter`.)

Suggested config keys: clinvar_eval (bool), clinvar_eval_parquet (REQUIRED path), clinvar_eval_n,
clinvar_eval_win, clinvar_eval_bs, clinvar_select_region ("overall" | exon | intron | promoter | ...).
================================================================================================
"""
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

_REQUIRED_COLS = {"Context_Seq_WT", "Context_Seq_Masked", "Ref", "Alt", "Label"}
_VARIANT_KEYS = ["Chromosome", "Position", "Ref", "Alt"]


class ClinVarInLoopEvaluator:
    """Precompute a fixed ClinVar SNV probe set once; score any GenDA model with `.evaluate(model)`.

    Required parquet columns: Context_Seq_WT, Context_Seq_Masked, Ref, Alt, Label.
    Optional: a `type` column (functional region: exon/intron/promoter/enhancer/repeat) enables the
    per-region breakdown; Chromosome/Position enable variant dedup. Raises on a missing/empty/malformed
    parquet so a misconfigured run fails fast instead of silently skipping the metric.
    """

    def __init__(self, parquet_path, tokenizer, device, n=None, win=4096,
                 batch_size=64, seed=0, balanced=False, verbose=True):
        if not parquet_path:
            raise ValueError("ClinVarInLoopEvaluator: parquet_path is required (got empty/None).")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"ClinVarInLoopEvaluator: parquet not found: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"ClinVarInLoopEvaluator: parquet {parquet_path} missing columns {missing}")

        self.tok = tokenizer
        self.device = device
        self.bs = batch_size
        self.win = win
        self.has_region = "type" in df.columns

        # collapse the (variant x region) rows to UNIQUE variants, keeping the set of regions per variant
        key_cols = [c for c in _VARIANT_KEYS if c in df.columns] or ["Ref", "Alt", "Context_Seq_WT"]
        region_of = {}
        if self.has_region:
            for key, grp in df.groupby(key_cols):
                region_of[key if isinstance(key, tuple) else (key,)] = sorted(set(grp["type"]))
        df_u = df.drop_duplicates(subset=key_cols).reset_index(drop=True)

        items = []  # (token_ids_with_masked_variant, var_offset, ref_id, alt_id, label, regions_tuple)
        for _, r in df_u.iterrows():
            wt, mk = r["Context_Seq_WT"], r["Context_Seq_Masked"]
            off = 0
            while off < len(wt) and off < len(mk) and wt[off] == mk[off]:
                off += 1
            if off >= len(wt) or wt[off] != r["Ref"]:
                continue
            ref_id = tokenizer.token_to_id.get(r["Ref"])
            alt_id = tokenizer.token_to_id.get(r["Alt"])
            if ref_id is None or alt_id is None:
                continue
            lo = max(0, off - win // 2)
            hi = min(len(wt), lo + win)
            lo = max(0, hi - win)
            ids = [tokenizer.token_to_id.get(b, tokenizer.unk_id) for b in wt[lo:hi]]
            voff = off - lo
            ids[voff] = tokenizer.mask_id  # mask ONLY the variant position (span_radius=0)
            kk = tuple(r[c] for c in key_cols)
            regions = tuple(region_of.get(kk, [])) if self.has_region else ()
            items.append((ids, voff, ref_id, alt_id, int(r["Label"]), regions))

        # optional fixed subsample (for speed); default uses ALL variants so it reproduces offline
        rng = np.random.RandomState(seed)
        y = np.array([it[4] for it in items])
        if n and len(items) > n:
            if balanced and 0 < y.sum() < len(y):
                idx = []
                for c in (0, 1):
                    ci = np.where(y == c)[0]
                    idx += rng.choice(ci, min(len(ci), n // 2), replace=False).tolist()
                rng.shuffle(idx)
            else:
                idx = rng.choice(len(items), n, replace=False).tolist()
            items = [items[i] for i in idx]

        self.items = items
        self.labels = np.array([it[4] for it in items])
        self.regions = [it[5] for it in items]
        if self.labels.size == 0 or self.labels.sum() in (0, len(self.labels)):
            raise ValueError("ClinVarInLoopEvaluator: probe set is empty or single-class — check parquet/labels.")
        if verbose:
            reg = sorted({r for rs in self.regions for r in rs})
            print(f"[ClinVarInLoopEvaluator] {len(items)} variants "
                  f"({int(self.labels.sum())} path / {int((1 - self.labels).sum())} benign), "
                  f"win={win}, regions={reg or 'none'}, parquet={parquet_path}", flush=True)

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    @torch.no_grad()
    def evaluate(self, model):
        """Return {'clinvar_auroc','clinvar_auprc','clinvar_auroc_by_region','clinvar_n'}.

        clinvar_auroc_by_region maps region -> (auroc, n). Read-only; restores train mode.
        """
        m = self._unwrap(model)
        was_training = m.training
        m.eval()
        tok = self.tok
        scores = np.empty(len(self.items), dtype=np.float64)
        k = 0
        for i in range(0, len(self.items), self.bs):
            chunk = self.items[i:i + self.bs]
            L = max(len(it[0]) for it in chunk)
            ids = torch.full((len(chunk), L), tok.pad_id, dtype=torch.long, device=self.device)
            for j, it in enumerate(chunk):
                ids[j, :len(it[0])] = torch.tensor(it[0], device=self.device)
            attn = (ids != tok.pad_id).long()
            out = m(input_ids=ids, attention_mask=attn)
            logits = out.logits if hasattr(out, "logits") else out[0]
            logp = torch.log_softmax(logits.float(), dim=-1)
            for j, it in enumerate(chunk):
                _, voff, ref_id, alt_id, _, _ = it
                lp = logp[j, voff]
                scores[k] = (lp[ref_id] - lp[alt_id]).item()  # Inverted_LLR; pathogenic = high
                k += 1
        if was_training:
            m.train()

        y = self.labels
        by_region = {}
        if self.has_region:
            all_regions = sorted({r for rs in self.regions for r in rs})
            for reg in all_regions:
                mask = np.array([reg in rs for rs in self.regions])
                yr = y[mask]
                if yr.size and 0 < yr.sum() < yr.size:
                    by_region[reg] = (float(roc_auc_score(yr, scores[mask])), int(yr.size))
        return {
            "clinvar_auroc": float(roc_auc_score(y, scores)),
            "clinvar_auprc": float(average_precision_score(y, scores)),
            "clinvar_auroc_by_region": by_region,
            "clinvar_n": int(len(scores)),
        }


# ----------------------------------------------------------------------------------------------
# Standalone smoke test: verify in-loop AUROC reproduces the offline ClinVar AUROC, overall + per region.
#   python3 clinvar_inloop_eval.py <best_model.pt> <run_config.yaml> <clinvar_SNV.parquet>
# ----------------------------------------------------------------------------------------------
