import os
import time
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F

from genda.data.masking import GenomicTokenizer
from genda.checkpoints import save_checkpoint

from .tracking import wandb


# --- Trainer Class ---
class GenomicTrainer:
    def __init__(self, model, tokenizer, device, accumulation_steps=1, use_amp=True,
                 use_span_weighting=False, span_lambda=100, max_span_weight=3.0,
                 entropy_exponent=2.0, clinvar_eval=None, clinvar_select="overall",
                 curriculum_stage=0):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.best_val_loss = float("inf")
        self.accumulation_steps = accumulation_steps
        self.is_main = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

        # --- ClinVar In-Loop Evaluation ---
        self.clinvar_eval = clinvar_eval
        self.clinvar_select = clinvar_select
        self.best_clinvar = 0.0

        # --- Inverse Curriculum Masking Ratio (manual stage gating) ---
        self.curriculum_stage = curriculum_stage
        if curriculum_stage > 0:
            from genda.data.masking import set_curriculum_stage
            set_curriculum_stage(curriculum_stage)
            phase_names = {1: "Global Scaffolding (60–100% mask)",
                           2: "Structural Grammar (20–70% mask)",
                           3: "High-Res Refinement (10% full-mask, 90% low 5-30%)"}
            print(f"🎓 Inverse Curriculum — Phase {curriculum_stage}: {phase_names[curriculum_stage]}")
            print(f"   (Manual mode — advance via --curriculum_stage when biological triggers are met)")
        
        # Mixed Precision Scaler (Recommended for ModernBERT)
        # Determine the best available mixed precision format
        self.use_amp = use_amp and device.type == "cuda"
        if self.use_amp:
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.amp_dtype = torch.float32

        self.use_span_weighting = use_span_weighting
        self.span_lambda = span_lambda
        self.max_span_weight = max_span_weight
        self.entropy_exponent = entropy_exponent
        if self.use_span_weighting:
            exponent_label = f"H^{entropy_exponent}" if entropy_exponent != 1.0 else "H"
            print(f"⚖️ Span-Weighted Loss Function ENABLED (entropy={exponent_label}).")
            # Dynamically fetch the token IDs for the 4 primary bases from your tokenizer
            valid_bases = ["A", "C", "G", "T"]
            valid_ids = [tokenizer.token_to_id[b] for b in valid_bases if b in tokenizer.token_to_id]
            self.acgt_ids = torch.tensor(valid_ids, dtype=torch.long, device=device)

            # If using k-mers > 1, you would expand valid_ids to include all pure ACGT k-mer combinations
            if tokenizer.kmer_size > 1:
                raise NotImplementedError("Only 1-mer tokenization is supported!")

            # Precompute normalization constant: max H^k for a 4-state uniform distribution
            max_h = torch.log(torch.tensor(len(valid_ids), dtype=torch.float32))  # ln(4)
            self.max_h_pow = max_h ** entropy_exponent
            
        #  Only use GradScaler if we are forced to use float16
        # bfloat16 has the same range as float32, so scaling is not needed and can cause bugs.
        self.needs_scaler = (self.amp_dtype == torch.float16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.needs_scaler)

        # Watch the model only on rank 0 (wandb.init is only called on rank 0)
        if wandb.run is not None:
            wandb.watch(self.model, log=None, log_freq=100)

    def calculate_accuracy(self, logits, labels):
        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        return correct, total

    def train(self, train_loader, val_loader, optimizer, scheduler, criterion, 
              total_steps, log_dir, val_interval=1000, patience=5, start_step=0):
        
        print(f"🚀 Starting Step-Based Training for {total_steps} optimizer steps...")
        self.model.train()

        # Counters
        global_step = start_step
        micro_step = 0
        patience_counter = 0
        accumulation_loss = 0

        # Token throughput tracking
        _world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        total_tokens = 0
        tokens_this_accum = 0
        step_start_time = time.time()

        # Infinite Iterator setup
        epoch = 0
        train_iter = iter(train_loader)
        optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(total=total_steps, desc="Training Steps", initial=start_step)

        while global_step < total_steps:
            # --- 1. Fetch Batch ---
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)
   
            micro_step += 1

            # --- 2. Move to Device ---
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            avg_mask_ratio = batch.get("avg_mask_ratio", None)
            std_mask_ratio = batch.get("std_mask_ratio", None)
            tokens_this_accum += input_ids.numel() * _world_size

            # --- 3. Forward with AMP ---
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                # self-defined span-weighted loss (if enabled). We do not pass labels to the model, because we need to calculate the loss manually to apply the weights.
                if self.use_span_weighting:
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    ce_loss_unreduced = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        labels.view(-1), 
                        ignore_index=-100, 
                        reduction='none'
                    ).view(labels.size(0), labels.size(1))
                    
                    weights = torch.ones_like(labels, dtype=logits.dtype)
                    
                    for b in range(labels.size(0)):
                        mask_flags = (labels[b] != -100)
                        if not mask_flags.any(): continue
                        
                        padded = F.pad(mask_flags, (1, 1), value=False)
                        diff = padded[1:].int() - padded[:-1].int()
                        starts = (diff == 1).nonzero(as_tuple=True)[0]
                        ends = (diff == -1).nonzero(as_tuple=True)[0]
                        
                        for start, end in zip(starts, ends):
                            span_length = (end - start).item()
                            span_labels = labels[b, start:end]
                            
                            # --- 1. Calculate Density (D_i) ---
                            is_valid_base = torch.isin(span_labels, self.acgt_ids)
                            valid_count = is_valid_base.sum().float()
                            D_i = valid_count / span_length if span_length > 0 else 0.0

                            if valid_count > 0:
                                # --- 2. Calculate Probabilities (p_i) ---
                                acgt_tokens = span_labels[is_valid_base]
                                _, counts = torch.unique(acgt_tokens, return_counts=True)
                                p_i = counts.float() / valid_count

                                # --- 3. Calculate Shannon Entropy (H) ---
                                H = -torch.sum(p_i * torch.log(p_i + 1e-9))
                            else:
                                H = torch.tensor(0.0, device=self.device)

                            # --- 4. Apply the Formula: D_i * H^k ---
                            raw_entropy_weight = D_i * (H ** self.entropy_exponent)

                            # --- 5. Normalize and Apply to Span ---
                            normalized_entropy = raw_entropy_weight / self.max_h_pow

                            # Combine expected span length parameter with the calculated entropy density
                            length_ratio = min(span_length, self.span_lambda) / self.span_lambda
                            w = 1.0 + (length_ratio * normalized_entropy * (self.max_span_weight - 1.0))

                            weights[b, start:end] = w
                            
                    weighted_loss = ce_loss_unreduced * weights
                    valid_tokens = (labels != -100).sum()
                    loss = weighted_loss.sum() / valid_tokens if valid_tokens > 0 else weighted_loss.sum()
                # default CrossEntropy (no span weighting)
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

            # Scale & Backward
            loss_scaled = loss / self.accumulation_steps
            self.scaler.scale(loss_scaled).backward()
            
            accumulation_loss += loss.item()

            # --- 4. Optimizer Step ---
            if micro_step % self.accumulation_steps == 0:
                # Unscale before clipping
                self.scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Step
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                # Token throughput
                now = time.time()
                elapsed = now - step_start_time
                tokens_per_sec = tokens_this_accum / elapsed if elapsed > 0 else 0
                total_tokens += tokens_this_accum
                tokens_this_accum = 0
                step_start_time = now

                # Logging
                avg_loss = accumulation_loss / self.accumulation_steps
                current_lr = scheduler.get_last_lr()[0]

                # --- Efficient CFG Audit (Calculate Once) ---
                # Check the batch on CPU (fast)
                batch_cpu = batch["input_ids"]
                batch_sz = batch_cpu.size(0)
                functional_tags = ["[PROMOTER]", "[EXON]", "[BACKGROUND]"]
                
                drop_count = 0
                dropped_example = None
                
                # Scan batch
                for i in range(batch_sz):
                    # Check first 10 tokens
                    tokens = [self.tokenizer.id_to_token.get(idx.item()) for idx in batch_cpu[i][:10]]
                    if not any(tag in tokens for tag in functional_tags):
                        drop_count += 1
                        if dropped_example is None: 
                            dropped_example = " ".join(tokens) # Save for print

                cfg_drop_rate = (drop_count / batch_sz) * 100

                if wandb.run is not None:
                    log_dict = {
                        "train_loss": avg_loss,
                        "lr": current_lr,
                        "global_step": global_step,
                        "cfg_dropout_pct": cfg_drop_rate,
                        "grad_norm": grad_norm.item(),
                        "total_tokens": total_tokens,
                        "tokens_per_sec": tokens_per_sec,
                    }
                    if avg_mask_ratio is not None:
                        log_dict["mask_ratio/mean"] = avg_mask_ratio
                    if std_mask_ratio is not None:
                        log_dict["mask_ratio/std"] = std_mask_ratio
                    wandb.log(log_dict)

                # Reset Accumulator
                accumulation_loss = 0
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=avg_loss, tok_s=f"{tokens_per_sec:,.0f}")

                # --- Console Print (Optional: Run every 500 steps) ---
                if global_step % 500 == 0:
                    print(f"\n📊 [Step {global_step}] CFG Batch Audit:")
                    print(f"   Batch Size: {batch_sz}")
                    print(f"   Dropped (Unconditional): {drop_count} samples ({cfg_drop_rate:.1f}%)")
                    if dropped_example:
                        print(f"   Example Dropped: {dropped_example}")

                # --- Periodic Checkpoint (every 500 steps, rank 0 only) ---
                if global_step % 500 == 0 and self.is_main:
                    save_checkpoint(
                        model=self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        best_val_loss=self.best_val_loss,
                        filepath=os.path.join(log_dir, f"step_{global_step}_model.pt"),
                    )
                    print(f"   💾 Periodic checkpoint saved: step_{global_step}_model.pt")

                # --- 5. Validation & Checkpointing ---
                if global_step % val_interval == 0:
                    # Sync all ranks BEFORE validation so none is mid-training-step
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()

                    val_loss, val_acc, val_ppl = self.evaluate(val_loader, criterion, mode="Val")

                    # Sync all ranks AFTER validation so no rank starts the next training
                    # step (which issues NCCL allreduce) before all have finished eval.
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()

                    # --- ClinVar In-Loop Evaluation (rank 0 only) ---
                    clinvar_res = None
                    if self.clinvar_eval is not None:
                        clinvar_res = self.clinvar_eval.evaluate(self.model)
                        by = "  ".join(f"{r}={a:.3f}" for r, (a, _) in clinvar_res["clinvar_auroc_by_region"].items())
                        print(f"   🧬 ClinVar AUROC: {clinvar_res['clinvar_auroc']:.4f}  "
                              f"AUPRC: {clinvar_res['clinvar_auprc']:.4f}  n={clinvar_res['clinvar_n']}")
                        if by:
                            print(f"      By region: {by}")

                    # Save 'Last' using your EXTERNAL function (rank 0 only)
                    if self.is_main:
                        save_checkpoint(
                            model=self.model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            global_step=global_step,
                            best_val_loss=self.best_val_loss,
                            filepath=os.path.join(log_dir, "last_model.pt"),
                        )

                    # Save 'Best' & Early Stopping
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        patience_counter = 0
                        print(f"   🎉 New Best Val Loss: {val_loss:.4f}")
                        
                        if self.is_main:
                            save_checkpoint(
                                model=self.model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                global_step=global_step,
                                best_val_loss=self.best_val_loss,
                                filepath=os.path.join(log_dir, "best_model.pt"),
                            )
                    else:
                        patience_counter += 1
                        print(f"   ⚠️ No improvement. Patience: {patience_counter}/{patience}")
                        if patience_counter >= patience:
                            print("   🛑 Early Stopping Triggered.")
                            break

                    # --- Save best ClinVar model (separate from loss-based best) ---
                    if clinvar_res is not None:
                        sel = clinvar_res["clinvar_auroc"] if self.clinvar_select == "overall" \
                              else clinvar_res["clinvar_auroc_by_region"].get(self.clinvar_select, (0.0, 0))[0]
                        if sel > self.best_clinvar:
                            self.best_clinvar = sel
                            print(f"   🎉 New Best ClinVar AUROC ({self.clinvar_select}): {sel:.4f}")
                            save_checkpoint(
                                model=self.model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                global_step=global_step,
                                best_val_loss=self.best_val_loss,
                                filepath=os.path.join(log_dir, "best_clinvar_model.pt"),
                            )

                    # Log Val
                    if wandb.run is not None:
                        wandb_log_dict = {
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "val_ppl": val_ppl,
                            "patience_counter": patience_counter,
                            "best_val_loss": self.best_val_loss,
                            "global_step": global_step
                        }
                        if clinvar_res is not None:
                            wandb_log_dict["clinvar_auroc"] = clinvar_res["clinvar_auroc"]
                            wandb_log_dict["clinvar_auprc"] = clinvar_res["clinvar_auprc"]
                            wandb_log_dict["clinvar_n"] = clinvar_res["clinvar_n"]
                            for reg, (a, _) in clinvar_res["clinvar_auroc_by_region"].items():
                                wandb_log_dict[f"clinvar_auroc_{reg}"] = a
                        wandb.log(wandb_log_dict)
                    
                    # Return to training mode
                    self.model.train()

        progress_bar.close()
        if self.is_main:
            save_checkpoint(self.model, optimizer, scheduler, global_step, self.best_val_loss,
                            os.path.join(log_dir, "last_model.pt"))
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def evaluate(self, loader, criterion, mode="Val"):
        self.model.eval()
        total_nll_sum = 0
        total_correct = 0
        total_masked = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{mode} Loop", leave=False, disable=not self.is_main):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Use AMP for evaluation too (Faster)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                # Calculate weighted metrics
                active_mask = labels != -100
                num_masked_in_batch = active_mask.sum().item()

                if num_masked_in_batch > 0:
                    total_nll_sum += loss.item() * num_masked_in_batch
                    corr, tot = self.calculate_accuracy(logits, labels)
                    total_correct += corr
                    total_masked += tot

        if dist.is_available() and dist.is_initialized():
            totals = torch.tensor([total_nll_sum, total_correct, total_masked], device=self.device, dtype=torch.float64)
            dist.all_reduce(totals)
            total_nll_sum, total_correct, total_masked = totals.tolist()

        if total_masked == 0:
            raise ValueError("Evaluation produced no masked target tokens")
        # Calculate Global Average
        avg_loss = total_nll_sum / total_masked if total_masked > 0 else 0
        accuracy = total_correct / total_masked if total_masked > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        print(
            f" -> [{mode}] Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | Acc: {accuracy:.2%}"
        )

        return avg_loss, accuracy, perplexity

    def test(self, test_loader, criterion, log_dir):
        """
        Loads the best model checkpoint and evaluates on the hidden Test Set.
        """
        best_path = os.path.join(log_dir, "best_model.pt")

        if not os.path.exists(best_path):
            print("⚠️ No best model found! Testing with current weights instead.")
        else:
            print(f"\nRestoring best model from: {best_path}")
            checkpoint = torch.load(best_path, map_location=self.device)
            # Ensure the key matches your save_checkpoint function
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            raw_model.load_state_dict(checkpoint["model_state_dict"])
            print(f"   ✅ Model loaded from Step {checkpoint.get('global_step', '?')}")

        print("\n--- 🧪 Starting Final Test Evaluation ---")
        loss, acc, ppl = self.evaluate(
            loader=test_loader, criterion=criterion, mode="TEST"
        )

        if wandb.run is not None:
            wandb.summary["test_accuracy"] = acc
            wandb.summary["test_loss"] = loss
            wandb.summary["test_perplexity"] = ppl

        return acc


