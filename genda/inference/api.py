"""Checkpoint-backed DNA generation without training infrastructure."""
import torch
from genda.config import checkpoint_config
from genda.checkpoints import load_checkpoint
from genda.data.tokenizer import GenomicTokenizer
from genda.model.factory import build_model
from .generator import GenomicGenerator


class Predictor:
    def __init__(self, checkpoint, config=None, device=None, attention=None):
        self.config = checkpoint_config(checkpoint, config)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        attention = attention or ("sdpa" if self.device.type == "cpu" else self.config["attn_implementation"])
        self.tokenizer = GenomicTokenizer(self.config["kmer_size"])
        self.model = build_model(self.config, self.tokenizer, attention)
        self.step, _ = load_checkpoint(checkpoint, self.model)
        self.model.to(self.device, dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32).eval()
        self.generator = GenomicGenerator(self.model, self.tokenizer, self.device)

    @torch.inference_mode()
    def generate(self, prompts, length=256, steps=32, temperature=1.0, cfg_scale=1.0,
                 negative_prompt="[HUMAN][BACKGROUND]", allow_eos=False,
                 remasking_strategy="single_token", use_bio_check=False):
        if length <= 0 or steps <= 0 or temperature <= 0:
            raise ValueError("length, steps and temperature must be positive")
        if remasking_strategy not in ("single_token", "poisson_span"):
            raise ValueError("Invalid remasking strategy")
        self.generator.remasking_strategy = remasking_strategy
        self.generator.span_lambda = self.config.get("span_lambda", 100)
        batch = [prompts] if isinstance(prompts, str) else list(prompts)
        if not batch:
            raise ValueError("At least one prompt is required")
        def encode(prompt):
            ids = []
            for part in prompt.split():
                if part in self.tokenizer.token_to_id:
                    ids.append(self.tokenizer.token_to_id[part])
                else:
                    ids.extend(self.tokenizer.encode(part).tolist())
            if self.tokenizer.unk_id in ids:
                raise ValueError(f"Unknown token in prompt: {prompt}")
            return ids
        lengths = {len(encode(prompt)) for prompt in batch + [negative_prompt]}
        if len(lengths) != 1:
            raise ValueError("All prompts and the negative prompt must have the same token length")
        if lengths.pop() + length + 2 > self.config["max_token_len"]:
            raise ValueError("Prompt + generated tokens + BOS/EOS exceed max_token_len")
        return self.generator.generate(prompts, gap_length=length, steps=steps,
                temperature=temperature, cfg_scale=cfg_scale, negative_prompt=negative_prompt,
                allow_eos=allow_eos, use_bio_check=use_bio_check)
