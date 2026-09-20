import tempfile
import unittest
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from genda.config import load_config
from genda.checkpoints import save_checkpoint, load_checkpoint
from genda.data.tokenizer import GenomicTokenizer
from genda.data.masking import DiffusionCollateFn
from genda.model.factory import build_model
from genda.inference.api import Predictor
from genda.training.trainer import GenomicTrainer


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.config = load_config()
        self.config.update(hidden_dim=32, layers=1, heads=4, max_token_len=32,
                           dropout=0.0, attn_implementation="sdpa", span_lambda=3,
                           adaptive_sliding_window=3, cond_dropout_rate=0.0)
        self.tokenizer = GenomicTokenizer()

    def model(self):
        return build_model(self.config, self.tokenizer)

    def test_vocab_and_masking(self):
        self.assertEqual(self.tokenizer.token_to_id["A"], 23)
        self.assertEqual(self.tokenizer.token_to_id["[HUMAN]"], 8)
        ids = torch.tensor(self.tokenizer.encode("<BOS>[HUMAN][EXON]ACGTACGTACGT<EOS>"))
        batch = DiffusionCollateFn(self.tokenizer, self.config)([{"input_ids": ids}])
        self.assertTrue((batch["labels"] != -100).any())
        self.assertTrue((batch["labels"][0, :3] == -100).all())
        self.assertEqual(self.tokenizer.decode(ids.tolist()), "ACGTACGTACGT")

    def test_training_resume_and_inference(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 4)
        ids = torch.tensor(self.tokenizer.encode("<BOS>[HUMAN][EXON]ACGTACGTACGT<EOS>"))
        loader = DataLoader([{"input_ids": ids}] * 4, batch_size=2,
                            collate_fn=DiffusionCollateFn(self.tokenizer, self.config))
        before = next(model.parameters()).detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "run_config.yaml").write_text(yaml.safe_dump(self.config))
            trainer = GenomicTrainer(model, self.tokenizer, torch.device("cpu"), use_amp=False)
            trainer.train(loader, loader, optimizer, scheduler, None, total_steps=1,
                          log_dir=str(folder), val_interval=1)
            self.assertFalse(torch.equal(before, next(model.parameters())))
            restored = self.model()
            opt2 = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            sched2 = get_cosine_schedule_with_warmup(opt2, 0, 4)
            step, best = load_checkpoint(folder / "last_model.pt", restored, opt2, sched2)
            self.assertEqual(step, 1)
            self.assertEqual(scheduler.state_dict(), sched2.state_dict())
            for first, second in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(first, second)
            trainer2 = GenomicTrainer(restored, self.tokenizer, torch.device("cpu"), use_amp=False)
            trainer2.best_val_loss = best
            trainer2.train(loader, loader, opt2, sched2, None, total_steps=2,
                           log_dir=str(folder), val_interval=1, start_step=step)
            predictor = Predictor(folder / "last_model.pt", device="cpu")
            sequences = predictor.generate(["[HUMAN][EXON]"] * 2, length=8, steps=2, cfg_scale=2.0)
            self.assertEqual(len(sequences), 2)
            for sequence in sequences:
                self.assertEqual(len(sequence), 8)
                self.assertLessEqual(set(sequence), set("ACGT"))
            with self.assertRaises(ValueError):
                predictor.generate("[HUMAN][EXON]", length=32)
            with self.assertRaises(ValueError):
                predictor.generate("[EXON]", length=8)
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(folder / "missing.pt", restored)
            broken = torch.load(folder / "last_model.pt", weights_only=True)
            broken["model_state_dict"].pop(next(iter(broken["model_state_dict"])))
            torch.save(broken, folder / "broken.pt")
            with self.assertRaises(RuntimeError):
                load_checkpoint(folder / "broken.pt", restored)

    def test_cli_with_synthetic_genome(self):
        from genda.training.cli import main as train
        from genda.inference.cli import main as predict
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fasta = root / "genome.fa"
            fasta.write_text(">chr1\n" + "ACGT" * 250 + "\n")
            gtf = root / "genes.gtf"
            gtf.write_text("# Synthetic fixture: explicit precomputed splits below.\n")
            cache = root / "cache"
            cache.mkdir()
            rows = "chrom,start,end,type,gene_id\n" + "".join(
                f"chr1,{100 + i * 20},{110 + i * 20},exon,gene{i}\n" for i in range(4))
            for name in ("train_set.csv", "val_set_fixed.csv", "test_set_fixed.csv"):
                (cache / name).write_text(rows)
            config = dict(self.config, fasta_path=str(fasta), gtf_path=str(gtf), seq_len=24,
                          batch_size=2, accumulation_steps=1, total_steps=1, warmup_steps=0,
                          val_interval=1, functional_ratio=1.0, jitter_bp=0)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            train(["--config", str(config_path), "--output-dir", str(root / "runs"),
                   "--cache-dir", str(cache), "--workers", "0", "--attention", "sdpa"])
            run = next((root / "runs").iterdir())
            self.assertTrue((run / "last_model.pt").is_file())
            predict(["--checkpoint", str(run / "last_model.pt"), "--output-dir", str(root / "generated"),
                     "--device", "cpu", "--length", "8", "--steps", "2", "--samples", "2"])
            fasta_output = (root / "generated" / "sequences.fasta").read_text().splitlines()
            self.assertEqual(len(fasta_output), 4)
            self.assertEqual(len(fasta_output[1]), 8)


if __name__ == "__main__":
    unittest.main()
