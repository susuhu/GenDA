"""Checkpoint-compatible genomic vocabulary and encoding."""
import itertools
import re
import numpy as np

class GenomicTokenizer:
    def __init__(self, kmer_size=1):
        self.kmer_size = kmer_size
        
        # 1. Define Bases
        bases = ["A", "C", "G", "T", "N"]
        
        # 2. Define Specials
        self.specials = ["<BOS>", "<EOS>", "<PAD>", "<MASK>", "<UNK>", "<CLS>", "<SEP>"]
        
        # 3. Define Context Tokens
        # FROZEN — do not insert or reorder. All IDs below are fixed for existing checkpoints.
        # IDs 7–22 (1-mer) / 7–22 (k-mer)
        self.context_tokens = [
            "[Gene_Start]", "[HUMAN]", "[MOUSE]",
            "[PROMOTER]", "[ENHANCER]", "[EXON]", "[INTRON]", "[SPLICE]", "[TFBS]", "[BACKGROUND]",
            "[Expression:High]", "[Expression:Low]",
            "[Cell:Liver]", "[Cell:Neuron]", "[Cell:Immune]", "[Cell:Muscle]",
        ]

        # 4. Extra tokens — appended AFTER bases so all existing IDs are unchanged.
        # Old checkpoints are unaffected: their embedding matrix ends at the old vocab size,
        # and inference never feeds these new IDs to an old model.
        # New training extends the embedding matrix to cover these IDs.
        self.extra_tokens = [
            "[REPEAT]",  # Transposable elements / repetitive regions (RepeatMasker)
            "[NULL]",    # CFG unconditional token — CFG dropout only, never a real sequence label
        ]

        # --- BRANCH: 1-MER vs K-MER ---
        if self.kmer_size == 1:
            self.vocab = self.specials + self.context_tokens + bases + self.extra_tokens
        else:
            real_bases = ["A", "C", "G", "T"]
            kmers = ["".join(p) for p in itertools.product(real_bases, repeat=kmer_size)]
            self.vocab = self.specials + self.context_tokens + kmers + self.extra_tokens

        # Mappings
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}

        # Essential IDs
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.mask_id = self.token_to_id["<MASK>"]
        self.pad_id = self.token_to_id["<PAD>"]
        self.unk_id = self.token_to_id["<UNK>"]

        # 🚀 OPTIMIZATION: Build Vectorized Mapper ONLY for 1-mer
        if self.kmer_size == 1:
            self.mapper = np.full(256, self.unk_id, dtype=np.int64)
            for b in bases:
                self.mapper[ord(b)] = self.token_to_id[b]
                self.mapper[ord(b.lower())] = self.token_to_id[b]

    def encode(self, sequence_str):
        """
        Universal encode function that picks the best strategy.
        """
        sequence_str = sequence_str.upper()
        
        # Regex to split out special tokens
        parts = re.split(r'(\[.*?\]|<.*?>)', sequence_str)
        
        ids = []
        for part in parts:
            if not part:
                continue
                
            if part.startswith('[') or part.startswith('<'):
                # Handle special tokens natively as a single chunk
                token_id = self.token_to_id.get(part, self.unk_id)
                ids.append(token_id)
            else:
                # Handle genomic sequence
                if self.kmer_size == 1:
                    arr = np.frombuffer(part.encode('latin1'), dtype=np.uint8)
                    ids.extend(self.mapper[arr].tolist())
                else:
                    tokens = [part[i : i + self.kmer_size] for i in range(0, len(part), self.kmer_size)]
                    if len(tokens) > 0 and len(tokens[-1]) < self.kmer_size:
                        tokens[-1] = "<UNK>"
                    ids.extend([self.token_to_id.get(t, self.unk_id) for t in tokens])
                    
        return np.array(ids, dtype=np.int64)

    def decode(self, ids, skip_specials=True):
        """
        Converts a list of token IDs back into a genomic string.
        """
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, "")
            if skip_specials and (token.startswith("<") or token.startswith("[")):
                continue
            tokens.append(token)
            
        return "".join(tokens)

    @property
    def all_special_ids(self):
        ids = set()
        for t in self.specials + self.context_tokens + self.extra_tokens:
            if t in self.token_to_id:
                ids.add(self.token_to_id[t])
        return list(ids)


