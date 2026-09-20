"""One architecture construction path for training and inference."""
from .backbones import GenDA_ModernBERT, GenDA_Llama


def build_model(config, tokenizer, attention=None):
    common = dict(
        tokenizer=tokenizer, hidden_dim=config["hidden_dim"], n_layers=config["layers"],
        dropout_rate=config["dropout"], num_attention_heads=config["heads"],
        max_position_embeddings=config["max_token_len"],
        attn_implementation=attention or config["attn_implementation"],
        position_encoding=config.get("position_encoding", "rope"),
    )
    backbone = config.get("backbone", "modernbert").lower()
    if backbone == "modernbert":
        return GenDA_ModernBERT(**common,
            global_attn_every_n_layers=config.get("global_attn_every_n_layers", 1),
            local_attention=config["max_token_len"])
    if backbone == "llama":
        return GenDA_Llama(**common, intermediate_size=config.get("intermediate_size"),
            num_kv_heads=config.get("num_kv_heads"), rope_theta=config.get("rope_theta", 10000.0))
    raise ValueError(f"Unsupported backbone: {backbone}")
