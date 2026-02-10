import random
from dataclasses import dataclass, field

import torch

from model import GPT, ModelConfig
from tokenizer import Tokenizer
from trainer import TrainConfig, train


@dataclass
class CopyingConfig:
    alphabet: str = "abcdefghij"
    min_seq_len: int = 3
    max_seq_len: int = 20
    context_len: int = 128
    n_layers: int = 3
    n_heads: int = 4
    embed_dim: int = 128
    dropout: float = 0.0
    train: TrainConfig = field(default_factory=TrainConfig)


def generate_sequence(tok: Tokenizer, cfg: CopyingConfig) -> tuple[list[int], list[int]]:
    length = random.randint(cfg.min_seq_len, cfg.max_seq_len)
    chars = [random.choice(cfg.alphabet) for _ in range(length)]
    encoded = tok.encode("".join(chars))

    # <bos> input <sep> input <eos>
    tokens = [tok.BOS] + encoded + [tok.SEP] + encoded + [tok.EOS]

    # pad to context_len
    pad_len = cfg.context_len - len(tokens)
    tokens = tokens + [tok.PAD] * pad_len

    # input: all but last, target: all but first (standard next-token)
    x = tokens[:-1]
    y = tokens[1:]
    return x, y


def make_get_batch(tok: Tokenizer, cfg: CopyingConfig):
    def get_batch(batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(batch_size):
            x, y = generate_sequence(tok, cfg)
            xs.append(x)
            ys.append(y)
        return (
            torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device),
        )
    return get_batch


def evaluate(tok: Tokenizer, cfg: CopyingConfig, model: GPT, device: str, n_samples: int = 500) -> dict:
    correct = 0
    for _ in range(n_samples):
        length = random.randint(cfg.min_seq_len, cfg.max_seq_len)
        chars = "".join(random.choice(cfg.alphabet) for _ in range(length))
        encoded = tok.encode(chars)

        # prompt: <bos> input <sep>
        prompt = [tok.BOS] + encoded + [tok.SEP]
        prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)

        output = model.generate(prompt_t, max_new_tokens=length + 1, eos_token=tok.EOS)
        generated = output[0, len(prompt):].tolist()

        # strip EOS if present
        if tok.EOS in generated:
            generated = generated[:generated.index(tok.EOS)]

        if generated == encoded:
            correct += 1

    return {"accuracy": correct / n_samples, "n_samples": n_samples}


def interactive(tok: Tokenizer, cfg: CopyingConfig, model: GPT, device: str):
    print("\nInteractive mode — type a string to copy (or 'quit' to exit):")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "quit":
            break
        if not text:
            continue

        valid = all(ch in tok.char_to_id for ch in text)
        if not valid:
            print(f"  (use only chars from: {cfg.alphabet})")
            continue

        encoded = tok.encode(text)
        prompt = [tok.BOS] + encoded + [tok.SEP]
        prompt_t = torch.tensor([prompt], dtype=torch.long, device=device)

        output = model.generate(prompt_t, max_new_tokens=len(encoded) + 1, eos_token=tok.EOS)
        generated = output[0, len(prompt):].tolist()
        if tok.EOS in generated:
            generated = generated[:generated.index(tok.EOS)]

        result = tok.decode(generated)
        match = "ok" if result == text else "MISMATCH"
        print(f"  {result}  [{match}]")


def main():
    cfg = CopyingConfig()
    tok = Tokenizer(cfg.alphabet)

    model_cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        context_len=cfg.context_len,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        embed_dim=cfg.embed_dim,
        dropout=cfg.dropout,
    )
    model = GPT(model_cfg)

    from trainer import get_device
    device = get_device(cfg.train.device)

    run_dir = train(
        model=model,
        get_batch=make_get_batch(tok, cfg),
        eval_fn=lambda m, d: evaluate(tok, cfg, m, d),
        train_cfg=cfg.train,
        experiment_config=cfg,
    )

    interactive(tok, cfg, model, device)


if __name__ == "__main__":
    main()
