class Tokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3

    SPECIAL_TOKENS = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<sep>"}

    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.char_to_id = {ch: i + 4 for i, ch in enumerate(alphabet)}
        self.id_to_char = {i + 4: ch for i, ch in enumerate(alphabet)}
        self.vocab_size = len(alphabet) + 4

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids if i in self.id_to_char)

    def decode_raw(self, ids: list[int]) -> str:
        parts = []
        for i in ids:
            if i in self.SPECIAL_TOKENS:
                parts.append(self.SPECIAL_TOKENS[i])
            elif i in self.id_to_char:
                parts.append(self.id_to_char[i])
            else:
                parts.append(f"<unk:{i}>")
        return " ".join(parts)
