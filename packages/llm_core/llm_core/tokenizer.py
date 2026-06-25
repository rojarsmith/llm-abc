from __future__ import annotations


class ByteTokenizer:
    """A dependency-free tokenizer for the first learning loop.

    It maps UTF-8 bytes to token ids 0..255 and reserves 256 as EOS.
    This is intentionally simple so the first model/API loop can run
    before introducing tiktoken and GPT-2 compatibility.
    """

    eos_id = 256
    vocab_size = 257

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        token_ids = list(text.encode("utf-8"))
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        byte_values = bytes(token_id for token_id in token_ids if 0 <= token_id <= 255)
        return byte_values.decode("utf-8", errors="backslashreplace")
