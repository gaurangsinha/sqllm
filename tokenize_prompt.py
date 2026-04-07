#!/usr/bin/env python3
"""
Thin tokenizer wrapper — the ONLY Python invoked during inference.

Usage:
  python3 tokenize_prompt.py encode "The capital of France is "
  → space-separated token IDs on stdout

  python3 tokenize_prompt.py decode 12345
  → decoded string on stdout
"""

import sys, os, json

TOKENIZER_FILE = os.path.join(os.path.dirname(__file__), "model", "tokenizer.json")

def load_tokenizer():
    with open(TOKENIZER_FILE) as f:
        return json.load(f)

# ── BPE encode (minimal, matches HF fast tokenizer for simple prompts) ────────

def get_pairs(word):
    return {(word[i], word[i+1]) for i in range(len(word)-1)}

def bpe_encode(word_chars, merges_index):
    """Apply BPE merges to a list of characters."""
    word = list(word_chars)
    while len(word) > 1:
        pairs = get_pairs(word)
        # find the highest-priority merge
        best = None
        best_rank = float("inf")
        for pair in pairs:
            rank = merges_index.get(pair, float("inf"))
            if rank < best_rank:
                best_rank = rank
                best = pair
        if best is None or best_rank == float("inf"):
            break
        a, b = best
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word)-1 and word[i] == a and word[i+1] == b:
                new_word.append(a + b)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = new_word
    return word

def encode(text, tokenizer):
    vocab       = tokenizer["model"]["vocab"]
    merges_list = tokenizer["model"]["merges"]
    merges_idx  = {tuple(m.split(" ")): i for i, m in enumerate(merges_list)}

    # Pre-tokenizer: GPT-2 style — prepend Ġ to every word except first
    import re
    # Simple whitespace + punctuation split matching llama's pre-tokenizer
    PAT = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+"""
    )
    tokens = []
    for i, m in enumerate(PAT.finditer(text)):
        word = m.group()
        # GPT-2 uses Ġ to represent leading space
        if word.startswith(" "):
            word = "\u0120" + word[1:]
        # byte-level: map each char to vocab
        chars = list(word)
        merged = bpe_encode(chars, merges_idx)
        for piece in merged:
            if piece in vocab:
                tokens.append(vocab[piece])
            else:
                # fall back: split to individual chars
                for c in piece:
                    if c in vocab:
                        tokens.append(vocab[c])
    return tokens

def decode_token(token_id, tokenizer):
    # Build reverse vocab
    vocab = tokenizer["model"]["vocab"]
    rev   = {v: k for k, v in vocab.items()}
    piece = rev.get(token_id, "")
    # Ġ → space, Ċ → newline
    piece = piece.replace("\u0120", " ").replace("\u010a", "\n")
    return piece

# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: tokenize_prompt.py encode <text>", file=sys.stderr)
        print("       tokenize_prompt.py decode <token_id>", file=sys.stderr)
        sys.exit(1)

    tok = load_tokenizer()
    cmd = sys.argv[1]

    if cmd == "encode":
        text = sys.argv[2]
        ids  = encode(text, tok)
        print(" ".join(str(i) for i in ids))

    elif cmd == "decode":
        token_id = int(sys.argv[2])
        print(decode_token(token_id, tok), end="")
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)
