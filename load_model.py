#!/usr/bin/env python3
"""
Phase 1 – Load SmolLM-135M into SQLite.

Schema (no JSON):
  weights(name TEXT, i0 INT, i1 INT, val REAL)
    i0  = row index (or element index for 1-D tensors)
    i1  = column index; 0 for 1-D tensors
  model_config(key TEXT, value TEXT)
  vocab(token_id INT PRIMARY KEY, token_str TEXT)
"""

import os, sys, json, struct, sqlite3, time, urllib.request

REPO      = "HuggingFaceTB/SmolLM-135M"
FILES     = ["model.safetensors", "config.json", "tokenizer.json"]
BASE_URL  = "https://huggingface.co/{repo}/resolve/main/{file}"
MODEL_DIR = "model"
DB_PATH   = "model.db"
BATCH     = 50_000          # rows per executemany batch

# ── helpers ──────────────────────────────────────────────────────────────────

def progress(count, block, total):
    if total > 0:
        pct = min(100, count * block * 100 // total)
        mb  = count * block / 1e6
        sys.stdout.write(f"\r  {pct:3d}%  {mb:7.1f} MB")
        sys.stdout.flush()

def download(url, dest):
    if os.path.exists(dest):
        print(f"  [cached]  {dest}")
        return
    print(f"  GET {url}")
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()

def parse_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode())
    return hdr, 8 + n

def read_tensor(f, dtype, offsets, data_start):
    begin, end = offsets
    f.seek(data_start + begin)
    raw = f.read(end - begin)
    if dtype == "BF16":
        vals = []
        for i in range(0, len(raw), 2):
            vals.append(struct.unpack("<f", b"\x00\x00" + raw[i:i+2])[0])
        return vals
    elif dtype == "F32":
        n = len(raw) // 4
        return list(struct.unpack(f"<{n}f", raw))
    elif dtype == "F16":
        n = len(raw) // 2
        return [float(v) for v in struct.unpack(f"<{n}e", raw)]
    raise ValueError(f"Unsupported dtype: {dtype}")

def rows_from(name, shape, vals):
    if len(shape) == 1:
        return [(name, i, 0, v) for i, v in enumerate(vals)]
    elif len(shape) == 2:
        cols = shape[1]
        return [(name, k // cols, k % cols, v) for k, v in enumerate(vals)]
    raise ValueError(f"Unexpected shape {shape} for '{name}'")

def insert_batch(cur, rows):
    for start in range(0, len(rows), BATCH):
        cur.executemany(
            "INSERT INTO weights(name,i0,i1,val) VALUES(?,?,?,?)",
            rows[start:start+BATCH]
        )

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=== Downloading model files ===")
    for f in FILES:
        download(BASE_URL.format(repo=REPO, file=f),
                 os.path.join(MODEL_DIR, f))

    with open(os.path.join(MODEL_DIR, "config.json")) as fh:
        config = json.load(fh)

    print(f"\n=== Creating database: {DB_PATH} ===")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    cur.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous  = NORMAL;
        PRAGMA page_size    = 65536;  -- max page size: ~8000 REAL values/page vs ~1000 at 8192
        PRAGMA cache_size   = -2000000;

        CREATE TABLE weights (
            name TEXT    NOT NULL,
            i0   INTEGER NOT NULL,
            i1   INTEGER NOT NULL,
            val  REAL    NOT NULL
        );

        CREATE TABLE model_config (
            key   TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE vocab (
            token_id  INTEGER PRIMARY KEY,
            token_str TEXT
        );
    """)

    cur.executemany(
        "INSERT INTO model_config(key,value) VALUES(?,?)",
        [(k, json.dumps(v)) for k, v in config.items()]
    )
    conn.commit()

    print("\n=== Loading weights ===")
    st_path = os.path.join(MODEL_DIR, "model.safetensors")
    t_total = time.time()

    with open(st_path, "rb") as f:
        header, data_start = parse_header(f)
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        N = len(tensors)
        print(f"  Tensors: {N}")

        for idx, (name, info) in enumerate(tensors.items(), 1):
            t0    = time.time()
            vals  = read_tensor(f, info["dtype"], info["data_offsets"], data_start)
            shape = info["shape"]
            rows  = rows_from(name, shape, vals)
            insert_batch(cur, rows)
            conn.commit()
            elapsed = time.time() - t0
            print(f"  [{idx:3d}/{N}] {name:60s} shape={str(shape):20s}  "
                  f"n={len(vals):9,d}  {elapsed:5.1f}s")

    print(f"\n  Total weight load: {time.time()-t_total:.0f}s")

    print("\n=== Building index ===")
    t0 = time.time()
    cur.execute("CREATE INDEX idx_w ON weights(name, i0, i1)")
    # Secondary index for join on i1 (used in matmul)
    cur.execute("CREATE INDEX idx_w2 ON weights(name, i1, i0)")
    conn.commit()
    print(f"  Done in {time.time()-t0:.0f}s")

    print("\n=== Loading vocabulary ===")
    with open(os.path.join(MODEL_DIR, "tokenizer.json")) as fh:
        tok = json.load(fh)
    vocab = tok["model"]["vocab"]
    cur.executemany(
        "INSERT OR REPLACE INTO vocab(token_id, token_str) VALUES(?,?)",
        [(v, k) for k, v in vocab.items()]
    )
    conn.commit()
    print(f"  Vocabulary size: {len(vocab):,}")

    conn.close()

    db_mb = os.path.getsize(DB_PATH) / 1e6
    print(f"\n✓ model.db ready  ({db_mb:.0f} MB)")

if __name__ == "__main__":
    main()
