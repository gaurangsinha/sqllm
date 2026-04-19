# SQL-LLM: A Pure SQLite Inference Engine

This project demonstrates a working Large Language Model (LLM) inference engine where the entire forward pass — including embedding lookup, RMSNorm, Rotary Positional Embedding (RoPE), Multi-Head Attention (Grouped Query Attention), SwiGLU Feed-Forward Network, and logits sampling — is performed entirely within SQLite using standard SQL queries! 

We use the [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) model, which runs on a standard Llama-like architecture.

## How it Works

The project is split into two phases:
1. **Model Loading (Python):** We download the model weights securely in `safetensors` format from Hugging Face and convert them into a flat relational schema (`weights(name, i0, i1, val)`) housed within a SQLite database. This avoids needing `PyTorch` or `transformers` for generation!
2. **Inference (SQL & Bash):** Once the database is populated, inference runs via a pure Bash script (`inference.sh`) driving `sqlite3`. The bash orchestrator loops through the model's 30 transformer layers, maps weight paths into SQL templates (`sql/*.sql`), and pushes SQLite to compute massive matrix multiplications natively via `JOIN` and `SUM()` aggregation.

### Why?
To prove that the fundamental abstract operations powering Generative AI today (matrix multiplication and non-linearities) can be natively expressed as standard classical relational algebra within a 25-year-old embedded database!

---

## Setup and Usage

### Prerequisites
- **Python 3.8+** (for downloading the model and mapping it into the DB securely).
- **`sqlite3`** (Requires macOS SQLite or a modern SQLite binary compiled with the Math Functions extension enabled for operations like `EXP()`, `LOG()`, `COS()`, `SIN()`, `SQRT()`).

### 1. Load the Model into SQLite
This step downloads the tokenizer, config, and translation matrices into massive structured relational indexes.
```bash
python3 load_model.py
```
> [!IMPORTANT]
> The generated `model.db` will take up around **14 GB** of disk space! This is because creating searchable SQL relations tracking distinct dimensional indices natively scales row limits.

### 2. Run Inference
The `inference.sh` script handles prompting and feeding intermediate state tables back through the 30 localized SQL files per sequence layer.
```bash
./inference.sh "The capital of France is" 1 model.db
```
*(Parameters: 1st is the text prompt, 2nd is max new tokens, 3rd is target database).*

---

## Engine Performance & Optimizations

**⚠️ SLOW INFERENCE WARNING:**
Performing full transformer passes on ~135 million parameters via nested relational row joins is computationally heavy. A stock SQLite build takes ~**6 minutes per token**.

The optimizations below are layered across three levels: **SQL schema design**, **compiled-in SQLite source patches**, and **database indexes**. Each can be applied independently; they stack.

---

### Level 1 — SQL Schema Optimizations (enabled by default)

These are already active in `00_schema.sql` and `inference.sh`:

1. **`WITHOUT ROWID` Temp Tables:** Intermediate tensors (`_hidden`, `_normed`, `_q`, `_k`, `_v`, etc.) use `WITHOUT ROWID` so data is clustered by primary key in memory, eliminating the secondary B-Tree lookup.
2. **In-Memory Temp Storage:** `PRAGMA temp_store = MEMORY` keeps all temp table I/O in RAM instead of spilling to `/tmp`.
3. **Single Transaction:** The entire forward pass runs inside one `BEGIN TRANSACTION` / `COMMIT`, avoiding per-statement journal writes.
4. **mmap:** `PRAGMA mmap_size = 30000000000` maps the 21 GB weights database directly into virtual memory, converting `read()` syscalls into page faults served by the OS cache.

---

### Level 2 — Covering Index on `weights`

```sql
CREATE INDEX idx_w2_cover ON weights(name, i1, i0, val);
```

This is the **single biggest lever**. Without it, every matmul query hits the weight index to find matching rows, then does a second lookup into the 21 GB main table to read `val`. With `val` embedded in the index leaf, that second lookup never happens — all weight data is served directly from the index.

> [!IMPORTANT]
> This index roughly doubles the database size from ~21 GB to ~28 GB. It is a storage-for-speed tradeoff.

---

### Level 3 — Optimized SQLite Binary (`bld/sqlite3`)

The SQLite source in `sqlite-src-3530000/` is patched and rebuilt with:

**Compiler flags** (`bld/Makefile`):
```diff
-CFLAGS = -O2 -g
+CFLAGS = -O3 -march=native -funroll-loops
```
`-O3 -march=native` enables AVX2/FMA auto-vectorization of the inner float accumulation loop. `-funroll-loops` unrolls the B-tree cursor and VDBE dispatch loops.

**`SQLITE_THREADSAFE=0`:** Removes mutex lock/unlock pairs from every VDBE API call. Safe here — inference is single-process.

**SUM() fast path** (`src/func.c`): SQLite's `SUM()` uses the Kahan-Babushka-Neumaier algorithm to guard against floating-point rounding error — three extra FP ops and a branch per row. For transformer `REAL` weights where argmax just needs to be *approximately* correct, this is unnecessary. A fast path bypasses it:
```c
/* Skip KBN compensation for REAL-only workloads */
p->rSum += sqlite3_value_double(argv[0]);
```

**Larger default page cache** (`src/sqliteLimit.h`): `SQLITE_DEFAULT_CACHE_SIZE` raised from 2 MB to 256 MB, keeping B-tree interior nodes for the weight indexes hot across all 30 layers.

**Compiled-in mmap** (`src/sqliteInt.h`): `SQLITE_DEFAULT_MMAP_SIZE` raised from 0 to 20 GB so mmap is active before any `PRAGMA` fires.

To build:
```bash
cd bld && make sqlite3
```

Pass the optimized binary as the 4th argument to `inference.sh`:
```bash
./inference.sh "The capital of France is" 1 model.db ../bld/sqlite3
```

---

### Benchmark Results

All runs on prompt `"The capital of France is"` → `" Paris"` (greedy decode, 1 token).

| Configuration | Wall Time | CPU Time | CPU% | vs Baseline |
|---|---|---|---|---|
| Stock `sqlite3`, no covering index | 6m 05s | 309s | 88% | 1.0× |
| Stock `sqlite3` + `idx_w2_cover` | 2m 39s | 151s | 98% | 2.29× |
| Optimized `bld/sqlite3`, no covering index | 4m 07s | 241s | 98% | 1.48× |
| **Optimized `bld/sqlite3` + `idx_w2_cover`** | **2m 12s** | **130s** | **99%** | **2.77×** |

The covering index alone (stock build) cuts time from 6m05s to 2m39s — a **2.3× improvement** purely from eliminating main-table fetches. The source patches add another **1.5×**. Combined: **2.77× faster**, with CPU utilization at 99% and essentially zero I/O wait remaining.

---

## Architecture & Relational Mapping Breakdown

Below is how standard LLM operations were mapped into SQL logic:
- **Tokenisation & Vocab**: Done via a tiny Python script `tokenize_prompt.py` operating purely string algorithms locally mapped against `vocab()`. 
- **Matrix Multiplication (`03_qkv_proj.sql`)**: Expressed strictly as an inner dimensional join: `SUM(a.val * b.val) GROUP BY a.dim, b.dim`.
- **Rotary Position Embeddings (`04_rope.sql`)**: Computes complex sine/cosine rotational bounds splitting dimensions into symmetric vector halves via adjacent `JOIN _rope_freqs_step1 f`.
- **Softmax (`05_attention.sql`)**: Leverages aggregate functions recursively finding sequence `MAX()`, mapping scale offsets via `EXP()`, and fractionally distributing attention weights off dynamic `SUM()`.
- **Residual Paths (`06_proj_and_add.sql`)**: Joins computed tensor results natively atop older states mapping simple arithmetic `SELECT (h.val + p.val)`.

## License
MIT License. Use at your own risk!


## Advanced Optimizations

Beyond the base engine configuration, we pursued several advanced optimizations to eliminate remaining bottlenecks (I/O, B-tree traversal overhead, and VDBE interpreter dispatch). 

### Option A: 64KB Page Size (Implemented)
The default SQLite page size is 4KB or 8KB. By building the database with  (the maximum allowed), we pack ~8,000 weight values sequentially per B-tree page instead of ~1,000. This flattens the B-tree depth and significantly reduces the number of page loads during the sequential sequential matrix multiplication scans.
*You must rebuild the database using `load_model.py` to apply this change.*

### Option B: Custom C Extension (Tested, Reverted)
We wrote a custom SQLite loadable C extension (`dot_product.dylib`) to replace `SUM(n.val * w.val)` with a tight C accumulation loop vectorized by compiler AVX2/FMA intrinsics.
**Result:** Performance *regressed* from 135s to 138s per token. In SQLite, the overhead of the Virtual DataBase Engine (VDBE) dispatching a User-Defined Function (UDF) pointer on every row is higher than the inline bytecode execution of the built-in `SUM`, even when the built-in function is doing Kahan compensation. We reverted the change.
