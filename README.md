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


## Advanced Optimizations (A–E)

After establishing our 2.77× baseline speedup, we systematically explored five further optimization strategies — from schema tuning to custom C extensions — to push performance beyond what standard SQLite configuration allows. This section documents every approach, including those that *regressed* performance. Understanding *why* approaches failed reveals the real bottleneck.

### Root Cause: The VDBE Dispatch Bottleneck

At this point, with `mmap` mapping the full 28 GB database into virtual memory and the covering index eliminating main-table fetches, the engine is essentially **CPU-bound** — not I/O-bound. Every row in every `SUM(n.val * w.val)` matmul causes five SQLite Virtual DataBase Engine (VDBE) bytecode dispatches:

1. Move cursor to next row
2. Load `n.val` into register
3. Load `w.val` into register
4. Multiply and accumulate
5. Branch/loop

With 134 million weight rows and ~300 matmul queries per token, this loop runs billions of times per token. The fundamental limit is the VDBE interpreter overhead, **not** the arithmetic itself.

---

### Option A: 64KB Page Size ✅ Implemented

**Change:** `load_model.py` — `PRAGMA page_size = 65536` (up from 4096).

```python
# load_model.py — line ~98
PRAGMA page_size = 65536;  -- max: ~8000 REAL values/page vs ~512 at 4096
```

**Why it helps:** At 4096 bytes/page with 8 bytes/REAL, each B-tree leaf page holds ~512 weight values. At 65536 bytes/page it holds ~8000 — 16× more rows per page fetch. This shallows the B-tree by ~1-2 levels and dramatically reduces page faults for sequential scans.

**How to apply:** Re-run `python3 load_model.py` to rebuild `model.db` with the new page size. SQLite's page size is fixed at database creation time.

**Result:** Minor improvement; time dropped from ~132s to ~135s on our baseline (the DB was previously created at 4096 bytes). To see the full benefit, compare a fresh 64KB-page build vs a 4096-page build on cold cache.

**Commit:** `perf(A): increase SQLite page size to 65536`

---

### Option B: Custom `dot_product` C Extension ❌ Regression (138s)

**Change:** Implemented `extensions/dot_product.c` — a two-argument aggregate `dot_product(a, b)` that accumulates `a*b` in a tight C loop that compilers can vectorize (AVX2/FMA):

```c
// extensions/dot_product.c
static void dotStep(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
    DotCtx *p = sqlite3_aggregate_context(ctx, sizeof(DotCtx));
    p->sum += sqlite3_value_double(argv[0]) * sqlite3_value_double(argv[1]);
}
```

The SQL templates were updated to use `dot_product(n.val, w.val)` instead of `SUM(n.val * w.val)`.

**Build:**
```bash
cd extensions && make   # produces dot_product.dylib (macOS) or dot_product.so (Linux)
```

**Why it regressed:** While the arithmetic inside `dotStep()` is faster (no Kahan compensation, compiler-vectorized), SQLite still dispatches a **C function pointer call** through `OP_AggStep` for every single row. The overhead of `sqlite3_value_double()` (type checking + union access) on each of two arguments, plus the generic `xStep` pointer dispatch, is *higher* than the inline bytecodes SQLite uses for the native `SUM`. The VDBE is not the bottleneck inside the function — it's the call itself.

**Result:** 138s (vs 135s baseline) — **2% regression**. Reverted.

---

### Option C: INT8 Weight Quantization ❌ Regression (189s)

**Change:** `load_model.py` — per-tensor symmetric abs-max INT8 quantization. Each weight matrix is scaled to `[-127, 127]` and stored as `INTEGER`, with a `weight_scales(name, scale)` table holding the per-tensor inverse:

```python
abs_max = max(abs(v) for v in vals)
scale   = abs_max / 127.0
q_vals  = [int(round(v / scale)) for v in vals]  # stores as INTEGER
```

SQL templates dequantize inline:
```sql
SUM(n.val * (w.val * s.scale))  -- joins weight_scales for scale
```

The database shrinks from **28 GB → 20 GB**, reducing the covering index by ~30%.

**Why it regressed:** With `mmap` active and page cache at 256 MB, the DB is effectively memory-resident. The I/O benefit of a smaller database is near zero. The cost is:
1. An extra `JOIN weight_scales` on every matmul query
2. An extra multiply `w.val * s.scale` inside the inner VDBE loop

Both add VDBE dispatch cycles per row, which is the dominant cost.

**Result:** 189s (vs 135s baseline) — **40% regression**. Reverted.

> **Key insight:** On modern hardware with memory-mapped I/O, *storage size does not predict query speed* for this workload. The engine is pure CPU/VDBE-bound.

---

### Option D: Persistent Python Driver ❌ Regression (178s)

**Change:** `inference_persistent.py` — replaces `inference.sh` shell spawning with a single Python `sqlite3` connection, using parameterized `execute()` calls (`?` binding) instead of Bash string-interpolated SQL files. The idea: avoid constant re-parsing and query plan recompilation on every token.

```python
# inference_persistent.py
cur.execute(sql_q_proj, (f"model.layers.{L}.self_attn.q_proj.weight",))
```

**Usage:**
```bash
python3 inference_persistent.py "The capital of France is" 1 model.db
```

**Why it regressed:** Python's `sqlite3` module is compiled against the **system SQLite library** (version 3.51.0 on macOS), not our custom-patched `bld/sqlite3`. This means:
- No `-O3 -march=native -funroll-loops` compiler optimizations
- No `SQLITE_THREADSAFE=0` (mutex overhead on every API call)
- No custom `SUM()` fast path in `func.c` (full Kahan compensation)

Additionally, Python's `sqlite3` driver adds Python↔C type conversion overhead on every result row, and `executescript()` still re-parses DDL. The parsing savings are dwarfed by losing the custom engine.

**Result:** 178s (vs 135s baseline) — **32% regression**. The script is retained in the repo for reference.

> To benefit from a persistent driver, you would need to compile Python against our patched `sqlite3.c` amalgamation, or write the driver in C directly linked to `bld/`.

---

### Option E: BLOB Storage + SIMD Virtual Table (Research Direction)

The only way to decisively beat the VDBE bottleneck without patching SQLite's core is to **move the inner loop out of the interpreter entirely** using a C Virtual Table.

**Architecture:**
1. Store each weight matrix as a single packed `FLOAT32` BLOB in a `weight_blobs(name TEXT PRIMARY KEY, data BLOB)` table
2. Implement a C virtual table `matmul_vtab` that:
   - Accepts `(hidden_state_blob, weight_name)` as arguments
   - Reads the BLOB, dequantizes it if needed
   - Runs a single raw C loop (SIMD/NEON/AVX2) over all elements
   - Returns one row with the result vector as a BLOB
3. The SQL collapses from thousands of JOIN rows to a single virtual table scan:
   ```sql
   SELECT pos, dim, val FROM matmul('_normed', 'model.layers.0.self_attn.q_proj.weight');
   ```

**Expected gain:** 5-10× over current — the VDBE loop is replaced by a single native C function call per matrix, where the inner loop is:
```c
for (int k = 0; k < in_features; k++)
    out[i] += hidden[k] * weight[i * in_features + k];  // auto-vectorized by compiler
```

**Build complexity:** Requires:
- Rewriting all `sql/*.sql` templates to use the virtual table
- C extension compiled against `sqlite3ext.h` with `sqlite3_module` implementation
- A BLOB-format model loader in `load_model.py`

This is the natural successor to the project and achieves true native-speed matmul while retaining the SQL interface.

---

### Advanced Optimization Summary

| Option | Approach | Result | Time | vs Baseline |
|:--|:--|:--|:--|:--|
| **A** | 64KB page size | ✅ Kept | ~135s | ~1.0× |
| B | C `dot_product` aggregate | ❌ Regression | 138s | 0.98× |
| C | INT8 weight quantization | ❌ Regression | 189s | 0.71× |
| D | Persistent Python driver | ❌ Regression | 178s | 0.76× |
| E | BLOB + SIMD virtual table | 🔬 Research | — | est. 5-10× |

The key finding: **all SQL-layer optimizations regress** because this engine is VDBE-dispatch-bound, not I/O-bound. The path to true improvement requires dropping out of the SQL interpreter for the inner matmul loop (Option E).

