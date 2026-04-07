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
Performing full transformer passes on ~135 Million parameters via nested relational row joins is computationally heavy. Out of the box, SQLite takes ~**5 to 6 minutes per token** running off `model.db`. 

Because SQLite processes matrix dot products via `(n.val * w.val)`, I/O ping-ponging presents a huge architectural throttle point. To maximize raw execution capabilities, we've implemented the following core engine traits:

1. **`WITHOUT ROWID` Tables:** Temporary tracking constructs (e.g. `_hidden`) disable row IDs so underlying data is fully clustered via strictly typed memory blocks instead of doubled B-Trees.
2. **RAM Allocation:** Added `PRAGMA temp_store = MEMORY;` avoiding temporary disk spillage to `/tmp`. 
3. **Mega-Transactions:** Execution boundaries are strictly enforced via a sweeping overarching `BEGIN TRANSACTION;` / `COMMIT;` scope avoiding sequential journaling latency.
4. **Memory Mapped Loading:** `PRAGMA mmap_size = 30000000000;` directly wraps database blocks natively into unified zero-copy paging RAM limit bounds.

### Achieving 2x Speed (The `model_cover.db` Trick)
If 5+ minute token times are too slow, generation time can be sliced mathematically in half (~**2.5 minutes**) by building an aggressive **Covering Index** across the primary dimensional values!
Inside your `model.db`, run:
```sql
CREATE INDEX idx_w2_cover ON weights(name, i1, i0, val);
```
Since the `val` parameter (raw float data) is now natively embedded in the B-Tree search indexes alongside coordinates, SQLite resolves matrix joins dynamically inside its RAM index arrays without ever needing an outer main-table fetch! (However, preparing this index duplicates data scaling your `.db` asset to roughly **28 GB**!).

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
