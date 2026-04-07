-- 00_schema.sql  (run once per inference session to create working tables)
-- All TEMP tables live in-memory for the duration of the sqlite3 process.

PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 30000000000;
-- We wrap the entire block of queries in a single overarching transaction to minimize journaling overhead
BEGIN TRANSACTION;

-- Input token sequence for this forward pass
CREATE TEMP TABLE IF NOT EXISTS _seq (
    pos      INTEGER PRIMARY KEY,   -- token position (0-based)
    token_id INTEGER NOT NULL
);

-- Current hidden state  [seq_len × hidden_size]
CREATE TEMP TABLE IF NOT EXISTS _hidden (
    pos INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    val REAL    NOT NULL,
    PRIMARY KEY (pos, dim)
) WITHOUT ROWID;

-- Normed hidden state (output of RMSNorm, reused each layer)
CREATE TEMP TABLE IF NOT EXISTS _normed (
    pos INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    val REAL    NOT NULL,
    PRIMARY KEY (pos, dim)
) WITHOUT ROWID;

-- Q/K/V projections  (full flat dimension, not split by head)
CREATE TEMP TABLE IF NOT EXISTS _q (
    pos INTEGER NOT NULL,
    dim INTEGER NOT NULL,   -- 0..num_q_heads*head_dim-1
    val REAL    NOT NULL,
    PRIMARY KEY (pos, dim)
) WITHOUT ROWID;
CREATE TEMP TABLE IF NOT EXISTS _k (
    pos INTEGER NOT NULL,
    dim INTEGER NOT NULL,   -- 0..num_kv_heads*head_dim-1
    val REAL    NOT NULL,
    PRIMARY KEY (pos, dim)
) WITHOUT ROWID;
CREATE TEMP TABLE IF NOT EXISTS _v (
    pos INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    val REAL    NOT NULL,
    PRIMARY KEY (pos, dim)
) WITHOUT ROWID;

-- Attention scores & weights before/after softmax
CREATE TEMP TABLE IF NOT EXISTS _attn_scores (
    q_pos  INTEGER NOT NULL,
    k_pos  INTEGER NOT NULL,
    q_head INTEGER NOT NULL,
    score  REAL    NOT NULL,
    PRIMARY KEY (q_pos, k_pos, q_head)
) WITHOUT ROWID;
CREATE TEMP TABLE IF NOT EXISTS _attn_weights (
    q_pos  INTEGER NOT NULL,
    k_pos  INTEGER NOT NULL,
    q_head INTEGER NOT NULL,
    weight REAL    NOT NULL,
    PRIMARY KEY (q_pos, k_pos, q_head)
) WITHOUT ROWID;

-- Attention context (weighted sum of V)   [seq_len × hidden_size]
CREATE TEMP TABLE IF NOT EXISTS _attn_ctx (
    pos INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    val REAL    NOT NULL,
    PRIMARY KEY (pos, dim)
) WITHOUT ROWID;

-- FFN intermediate projections
CREATE TEMP TABLE IF NOT EXISTS _gate (
    pos INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    val REAL    NOT NULL,
    PRIMARY KEY (pos, dim)
) WITHOUT ROWID;
CREATE TEMP TABLE IF NOT EXISTS _up (
    pos INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    val REAL    NOT NULL,
    PRIMARY KEY (pos, dim)
) WITHOUT ROWID;
CREATE TEMP TABLE IF NOT EXISTS _ffn_mid (
    pos INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    val REAL    NOT NULL,
    PRIMARY KEY (pos, dim)
) WITHOUT ROWID;

-- Final logits over vocabulary
CREATE TEMP TABLE IF NOT EXISTS _logits (
    token_id INTEGER PRIMARY KEY,
    score    REAL    NOT NULL
);
