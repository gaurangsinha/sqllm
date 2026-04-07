-- rope.sql (template)
-- Rotary Positional Embedding (RoPE) applied to _q and _k in-place
-- SmolLM-135M RoPE:
--   head_dim = hidden_size / num_heads = 576 / 9 = 64
--   For each head, apply rotation to adjacent pairs of dims.
--   We'll use a temp table to compute the rotations and then replace the target table.

-- {TGT}     : Target table (_q or _k)
-- {THETA}   : rope_theta (10000.0)
-- {HEADS}   : Number of heads for this tensor (9 for Q, 3 for K)
-- {HEAD_DIM}: Dimension per head (64)
-- {SEQ_LEN} : Length of the sequence (max pos + 1)

-- 1. Precalculate freqs
CREATE TEMP TABLE IF NOT EXISTS _rope_freqs_step1 (
    head     INTEGER,
    pair_idx INTEGER,  -- 0 to HEAD_DIM/2 - 1
    freq     REAL
);
DELETE FROM _rope_freqs_step1;

-- Generate frequencies: 1.0 / (THETA ^ (2 * pair_idx / HEAD_DIM))
-- SQLite doesn't have POWER, use EXP and log trick if needed, or simple math:
-- freq = EXP(-2.0 * pair_idx / HEAD_DIM * LOG(THETA))
WITH RECURSIVE
  pairs(idx) AS (
    SELECT 0
    UNION ALL
    SELECT idx+1 FROM pairs WHERE idx < {HEAD_DIM}/2 - 1
  ),
  heads(h) AS (
      SELECT 0
      UNION ALL
      SELECT h+1 FROM heads WHERE h < {HEADS} - 1
  )
INSERT INTO _rope_freqs_step1(head, pair_idx, freq)
SELECT h.h, p.idx, EXP(-2.0 * p.idx / {HEAD_DIM} * LOG({THETA}))
FROM heads h CROSS JOIN pairs p;

-- 2. Apply rotation to {TGT}
CREATE TEMP TABLE IF NOT EXISTS _rope_rotated (
    pos INTEGER,
    dim INTEGER,
    val REAL
);
DELETE FROM _rope_rotated;

-- Apply to first half (i < HEAD_DIM/2)
INSERT INTO _rope_rotated(pos, dim, val)
SELECT
    t.pos,
    t.dim,
    (t.val * COS(t.pos * f.freq)) - (t_other.val * SIN(t.pos * f.freq))
FROM {TGT} t
JOIN _rope_freqs_step1 f
  ON f.head = t.dim / {HEAD_DIM} AND f.pair_idx = (t.dim % {HEAD_DIM})
JOIN {TGT} t_other
  ON t_other.pos = t.pos AND t_other.dim = t.dim + ({HEAD_DIM} / 2)
WHERE (t.dim % {HEAD_DIM}) < ({HEAD_DIM} / 2);

-- Apply to second half (j >= HEAD_DIM/2)
INSERT INTO _rope_rotated(pos, dim, val)
SELECT
    t.pos,
    t.dim,
    (t.val * COS(t.pos * f.freq)) + (t_other.val * SIN(t.pos * f.freq))
FROM {TGT} t
JOIN _rope_freqs_step1 f
  ON f.head = t.dim / {HEAD_DIM} AND f.pair_idx = ((t.dim % {HEAD_DIM}) - ({HEAD_DIM} / 2))
JOIN {TGT} t_other
  ON t_other.pos = t.pos AND t_other.dim = t.dim - ({HEAD_DIM} / 2)
WHERE (t.dim % {HEAD_DIM}) >= ({HEAD_DIM} / 2);

-- 3. Replace {TGT}
DELETE FROM {TGT};
INSERT INTO {TGT}(pos, dim, val) SELECT pos, dim, val FROM _rope_rotated;
