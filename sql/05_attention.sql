-- attention.sql (template)
-- GQA (Grouped Query Attention) and scaled dot-product attention
-- SmolLM-135M:
--   _q has 9 heads (dims 0..575)
--   _k has 3 heads (dims 0..191)
--   _v has 3 heads (dims 0..191)
--   head_dim = 64
--
-- We need to compute:
--   scores = Q @ K^T / sqrt(head_dim)
--   weights = softmax(scores + mask)   (causal mask)
--   context = weights @ V

-- {NUM_Q_HEADS}  = 9
-- {NUM_KV_HEADS} = 3
-- {HEAD_DIM}     = 64
-- {REPEATS}      = NUM_Q_HEADS / NUM_KV_HEADS = 3

-- 1. Compute scores: Q @ K^T / SQRT(HEAD_DIM)
-- Note K is repeated: q_head h uses kv_head (h / REPEATS)
DELETE FROM _attn_scores;

INSERT INTO _attn_scores(q_pos, k_pos, q_head, score)
SELECT
    q.pos AS q_pos,
    k.pos AS k_pos,
    q.dim / {HEAD_DIM} AS q_head,
    SUM(q.val * k.val) / SQRT({HEAD_DIM}) AS score
FROM _q q
JOIN _k k
  -- match heads with repeats:
  -- k.dim / HEAD_DIM = (q.dim / HEAD_DIM) / REPEATS
  ON (k.dim / {HEAD_DIM}) = ((q.dim / {HEAD_DIM}) / {REPEATS})
  -- match dimension within the head
  AND (k.dim % {HEAD_DIM}) = (q.dim % {HEAD_DIM})
WHERE q.pos >= k.pos   -- Causal masking!
GROUP BY q.pos, k.pos, q_head;

-- 2. Softmax over k_pos for each (q_pos, q_head)
-- Find max for numerical stability
CREATE TEMP TABLE IF NOT EXISTS _attn_max (
    q_pos INTEGER,
    q_head INTEGER,
    max_score REAL,
    PRIMARY KEY(q_pos, q_head)
) WITHOUT ROWID;
DELETE FROM _attn_max;

INSERT INTO _attn_max(q_pos, q_head, max_score)
SELECT q_pos, q_head, MAX(score)
FROM _attn_scores
GROUP BY q_pos, q_head;

-- Compute exp(score - max) and sum
CREATE TEMP TABLE IF NOT EXISTS _attn_exp (
    q_pos INTEGER,
    k_pos INTEGER,
    q_head INTEGER,
    exp_score REAL,
    PRIMARY KEY(q_pos, k_pos, q_head)
) WITHOUT ROWID;
DELETE FROM _attn_exp;

INSERT INTO _attn_exp(q_pos, k_pos, q_head, exp_score)
SELECT s.q_pos, s.k_pos, s.q_head, EXP(s.score - m.max_score)
FROM _attn_scores s
JOIN _attn_max m ON s.q_pos = m.q_pos AND s.q_head = m.q_head;

CREATE TEMP TABLE IF NOT EXISTS _attn_sum (
    q_pos INTEGER,
    q_head INTEGER,
    sum_exp REAL,
    PRIMARY KEY(q_pos, q_head)
) WITHOUT ROWID;
DELETE FROM _attn_sum;

INSERT INTO _attn_sum(q_pos, q_head, sum_exp)
SELECT q_pos, q_head, SUM(exp_score)
FROM _attn_exp
GROUP BY q_pos, q_head;

-- Compute final weights: exp / sum
DELETE FROM _attn_weights;
INSERT INTO _attn_weights(q_pos, k_pos, q_head, weight)
SELECT e.q_pos, e.k_pos, e.q_head, e.exp_score / s.sum_exp
FROM _attn_exp e
JOIN _attn_sum s ON e.q_pos = s.q_pos AND e.q_head = s.q_head;

-- 3. Context: weights @ V
-- V head is q_head / REPEATS
DELETE FROM _attn_ctx;

INSERT INTO _attn_ctx(pos, dim, val)
SELECT
    w.q_pos AS pos,
    (w.q_head * {HEAD_DIM}) + (v.dim % {HEAD_DIM}) AS dim,
    SUM(w.weight * v.val) AS val
FROM _attn_weights w
JOIN _v v
  ON v.pos = w.k_pos
  AND (v.dim / {HEAD_DIM}) = (w.q_head / {REPEATS})
GROUP BY w.q_pos, w.q_head, v.dim % {HEAD_DIM};
