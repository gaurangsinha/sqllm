-- proj_and_add.sql (template)
-- Projects back to hidden dimension and adds to residual stream in-place.
-- {TGT}  : Table to projection from (e.g. _attn_ctx or _ffn_mid)
-- {W}    : Weight matrix 'model.layers.0.self_attn.o_proj.weight'
-- {SRC}  : Target table for residual add (_hidden)

-- First do projection into a temp table
CREATE TEMP TABLE IF NOT EXISTS _proj_tmp (
    pos INTEGER,
    dim INTEGER,
    val REAL,
    PRIMARY KEY(pos, dim)
) WITHOUT ROWID;
DELETE FROM _proj_tmp;

INSERT INTO _proj_tmp(pos, dim, val)
SELECT pos, dim, val FROM matmul('{TGT}', '{W}');

-- Now add to residual (_hidden)
-- SQLite doesn't have an easy UPDATE ... FROM for this specific aggregation pattern
-- in older versions without making it complex, so we'll rewrite _hidden.
CREATE TEMP TABLE IF NOT EXISTS _hidden_new (
    pos INTEGER,
    dim INTEGER,
    val REAL,
    PRIMARY KEY(pos, dim)
) WITHOUT ROWID;
DELETE FROM _hidden_new;

INSERT INTO _hidden_new(pos, dim, val)
SELECT
    h.pos,
    h.dim,
    h.val + p.val
FROM {SRC} h
JOIN _proj_tmp p ON p.pos = h.pos AND p.dim = h.dim;

DELETE FROM {SRC};
INSERT INTO {SRC}(pos, dim, val) SELECT pos, dim, val FROM _hidden_new;
