-- rmsnorm.sql  (template — Bash substitutes NORM_WEIGHT and SRC/DST tables)
--
-- RMSNorm:
--   rms  = sqrt( mean(x²) + eps )
--   x̂   = x / rms
--   out  = x̂ * weight
--
-- Called with:
--   NORM_WEIGHT : e.g. 'model.layers.0.input_layernorm.weight'
--   SRC         : source table  (_hidden)
--   DST         : destination   (_normed)
--   EPS         : rms_norm_eps  (1e-5)

-- Step 1: per-position RMS
CREATE TEMP TABLE IF NOT EXISTS _rms_tmp (
    pos     INTEGER PRIMARY KEY,
    rms_val REAL    NOT NULL
);
DELETE FROM _rms_tmp;

INSERT INTO _rms_tmp(pos, rms_val)
SELECT pos,
       SQRT( AVG(val * val) + {EPS} )
FROM   {SRC}
GROUP  BY pos;

-- Step 2: normalise and scale by learnable weight
DELETE FROM {DST};

INSERT INTO {DST}(pos, dim, val)
SELECT h.pos,
       h.dim,
       h.val / r.rms_val * w.val  AS val
FROM   {SRC}  h
JOIN   _rms_tmp r ON r.pos  = h.pos
JOIN   weights  w ON w.name  = '{NORM_WEIGHT}'
                 AND w.i0    = h.dim
                 AND w.i1    = 0;
