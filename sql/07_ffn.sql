-- ffn.sql (template)
-- SwiGLU Feed-Forward Network
-- ffn_mid = (gate_proj(x) * SiLU(gate_proj(x))) * up_proj(x)
-- {L_GATE} : gate_proj weight name
-- {L_UP}   : up_proj weight name

-- 1. Compute gate_proj into _gate
DELETE FROM _gate;
INSERT INTO _gate(pos, dim, val)
SELECT pos, dim, val FROM matmul('_normed', '{L_GATE}');

-- 2. Compute up_proj into _up
DELETE FROM _up;
INSERT INTO _up(pos, dim, val)
SELECT pos, dim, val FROM matmul('_normed', '{L_UP}');

-- 3. SwiGLU activation and element-wise multiply
-- ffn_mid = gate * (gate / (1 + exp(-gate))) * up
DELETE FROM _ffn_mid;
INSERT INTO _ffn_mid(pos, dim, val)
SELECT
    g.pos,
    g.dim,
    (g.val * (1.0 / (1.0 + EXP(-g.val)))) * u.val AS val
FROM _gate g
JOIN _up u ON u.pos = g.pos AND u.dim = g.dim;
