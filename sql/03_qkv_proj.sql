-- qkv_proj.sql (template)
-- Linear projections for Q, K, V
-- {TGT}  : Target table (_q, _k, _v)
-- {W}    : Weight matrix name ('model.layers.0.self_attn.q_proj.weight')

-- W shape: [out_features, in_features]
-- W is stored as i0=out_feature, i1=in_feature
-- We compute: TGT_dim(i0) = SUM( _normed(i1) * W(i0, i1) )

DELETE FROM {TGT};

INSERT INTO {TGT}(pos, dim, val)
SELECT pos,
       dim,
       val
FROM   matmul('_normed', '{W}');
