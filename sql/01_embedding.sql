-- embedding.sql
-- Populate _hidden from embed_tokens.weight for each token in _seq.
-- embed_tokens.weight shape: [vocab_size, hidden_size] = [49152, 576]
-- stored as weights(name, i0=token_id, i1=dim, val)

DELETE FROM _hidden;

INSERT INTO _hidden(pos, dim, val)
SELECT s.pos,
       w.i1   AS dim,
       w.val
FROM   _seq s
JOIN   weights w
          ON  w.name = 'model.embed_tokens.weight'
          AND w.i0   = s.token_id;
