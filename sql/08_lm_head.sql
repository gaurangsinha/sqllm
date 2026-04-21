-- lm_head.sql (template)
-- Projects the final hidden state of the LAST token into logits over the vocab space,
-- and greedily samples the next token ID (argmax).

DELETE FROM _logits;

CREATE TEMP TABLE IF NOT EXISTS _normed_last (pos INTEGER, dim INTEGER, val REAL);
DELETE FROM _normed_last;
INSERT INTO _normed_last(pos, dim, val) 
SELECT 0, dim, val FROM _normed WHERE pos = {LAST_POS};

INSERT INTO _logits(token_id, score)
SELECT dim AS token_id, val AS score
FROM matmul('_normed_last', 'model.embed_tokens.weight');

-- Return the sampled token ID
SELECT token_id FROM _logits ORDER BY score DESC LIMIT 1;
