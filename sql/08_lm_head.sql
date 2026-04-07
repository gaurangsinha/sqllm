-- lm_head.sql (template)
-- Projects the final hidden state of the LAST token into logits over the vocab space,
-- and greedily samples the next token ID (argmax).

DELETE FROM _logits;

-- Only look at the final token's hidden state
INSERT INTO _logits(token_id, score)
SELECT
    w.i0 AS token_id,
    SUM(n.val * w.val) AS score
FROM _normed n
JOIN weights w ON w.name = 'model.embed_tokens.weight' AND w.i1 = n.dim
WHERE n.pos = {LAST_POS}
GROUP BY w.i0;

-- Return the sampled token ID
SELECT token_id FROM _logits ORDER BY score DESC LIMIT 1;
