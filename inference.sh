#!/usr/bin/env bash
# inference.sh
# Orchestrates SmolLM-135M using pure SQL for the forward pass

# Fetch prompt configs from CLI args; defaults to basic inference
prompt="${1:-The capital of France is}"
max_new_tokens="${2:-1}"
# Allows hot-swapping DB targets (like model_cover.db mapping Covered Indexes)
db_file="${3:-model.db}"
# 4th arg: path to sqlite3 binary (default: sqlite3 from PATH).
# Use this to compare stock vs optimised builds:
#   bash inference.sh "The capital of France is" 1 model.db /path/to/optimised/sqlite3
SQLITE_BIN="${4:-sqlite3}"

if [[ ! -f "$db_file" ]]; then
    echo "Error: $db_file not found. Run load_model.py first."
    exit 1
fi

echo "Prompt: '$prompt'"
# Call out to Python solely to trigger Hugging Face byte-pair tokenization on our sequence string 
tokens=$(python3 tokenize_prompt.py encode "$prompt")
echo "Initial prompt tokens: $tokens"

# Pre-process SQL files to create the forward-pass execution script
# We construct the execution block once (using basic sed replacements)
# to avoid repeatedly spinning up distinct sqlite3 runtime environments sequentially.
build_sql_script() {
    local tokens="$1"
    local num_tokens=$(echo $tokens | wc -w)
    # Positions are 0-indexed across relational SQL tables
    local last_pos=$((num_tokens - 1))

    # 00_schema.sql: Configures PRAGMAs, MEMORY layouts, and initializes clustered temp tracking
    cat sql/00_schema.sql

    # Feed input tokens directly into raw SQL statements mapping to `_seq` Temp DB table
    echo "DELETE FROM _seq;"
    local pos=0
    for tok in $tokens; do
        echo "INSERT INTO _seq(pos, token_id) VALUES ($pos, $tok);"
        pos=$((pos + 1))
    done

    # 1. Embedding Matrix Generation (Translating Integer Token IDs -> Embeddings Vectors)
    cat sql/01_embedding.sql

    # 2. 30 Transformer Layers Loop
    # We unroll loops using raw loop bash string interpolation mapping weights dynamically replacing variables `{W}`!
    for L in {0..29}; do
        echo "-- +++ LAYER $L +++"
        # Attention RMSNorm Pre-computations (stabilizes numeric data variance before routing logic computes)
        sed -e "s/{NORM_WEIGHT}/model.layers.${L}.input_layernorm.weight/g" \
            -e "s/{SRC}/_hidden/g" \
            -e "s/{DST}/_normed/g" \
            -e "s/{EPS}/1e-5/g" sql/02_rmsnorm.sql

        # Q, K, V Projections
        sed -e "s/{TGT}/_q/g" -e "s/{W}/model.layers.${L}.self_attn.q_proj.weight/g" sql/03_qkv_proj.sql
        sed -e "s/{TGT}/_k/g" -e "s/{W}/model.layers.${L}.self_attn.k_proj.weight/g" sql/03_qkv_proj.sql
        sed -e "s/{TGT}/_v/g" -e "s/{W}/model.layers.${L}.self_attn.v_proj.weight/g" sql/03_qkv_proj.sql

        # RoPE on Q, K
        sed -e "s/{TGT}/_q/g" -e "s/{THETA}/10000.0/g" -e "s/{HEADS}/9/g" -e "s/{HEAD_DIM}/64/g" sql/04_rope.sql
        sed -e "s/{TGT}/_k/g" -e "s/{THETA}/10000.0/g" -e "s/{HEADS}/3/g" -e "s/{HEAD_DIM}/64/g" sql/04_rope.sql

        # Attention
        sed -e "s/{NUM_Q_HEADS}/9/g" -e "s/{NUM_KV_HEADS}/3/g" -e "s/{HEAD_DIM}/64/g" -e "s/{REPEATS}/3/g" sql/05_attention.sql

        # Output Projection + Residual Add
        sed -e "s/{TGT}/_attn_ctx/g" \
            -e "s/{W}/model.layers.${L}.self_attn.o_proj.weight/g" \
            -e "s/{SRC}/_hidden/g" sql/06_proj_and_add.sql

        # FFN RMSNorm
        sed -e "s/{NORM_WEIGHT}/model.layers.${L}.post_attention_layernorm.weight/g" \
            -e "s/{SRC}/_hidden/g" \
            -e "s/{DST}/_normed/g" \
            -e "s/{EPS}/1e-5/g" sql/02_rmsnorm.sql

        # FFN (SwiGLU)
        sed -e "s/{L_GATE}/model.layers.${L}.mlp.gate_proj.weight/g" \
            -e "s/{L_UP}/model.layers.${L}.mlp.up_proj.weight/g" sql/07_ffn.sql

        # FFN Output Projection + Residual Add
        sed -e "s/{TGT}/_ffn_mid/g" \
            -e "s/{W}/model.layers.${L}.mlp.down_proj.weight/g" \
            -e "s/{SRC}/_hidden/g" sql/06_proj_and_add.sql
    done

    # 3. Final RMSNorm Stabilization
    sed -e "s/{NORM_WEIGHT}/model.norm.weight/g" \
        -e "s/{SRC}/_hidden/g" \
        -e "s/{DST}/_normed/g" \
        -e "s/{EPS}/1e-5/g" sql/02_rmsnorm.sql

    # 4. LM Head & Sampling
    # Matches logits against vocabulary matrix outputting token IDs strictly sorting max values via pure native SQLite
    sed -e "s/{LAST_POS}/${last_pos}/g" sql/08_lm_head.sql
    echo "COMMIT;"
}

# Add SQLite math extension load if needed, but modern SQLite has math functions built-in
SQLITE_EXTRA_ARGS=""

# Main Generation Pipeline Iteration
echo "Running inference loop..."
for ((i=0; i<$max_new_tokens; i++)); do
    echo "Processing token step $((i+1))/$max_new_tokens..."
    start_time=$(date +%s)

    # Dumps interpolated static bash structure generation locally mapping unique SQLite transactions mapping layer dependencies.
    script_file="/tmp/sql_inference_$$.sql"
    build_sql_script "$tokens" > "$script_file"

    # Triggers absolute SQLite engine passing parameters and capturing the exact final prediction. `tail` parses native SQL output!
    next_token=$("$SQLITE_BIN" $SQLITE_EXTRA_ARGS "$db_file" < "$script_file" | tail -n 1)

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [[ -z "$next_token" ]]; then
       echo "Error: Query returned no token."
       break
    fi

    tokens="$tokens $next_token"
    decoded=$(python3 tokenize_prompt.py decode "$next_token")
    echo "==> Got token: $next_token ('$decoded') in $duration seconds."
done

echo ""
echo "Final tokens: $tokens"
FINAL_STR=""
for t in $tokens; do
   d=$(python3 tokenize_prompt.py decode "$t")
   FINAL_STR="${FINAL_STR}${d}"
done
echo "Final text:"
echo "$FINAL_STR"

#rm -f /tmp/sql_inference_$$.sql
