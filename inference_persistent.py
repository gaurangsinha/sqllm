#!/usr/bin/env python3
"""
inference_persistent.py
Option D: Persistent Python Driver

This script replaces the bash-driven `inference.sh` with a persistent Python
process that maintains a single SQLite connection. It utilizes prepared statements
with dynamic parameter binding (`?`) instead of concatenating raw SQL strings.
This avoids the overhead of constantly re-parsing and re-planning the AST on
every query execution within the transformer loop.
"""

import sys
import time
import sqlite3
import subprocess

def encode_prompt(prompt):
    result = subprocess.run(["python3", "tokenize_prompt.py", "encode", prompt], capture_output=True, text=True)
    return [int(x) for x in result.stdout.strip().split()]

def decode_token(tok):
    result = subprocess.run(["python3", "tokenize_prompt.py", "decode", str(tok)], capture_output=True, text=True)
    return result.stdout.replace('\n', '')

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    db_file = sys.argv[3] if len(sys.argv) > 3 else "model.db"

    print(f"Prompt: '{prompt}'")
    tokens = encode_prompt(prompt)
    print(f"Initial prompt tokens: {' '.join(map(str, tokens))}")

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    # Increase cache size for query planner tracking inside Python driver execution context
    cur.executescript("""
        PRAGMA cache_size = -2000000;
        PRAGMA mmap_size = 21474836480; 
        PRAGMA temp_store = MEMORY;
    """)

    def load_sql(filename):
        with open(filename, 'r') as f:
            return f.read()

    # Load templates and parameterize them for execute() binding.
    sql_schema = load_sql("sql/00_schema.sql")
    
    # 01 Embedding: Use ? for weight name
    sql_embedding = load_sql("sql/01_embedding.sql").replace("'model.embed_tokens.weight'", "?")
    
    # 02 RMSNorm: replace {SRC}, {DST}, {EPS} via string interpolation (table names/constants), replace '{NORM_WEIGHT}' with ?
    sql_rmsnorm = load_sql("sql/02_rmsnorm.sql").replace("{EPS}", "1e-5").replace("'{NORM_WEIGHT}'", "?")
    sql_rmsnorm_hidden_normed = sql_rmsnorm.replace("{SRC}", "_hidden").replace("{DST}", "_normed")
    
    # 03 QKV Proj: replace {TGT} via string, '{W}' with ?
    sql_q_proj = load_sql("sql/03_qkv_proj.sql").replace("{TGT}", "_q").replace("'{W}'", "?")
    sql_k_proj = load_sql("sql/03_qkv_proj.sql").replace("{TGT}", "_k").replace("'{W}'", "?")
    sql_v_proj = load_sql("sql/03_qkv_proj.sql").replace("{TGT}", "_v").replace("'{W}'", "?")
    
    # 04 RoPE
    sql_rope_q = load_sql("sql/04_rope.sql").replace("{TGT}", "_q").replace("{THETA}", "10000.0").replace("{HEADS}", "9").replace("{HEAD_DIM}", "64")
    sql_rope_k = load_sql("sql/04_rope.sql").replace("{TGT}", "_k").replace("{THETA}", "10000.0").replace("{HEADS}", "3").replace("{HEAD_DIM}", "64")
    
    # 05 Attention
    sql_attention = load_sql("sql/05_attention.sql").replace("{NUM_Q_HEADS}", "9").replace("{NUM_KV_HEADS}", "3").replace("{HEAD_DIM}", "64").replace("{REPEATS}", "3")
    
    # 06 Proj and Add
    sql_proj = load_sql("sql/06_proj_and_add.sql").replace("'{W}'", "?").replace("{SRC}", "_hidden")
    sql_proj_attn = sql_proj.replace("{TGT}", "_attn_ctx")
    sql_proj_ffn = sql_proj.replace("{TGT}", "_ffn_mid")
    
    # 07 FFN
    sql_ffn = load_sql("sql/07_ffn.sql").replace("'{L_GATE}'", "?").replace("'{L_UP}'", "?")
    
    # 08 LM Head
    sql_lm_head = load_sql("sql/08_lm_head.sql")

    print("Running inference loop...")
    for step in range(max_new_tokens):
        print(f"Processing token step {step+1}/{max_new_tokens}...")
        t0 = time.time()
        
        last_pos = len(tokens) - 1

        # Triggers schema creation
        cur.executescript(sql_schema)
        
        cur.execute("DELETE FROM _seq")
        cur.executemany("INSERT INTO _seq(pos, token_id) VALUES (?, ?)", enumerate(tokens))
        
        cur.executescript("DELETE FROM _hidden;")
        cur.execute(sql_embedding.split("DELETE FROM _hidden;\n")[1], ("model.embed_tokens.weight",))
        
        for L in range(30):
            # RMSNorm
            cur.executescript(sql_rmsnorm_hidden_normed.split("DELETE FROM _rms_tmp;")[0] + "DELETE FROM _rms_tmp;")
            cur.execute(sql_rmsnorm_hidden_normed.split("DELETE FROM _rms_tmp;\n\n")[1].split("-- Step 2:")[0].strip())
            cur.executescript("DELETE FROM _normed;")
            cur.execute(sql_rmsnorm_hidden_normed.split("DELETE FROM _normed;\n\n")[1].strip(), (f"model.layers.{L}.input_layernorm.weight",))
            
            # QKV
            cur.executescript("DELETE FROM _q;")
            cur.execute(sql_q_proj.split("DELETE FROM _q;\n\n")[1].strip(), (f"model.layers.{L}.self_attn.q_proj.weight",))
            cur.executescript("DELETE FROM _k;")
            cur.execute(sql_k_proj.split("DELETE FROM _k;\n\n")[1].strip(), (f"model.layers.{L}.self_attn.k_proj.weight",))
            cur.executescript("DELETE FROM _v;")
            cur.execute(sql_v_proj.split("DELETE FROM _v;\n\n")[1].strip(), (f"model.layers.{L}.self_attn.v_proj.weight",))
            
            # RoPE
            cur.executescript(sql_rope_q)
            cur.executescript(sql_rope_k)
            
            # Attention
            cur.executescript(sql_attention)
            
            # Proj and Add (Attn)
            cur.executescript(sql_proj_attn.split("DELETE FROM _proj_tmp;\n\n")[0] + "DELETE FROM _proj_tmp;")
            cur.execute(sql_proj_attn.split("DELETE FROM _proj_tmp;\n\n")[1].split("-- Now add")[0].strip(), (f"model.layers.{L}.self_attn.o_proj.weight",))
            cur.executescript(sql_proj_attn.split("-- Now add to residual (_hidden)\n")[1].split("INSERT INTO _hidden_new(pos, dim, val)")[0] + "DELETE FROM _hidden_new;")
            cur.executescript("INSERT INTO _hidden_new(pos, dim, val)\n" + sql_proj_attn.split("INSERT INTO _hidden_new(pos, dim, val)\n")[1])
            
            # FFN RMSNorm
            cur.executescript(sql_rmsnorm_hidden_normed.split("DELETE FROM _rms_tmp;")[0] + "DELETE FROM _rms_tmp;")
            cur.execute(sql_rmsnorm_hidden_normed.split("DELETE FROM _rms_tmp;\n\n")[1].split("-- Step 2:")[0].strip())
            cur.executescript("DELETE FROM _normed;")
            cur.execute(sql_rmsnorm_hidden_normed.split("DELETE FROM _normed;\n\n")[1].strip(), (f"model.layers.{L}.post_attention_layernorm.weight",))
            
            # FFN
            cur.executescript("DELETE FROM _gate;")
            cur.execute(sql_ffn.split("DELETE FROM _gate;\n")[1].split("-- 2.")[0].strip(), (f"model.layers.{L}.mlp.gate_proj.weight",))
            cur.executescript("DELETE FROM _up;")
            cur.execute(sql_ffn.split("DELETE FROM _up;\n")[1].split("-- 3.")[0].strip(), (f"model.layers.{L}.mlp.up_proj.weight",))
            cur.executescript(sql_ffn.split("-- 3. SwiGLU activation and element-wise multiply\n")[1])
            
            # Proj and Add (FFN)
            cur.executescript(sql_proj_ffn.split("DELETE FROM _proj_tmp;\n\n")[0] + "DELETE FROM _proj_tmp;")
            cur.execute(sql_proj_ffn.split("DELETE FROM _proj_tmp;\n\n")[1].split("-- Now")[0].strip(), (f"model.layers.{L}.mlp.down_proj.weight",))
            cur.executescript(sql_proj_ffn.split("-- Now add to residual (_hidden)\n")[1].split("INSERT INTO _hidden_new(pos, dim, val)")[0] + "DELETE FROM _hidden_new;")
            cur.executescript("INSERT INTO _hidden_new(pos, dim, val)\n" + sql_proj_ffn.split("INSERT INTO _hidden_new(pos, dim, val)\n")[1])

        # Final RMSNorm
        cur.executescript(sql_rmsnorm_hidden_normed.split("DELETE FROM _rms_tmp;")[0] + "DELETE FROM _rms_tmp;")
        cur.execute(sql_rmsnorm_hidden_normed.split("DELETE FROM _rms_tmp;\n\n")[1].split("-- Step 2:")[0].strip())
        cur.executescript("DELETE FROM _normed;")
        cur.execute(sql_rmsnorm_hidden_normed.split("DELETE FROM _normed;\n\n")[1].strip(), ("model.norm.weight",))
        
        # LM Head
        cur.executescript("DELETE FROM _logits;")
        cur.execute(sql_lm_head.replace("{LAST_POS}", str(last_pos)).split("DELETE FROM _logits;\n\n")[1].split("-- Return")[0].strip())
        res = cur.execute(sql_lm_head.split("-- Return the sampled token ID\n")[1].strip()).fetchone()
        
        next_token = res[0]
        duration = time.time() - t0
        
        tokens.append(next_token)
        decoded = decode_token(next_token)
        print(f"==> Got token: {next_token} ('{decoded}') in {duration:.1f} seconds.")

    print("\nFinal tokens:", " ".join(map(str, tokens)))
    final_text = "".join(decode_token(t) for t in tokens)
    print("Final text:")
    print(final_text)

if __name__ == "__main__":
    main()
