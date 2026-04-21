#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __clang__
#define SIMD_LOOP _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
#define SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SIMD_LOOP
#endif

typedef struct MatmulVtab MatmulVtab;
struct MatmulVtab {
    sqlite3_vtab base;
    sqlite3 *db;
};

typedef struct MatmulCursor MatmulCursor;
struct MatmulCursor {
    sqlite3_vtab_cursor base;
    float *out_data;
    int *out_pos;
    int P;
    int out_features;
    int current_pos;
    int current_dim;
};

static int matmulConnect(sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVTab, char **pzErr){
    MatmulVtab *pNew = sqlite3_malloc(sizeof(*pNew));
    if(!pNew) return SQLITE_NOMEM;
    memset(pNew, 0, sizeof(*pNew));
    pNew->db = db;
    int rc = sqlite3_declare_vtab(db, "CREATE TABLE x(pos INTEGER, dim INTEGER, val REAL, hidden_table HIDDEN, weight_name HIDDEN)");
    *ppVTab = (sqlite3_vtab*)pNew;
    return rc;
}

static int matmulDisconnect(sqlite3_vtab *pVtab){
    sqlite3_free(pVtab);
    return SQLITE_OK;
}

static int matmulOpen(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCursor){
    MatmulCursor *pCur = sqlite3_malloc(sizeof(*pCur));
    if(!pCur) return SQLITE_NOMEM;
    memset(pCur, 0, sizeof(*pCur));
    *ppCursor = &pCur->base;
    return SQLITE_OK;
}

static int matmulClose(sqlite3_vtab_cursor *cur){
    MatmulCursor *pCur = (MatmulCursor*)cur;
    if(pCur->out_data) free(pCur->out_data);
    if(pCur->out_pos) free(pCur->out_pos);
    sqlite3_free(pCur);
    return SQLITE_OK;
}

static int matmulBestIndex(sqlite3_vtab *pVtab, sqlite3_index_info *pIdxInfo){
    int hidden_table_idx = -1;
    int weight_name_idx = -1;
    
    for(int i = 0; i < pIdxInfo->nConstraint; i++){
        if(pIdxInfo->aConstraint[i].usable && pIdxInfo->aConstraint[i].op == SQLITE_INDEX_CONSTRAINT_EQ){
            if(pIdxInfo->aConstraint[i].iColumn == 3) hidden_table_idx = i;
            if(pIdxInfo->aConstraint[i].iColumn == 4) weight_name_idx = i;
        }
    }
    
    if(hidden_table_idx < 0 || weight_name_idx < 0) return SQLITE_CONSTRAINT;
    
    pIdxInfo->aConstraintUsage[hidden_table_idx].argvIndex = 1;
    pIdxInfo->aConstraintUsage[hidden_table_idx].omit = 1;
    
    pIdxInfo->aConstraintUsage[weight_name_idx].argvIndex = 2;
    pIdxInfo->aConstraintUsage[weight_name_idx].omit = 1;
    
    pIdxInfo->estimatedCost = 100.0;
    return SQLITE_OK;
}

static int matmulFilter(sqlite3_vtab_cursor *pVtabCursor, int idxNum, const char *idxStr, int argc, sqlite3_value **argv){
    MatmulCursor *pCur = (MatmulCursor*)pVtabCursor;
    MatmulVtab *pVtab = (MatmulVtab*)pVtabCursor->pVtab;
    
    if(pCur->out_data) {
        free(pCur->out_data);
        pCur->out_data = 0;
    }
    if(pCur->out_pos) {
        free(pCur->out_pos);
        pCur->out_pos = 0;
    }
    
    const char *hidden_table = (const char*)sqlite3_value_text(argv[0]);
    const char *weight_name = (const char*)sqlite3_value_text(argv[1]);
    
    if(!hidden_table || !weight_name) return SQLITE_ERROR;

    sqlite3_stmt *stmt_w = 0;
    int rc = sqlite3_prepare_v2(pVtab->db, "SELECT out_features, in_features, data FROM weight_blobs WHERE name = ?", -1, &stmt_w, 0);
    if(rc != SQLITE_OK) return rc;
    sqlite3_bind_text(stmt_w, 1, weight_name, -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt_w);
    if(rc != SQLITE_ROW) {
        sqlite3_finalize(stmt_w);
        return SQLITE_ERROR;
    }
    
    int out_features = sqlite3_column_int(stmt_w, 0);
    int in_features = sqlite3_column_int(stmt_w, 1);
    const float *w_data = (const float*)sqlite3_column_blob(stmt_w, 2);
    int blob_bytes = sqlite3_column_bytes(stmt_w, 2);
    if(blob_bytes != out_features * in_features * sizeof(float)){
       sqlite3_finalize(stmt_w);
       return SQLITE_ERROR;
    }
    
    char *sql_h = sqlite3_mprintf("SELECT pos, dim, val FROM %Q ORDER BY pos, dim", hidden_table);
    sqlite3_stmt *stmt_h = 0;
    rc = sqlite3_prepare_v2(pVtab->db, sql_h, -1, &stmt_h, 0);
    sqlite3_free(sql_h);
    if(rc != SQLITE_OK){
        sqlite3_finalize(stmt_w);
        return rc;
    }
    
    int P = 0;
    int max_pos = -1;
    while(sqlite3_step(stmt_h) == SQLITE_ROW){
        int pos = sqlite3_column_int(stmt_h, 0);
        if(pos > max_pos) max_pos = pos;
    }
    pCur->out_pos = 0;
    P = max_pos + 1;
    if(P <= 0 || in_features <= 0 || out_features <= 0){
        sqlite3_finalize(stmt_h);
        sqlite3_finalize(stmt_w);
        pCur->P = 0; pCur->out_features = 0; pCur->out_data = 0;
        return SQLITE_OK;
    }
    
    float *h_data = calloc(P * in_features, sizeof(float));
    char *active_pos = calloc(P, sizeof(char));
    sqlite3_reset(stmt_h);
    while(sqlite3_step(stmt_h) == SQLITE_ROW){
        int pos = sqlite3_column_int(stmt_h, 0);
        int dim = sqlite3_column_int(stmt_h, 1);
        float val = sqlite3_column_double(stmt_h, 2);
        if(pos >= 0 && pos < P && dim >= 0 && dim < in_features){
            h_data[pos * in_features + dim] = val;
            active_pos[pos] = 1;
        }
    }
    sqlite3_finalize(stmt_h);
    
    int num_active = 0;
    for(int p=0; p<P; p++){ if(active_pos[p]) num_active++; }

    float *out_data = calloc(num_active * out_features, sizeof(float));
    int *out_pos = malloc(num_active * sizeof(int));
    
    int act_idx = 0;
    for(int p = 0; p < P; p++){
        if(!active_pos[p]) continue;
        out_pos[act_idx] = p;
        const float *h_vec = &h_data[p * in_features];
        float *out_vec = &out_data[act_idx * out_features];
        for(int o = 0; o < out_features; o++){
            const float *w_row = &w_data[o * in_features];
            float sum = 0.0f;
            SIMD_LOOP
            for(int i = 0; i < in_features; i++){
                sum += h_vec[i] * w_row[i];
            }
            out_vec[o] = sum;
        }
        act_idx++;
    }
    
    sqlite3_finalize(stmt_w);
    free(h_data);
    free(active_pos);
    
    pCur->P = num_active;
    pCur->out_features = out_features;
    pCur->out_data = out_data;
    pCur->out_pos = out_pos;
    pCur->current_pos = 0;
    pCur->current_dim = 0;
    
    return SQLITE_OK;
}

static int matmulNext(sqlite3_vtab_cursor *cur){
    MatmulCursor *pCur = (MatmulCursor*)cur;
    pCur->current_dim++;
    if(pCur->current_dim >= pCur->out_features){
        pCur->current_dim = 0;
        pCur->current_pos++;
    }
    return SQLITE_OK;
}

static int matmulEof(sqlite3_vtab_cursor *cur){
    MatmulCursor *pCur = (MatmulCursor*)cur;
    return (pCur->current_pos >= pCur->P);
}

static int matmulColumn(sqlite3_vtab_cursor *cur, sqlite3_context *ctx, int i){
    MatmulCursor *pCur = (MatmulCursor*)cur;
    if(i == 0){
        sqlite3_result_int(ctx, pCur->out_pos ? pCur->out_pos[pCur->current_pos] : pCur->current_pos);
    }else if(i == 1){
        sqlite3_result_int(ctx, pCur->current_dim);
    }else if(i == 2){
        float val = pCur->out_data[pCur->current_pos * pCur->out_features + pCur->current_dim];
        sqlite3_result_double(ctx, val);
    }else{
        sqlite3_result_null(ctx);
    }
    return SQLITE_OK;
}

static int matmulRowid(sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid){
    MatmulCursor *pCur = (MatmulCursor*)cur;
    *pRowid = pCur->current_pos * pCur->out_features + pCur->current_dim;
    return SQLITE_OK;
}

static sqlite3_module matmulModule = {
  0,               /* iVersion */
  0,               /* xCreate */
  matmulConnect,   /* xConnect */
  matmulBestIndex, /* xBestIndex */
  matmulDisconnect,/* xDisconnect */
  0,               /* xDestroy */
  matmulOpen,      /* xOpen */
  matmulClose,     /* xClose */
  matmulFilter,    /* xFilter */
  matmulNext,      /* xNext */
  matmulEof,       /* xEof */
  matmulColumn,    /* xColumn */
  matmulRowid,     /* xRowid */
  0,0,0,0,0,0,0,0
};

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_matmul_init(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi){
  int rc = SQLITE_OK;
  SQLITE_EXTENSION_INIT2(pApi);
  rc = sqlite3_create_module(db, "matmul", &matmulModule, 0);
  return rc;
}
