# Optuna tuning of BM25 and hybrid fusion

200 TPE trials per study on 324 dev queries (seed 0); defaults were trial 0. Test: 323 queries, scored once per setting.

| System | Setting | Parameters | Dev nDCG@10 | Test nDCG@10 | Test Recall@100 | Test MRR@10 |
|---|---|---|---|---|---|---|
| bm25 | default | k1=0.9, b=0.4 | 0.2837 | 0.3218 | 0.2425 | 0.5272 |
| bm25 | tuned | k1=2.34, b=0.505 | 0.2890 | 0.3249 | 0.2421 | 0.5326 |
| hybrid | default | k1=0.9, b=0.4, rrf_k=60, w_dense=1 | 0.3349 | 0.3650 | 0.3185 | 0.5825 |
| hybrid | tuned | k1=2.72, b=0.145, rrf_k=14.6, w_dense=2.02 | 0.3415 | 0.3583 | 0.3233 | 0.5593 |

Tuned minus default on test (paired bootstrap, 95% CI):

- bm25 ndcg@10: +0.0031 [-0.0024, +0.0091], one-sided p = 0.131
- bm25 recall@100: -0.0003 [-0.0028, +0.0024], one-sided p = 0.599
- bm25 mrr@10: +0.0054 [-0.0083, +0.0187], one-sided p = 0.218
- hybrid ndcg@10: -0.0067 [-0.0148, +0.0016], one-sided p = 0.943
- hybrid recall@100: +0.0049 [-0.0017, +0.0116], one-sided p = 0.074
- hybrid mrr@10: -0.0232 [-0.0443, -0.0024], one-sided p = 0.984
