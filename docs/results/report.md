# Results

BEIR NFCorpus test split: 323 queries, 3633 documents.

| System | nDCG@10 | Recall@100 | MRR@10 | Median latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|
| bm25 | 0.3218 | 0.2425 | 0.5272 | 0.4 | 0.7 |
| dense | 0.3437 | 0.3125 | 0.5308 | 336.1 | 724.3 |
| hybrid | 0.3650 | 0.3185 | 0.5825 | 336.7 | 725.0 |
| hybrid+rerank | 0.3528 | 0.3185 | 0.5791 | 14852.4 | 18115.2 |

Paired bootstrap over queries (difference = system minus baseline):

| Baseline | System | Metric | Difference | 95% CI | one-sided p |
|---|---|---|---|---|---|
| bm25 | dense | ndcg@10 | +0.0219 | [+0.0016, +0.0419] | 0.0180 |
| bm25 | dense | recall@100 | +0.0700 | [+0.0506, +0.0900] | 0.0000 |
| bm25 | dense | mrr@10 | +0.0036 | [-0.0361, +0.0411] | 0.4380 |
| bm25 | hybrid | ndcg@10 | +0.0432 | [+0.0303, +0.0563] | 0.0000 |
| bm25 | hybrid | recall@100 | +0.0760 | [+0.0597, +0.0924] | 0.0000 |
| bm25 | hybrid | mrr@10 | +0.0552 | [+0.0270, +0.0815] | 0.0000 |
| bm25 | hybrid+rerank | ndcg@10 | +0.0310 | [+0.0174, +0.0446] | 0.0000 |
| bm25 | hybrid+rerank | recall@100 | +0.0760 | [+0.0597, +0.0924] | 0.0000 |
| bm25 | hybrid+rerank | mrr@10 | +0.0519 | [+0.0220, +0.0834] | 0.0005 |
| hybrid | hybrid+rerank | ndcg@10 | -0.0123 | [-0.0259, +0.0014] | 0.9620 |
| hybrid | hybrid+rerank | recall@100 | +0.0000 | [+0.0000, +0.0000] | 1.0000 |
| hybrid | hybrid+rerank | mrr@10 | -0.0033 | [-0.0304, +0.0239] | 0.5880 |
| dense | hybrid | ndcg@10 | +0.0214 | [+0.0081, +0.0346] | 0.0005 |
| dense | hybrid | recall@100 | +0.0060 | [-0.0036, +0.0156] | 0.1185 |
| dense | hybrid | mrr@10 | +0.0516 | [+0.0221, +0.0831] | 0.0000 |

Config: `{"bm25": {"k1": 0.9, "b": 0.4}, "depth": 100, "rrf_k": 60, "rerank_depth": 100, "encoder": "BAAI/bge-small-en-v1.5", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2", "bootstrap": {"n_boot": 2000, "seed": 0}}`
