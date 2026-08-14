# -*- coding: utf-8 -*-
"""BGE-small-en-v1.5 dense-retrieval baseline on all four datasets.

Adds a stronger dense baseline (BAAI/bge-small-en-v1.5, 384-dim, 33M params)
alongside all-MiniLM-L6-v2 ('SBERT-Dense' row in the main tables).
Official BGE usage: queries are prefixed with the retrieval instruction;
passages are encoded raw. Embeddings L2-normalized, inner product = cosine.

Checkpointed in 1000-doc sub-chunks; safe to re-run after interruption.

Usage:
    python bge_baseline.py [dataset]     # default: all four
"""
import json
import os
import sys

import numpy as np

from reproduce import (ART, RES, load_jsonl, load_qrels, build_eval_arrays,
                       per_query_metrics)

import _layout as L

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'models', 'bge-small')
QUERY_INSTRUCTION = ('Represent this sentence for searching relevant '
                     'passages: ')

DATASETS = {ds: L.raw_ds(ds)
            for ds in ('scidocs', 'scifact', 'nfcorpus', 'trec-covid')}


def ds_paths(ds):
    base = DATASETS[ds]
    return (os.path.join(base, 'corpus.jsonl'),
            os.path.join(base, 'queries.jsonl'),
            os.path.join(base, 'qrels', 'test.tsv'))


def encode_corpus(model, ds):
    corpus_path, queries_path, _ = ds_paths(ds)
    docs = load_jsonl(corpus_path)
    out_dir = L.emb_dir(ds, bge=True)
    os.makedirs(out_dir, exist_ok=True)
    starts = sorted(int(f[6:-4]) for f in os.listdir(out_dir)
                    if f.startswith('chunk_') and f.endswith('.npy'))
    done = 0
    for s in starts:
        if s != done:
            break
        done += int(np.load(os.path.join(out_dir, f'chunk_{s}.npy'),
                            mmap_mode='r').shape[0])
    print(ds, 'BGE already encoded:', done, '/', len(docs), flush=True)
    SUB = 1000
    for s in range(done, len(docs), SUB):
        part = docs[s:s + SUB]
        texts = [(d.get('title') or '') + ' ' + (d.get('text') or '')
                 for d in part]
        emb = model.encode(texts, batch_size=64, show_progress_bar=False,
                           normalize_embeddings=True).astype(np.float32)
        tmp = os.path.join(out_dir, f'chunk_{s}.tmp.npy')
        np.save(tmp, emb)
        os.replace(tmp, os.path.join(out_dir, f'chunk_{s}.npy'))
    ids = [str(d['_id']) for d in docs]
    json.dump(ids, open(os.path.join(out_dir, 'ids.json'), 'w'))

    qs = load_jsonl(queries_path)
    q_emb = model.encode([QUERY_INSTRUCTION + (q.get('text') or '') for q in qs],
                         batch_size=64, show_progress_bar=False,
                         normalize_embeddings=True).astype(np.float32)
    np.save(L.art_path(ds, f'{ds}_bge_qemb.npy'), q_emb)
    json.dump([str(q['_id']) for q in qs],
              open(L.art_path(ds, f'{ds}_bge_qids.json'), 'w'))
    print(ds, 'BGE encoded:', len(ids), 'docs,', len(qs), 'queries',
          flush=True)
    return ids


def evaluate(ds):
    out_dir = L.emb_dir(ds, bge=True)
    ids = json.load(open(os.path.join(out_dir, 'ids.json')))
    starts = sorted(int(f[6:-4]) for f in os.listdir(out_dir)
                    if f.startswith('chunk_') and f.endswith('.npy'))
    E = np.vstack([np.load(os.path.join(out_dir, f'chunk_{s}.npy'))
                   for s in starts])
    assert len(E) == len(ids), f'{ds}: embedding/id mismatch'
    didx = {d: i for i, d in enumerate(ids)}

    _, _, qrels_path = ds_paths(ds)
    qrels = load_qrels(qrels_path)
    qids = sorted(qrels.keys())
    q_emb = np.load(L.art_path(ds, f'{ds}_bge_qemb.npy'))
    q_emb_ids = json.load(open(L.art_path(ds, f'{ds}_bge_qids.json')))
    qemb_map = {q: q_emb[i] for i, q in enumerate(q_emb_ids)}
    Q = np.stack([qemb_map[q] for q in qids])

    S = (Q @ E.T).astype(np.float64)
    orders = [np.argsort(-S[qi])[:100] for qi in range(len(qids))]
    rel_list, gains_list = build_eval_arrays(qrels, qids, didx, len(ids))
    res = per_query_metrics(orders, rel_list, gains_list)
    avg = {k: float(np.mean(v)) for k, v in res.items()}
    print(ds, 'BGE-Dense:', {k: round(v, 4) for k, v in avg.items()},
          flush=True)
    os.makedirs(RES, exist_ok=True)
    np.savez_compressed(os.path.join(RES, f'{ds}_bge_perquery.npz'),
                        qids=np.array(qids),
                        **{f'BGE-Dense||{k}': np.array(v)
                           for k, v in res.items()})
    json.dump({'n_queries': len(qids), 'avg': avg},
              open(os.path.join(RES, f'bge_{ds}.json'), 'w'), indent=1)


def get_model():
    """Load BGE-small-en-v1.5; download from HuggingFace on first use."""
    from sentence_transformers import SentenceTransformer
    if os.path.exists(MODEL_DIR):
        return SentenceTransformer(MODEL_DIR, device='cpu')
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    model.save(MODEL_DIR)
    return model


def main():
    targets = [sys.argv[1]] if len(sys.argv) > 1 else list(DATASETS)
    model = get_model()
    for ds in targets:
        emb_dir = L.emb_dir(ds, bge=True)
        done_file = L.art_path(ds, f'{ds}_bge_qemb.npy')
        if not os.path.exists(done_file):
            encode_corpus(model, ds)
        evaluate(ds)
    print('DONE')


if __name__ == '__main__':
    main()
