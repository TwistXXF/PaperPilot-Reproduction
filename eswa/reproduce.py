#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PaperPilot reproduction pipeline
================================
Reproduces every number, table, and figure in:
  "Adaptive Retrieval-Augmented Generation for Scientific Literature Analysis:
   A Cross-Domain Evaluation with Real Citation Metadata"

Usage:
    python reproduce.py <stage>

Stages (in pipeline order):
    download    Download SCIDOCS + SciFact raw data (skipped if data/ already present,
                as in the distributed repository)
    metadata    Fetch real citation metadata from the Semantic Scholar API
                (skipped if data/metadata/ already present)
    encode      Encode corpora and queries with all-MiniLM-L6-v2 (CPU)
    retrieval   BM25 / LSA / SBERT / hybrid / UMA-RAG / LP-RAG / CA-HR, per-query metrics
    tables      Aggregate tables + significance tests (Wilcoxon + Cohen's d)
    ablation    CA-HR component ablation, both datasets
    robust      Query word-drop noise robustness (10-40%)
    router      PAV-Agent 5-fold CV routing analysis
    figures     Regenerate Fig. 1-4
    all         Run everything (download .. figures)

Expected runtimes on a modern CPU-only workstation:
    encode ~30-60 min (dominated by the 25,657-document SCIDOCS corpus);
    everything else < 10 min total.
"""
import json, os, re, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
META_DIR = os.path.join(DATA, 'metadata')
ART = os.path.join(ROOT, 'artifacts')          # generated intermediates
RES = os.path.join(ROOT, 'results')            # final published results
FIG = os.path.join(ROOT, 'figures')
MODEL_DIR = os.path.join(ROOT, 'models', 'minilm')

# ---- hyperparameters (identical to the paper) ----
REF_YEAR = 2024
ALPHA, BETA, GAMMA = 0.6, 0.15, 0.10   # CA-HR fusion / citation / recency weights
DELTA, EPSILON = 0.10, 0.10            # UMA-RAG venue / citation weights
ETA, MU = 0.2, 5000.0                  # LP-RAG length-penalty strength / scale
TOPK = 100                             # CA-HR rerank candidate depth
LAMBDA_RECENCY = 0.1                   # per-year recency decay
RANDOM_STATE = 42
METHODS = ['BM25', 'LSA-Dense', 'SBERT-Dense', 'Neural-Hybrid', 'UMA-RAG', 'LP-RAG', 'CA-HR']
METRICS = ['R@1', 'R@5', 'R@10', 'N@10', 'MRR']

_ws = re.compile(r'\s+')
_punct = re.compile(r'[,\.\(\);:!\?\"\'\[\]\{\}<>\/\\\-]')


def tokenize(t):
    return _ws.split(_punct.sub(' ', t.lower()).strip())


def minmax(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_qrels(path):
    qrels = {}
    with open(path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 3:
                qrels.setdefault(parts[0], {})[parts[1]] = int(float(parts[2]))
    return qrels


def ds_paths(ds):
    base = os.path.join(DATA, ds)
    return (os.path.join(base, 'corpus.jsonl'),
            os.path.join(base, 'queries.jsonl'),
            os.path.join(base, 'qrels', 'test.tsv'))


# =====================================================================
# Stage: download
# =====================================================================
def stage_download():
    """Download raw BEIR-format data if not already present.
    SCIDOCS: mirrored from the mteb/scidocs HuggingFace dataset (BEIR format).
    SciFact: official BEIR distribution (TU Darmstadt).
    NFCorpus / TREC-COVID: official BEIR zips (TU Darmstadt)."""
    import urllib.request
    import zipfile
    targets = [
        ('https://huggingface.co/datasets/mteb/scidocs/resolve/main/corpus.jsonl',
         os.path.join(DATA, 'scidocs', 'corpus.jsonl')),
        ('https://huggingface.co/datasets/mteb/scidocs/resolve/main/queries.jsonl',
         os.path.join(DATA, 'scidocs', 'queries.jsonl')),
        ('https://huggingface.co/datasets/mteb/scidocs/resolve/main/qrels/test.tsv',
         os.path.join(DATA, 'scidocs', 'qrels', 'test.tsv')),
        ('https://huggingface.co/datasets/mteb/scifact/resolve/main/corpus.jsonl',
         os.path.join(DATA, 'scifact', 'corpus.jsonl')),
        ('https://huggingface.co/datasets/mteb/scifact/resolve/main/queries.jsonl',
         os.path.join(DATA, 'scifact', 'queries.jsonl')),
        ('https://huggingface.co/datasets/mteb/scifact/resolve/main/qrels/test.tsv',
         os.path.join(DATA, 'scifact', 'qrels', 'test.tsv')),
    ]
    for url, dst in targets:
        if os.path.exists(dst):
            print('present, skip:', dst)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print('downloading', url)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=600) as r, open(dst, 'wb') as f:
            f.write(r.read())
    # BEIR zip distributions for the two additional datasets
    beir = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/'
    for name in ('nfcorpus', 'trec-covid'):
        if os.path.exists(os.path.join(DATA, name, 'corpus.jsonl')):
            print('present, skip:', name)
            continue
        zp = os.path.join(DATA, f'{name}.zip')
        if not os.path.exists(zp):
            print('downloading', beir + name + '.zip')
            req = urllib.request.Request(beir + name + '.zip',
                                         headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3600) as r, \
                    open(zp, 'wb') as f:
                f.write(r.read())
        with zipfile.ZipFile(zp) as z:
            z.extractall(DATA)
        print('extracted', name)
    print('download stage complete')


# =====================================================================
# Stage: metadata (real Semantic Scholar values)
# =====================================================================
def stage_metadata():
    """Fetch citationCount / year / venue for every document from the Semantic
    Scholar batch API. SCIDOCS ids are S2 sha1 paperIds; SciFact ids are S2ORC
    CorpusIds. Checkpointed; respectful of the public rate limit."""
    import requests
    os.makedirs(META_DIR, exist_ok=True)
    for ds, prefix in [('scidocs', ''), ('scifact', 'CorpusId:')]:
        out_path = os.path.join(META_DIR, f'{ds}_metadata.json')
        corpus_path, _, _ = ds_paths(ds)
        ids = [str(d['_id']) for d in load_jsonl(corpus_path)]
        done = json.load(open(out_path, encoding='utf-8')) if os.path.exists(out_path) else {}
        todo = [i for i in ids if i not in done]
        print(ds, len(ids), 'docs,', len(done), 'cached,', len(todo), 'to fetch')
        if not todo:
            continue
        B = 400
        for s in range(0, len(todo), B):
            batch = todo[s:s + B]
            payload = [f'{prefix}{i}' for i in batch]
            for attempt in range(8):
                try:
                    r = requests.post('https://api.semanticscholar.org/graph/v1/paper/batch',
                                      params={'fields': 'citationCount,year,venue'},
                                      json={'ids': payload}, timeout=60)
                    if r.status_code == 200:
                        break
                    time.sleep(10 * (attempt + 1))
                except Exception:
                    time.sleep(10 * (attempt + 1))
            else:
                raise SystemExit('S2 batch failed permanently; rerun to resume')
            for orig, rec in zip(batch, r.json()):
                done[orig] = None if rec is None else {
                    'citations': rec.get('citationCount'),
                    'year': rec.get('year'),
                    'venue': rec.get('venue') or ''}
            json.dump(done, open(out_path, 'w', encoding='utf-8'))
            time.sleep(2.5)
        missing = sum(1 for v in done.values() if v is None)
        print(ds, 'metadata done:', len(done), 'records,', missing, 'unmatched')


# =====================================================================
# Stage: encode (real pretrained dense retriever)
# =====================================================================
def stage_encode(datasets=('scidocs', 'scifact')):
    """Encode all documents and queries with sentence-transformers
    all-MiniLM-L6-v2 (384-dim, L2-normalized). CPU, checkpointed chunks."""
    from sentence_transformers import SentenceTransformer
    if os.path.exists(MODEL_DIR):
        model = SentenceTransformer(MODEL_DIR, device='cpu')
    else:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
        model.save(MODEL_DIR)
    for ds in datasets:
        corpus_path, queries_path, _ = ds_paths(ds)
        docs = load_jsonl(corpus_path)
        out_dir = os.path.join(ART, f'{ds}_emb')
        os.makedirs(out_dir, exist_ok=True)
        # contiguous coverage already on disk (chunk size may vary)
        starts = sorted(int(f[6:-4]) for f in os.listdir(out_dir)
                        if f.startswith('chunk_') and f.endswith('.npy'))
        done = 0
        for s in starts:
            if s != done:
                break
            done += int(np.load(os.path.join(out_dir, f'chunk_{s}.npy'),
                                mmap_mode='r').shape[0])
        print(ds, 'already encoded:', done, '/', len(docs))
        SUB = 1000
        for s in range(done, len(docs), SUB):
            part = docs[s:s + SUB]
            texts = [(d.get('title') or '') + ' ' + (d.get('text') or '') for d in part]
            emb = model.encode(texts, batch_size=64, show_progress_bar=False,
                               normalize_embeddings=True).astype(np.float32)
            tmp = os.path.join(out_dir, f'chunk_{s}.tmp.npy')
            np.save(tmp, emb)
            os.replace(tmp, os.path.join(out_dir, f'chunk_{s}.npy'))
        ids = [str(d['_id']) for d in docs]
        json.dump(ids, open(os.path.join(out_dir, 'ids.json'), 'w'))
        qs = load_jsonl(queries_path)
        q_emb = model.encode([(q.get('text') or '') for q in qs], batch_size=64,
                             show_progress_bar=False,
                             normalize_embeddings=True).astype(np.float32)
        np.save(os.path.join(ART, f'{ds}_qemb.npy'), q_emb)
        json.dump([str(q['_id']) for q in qs], open(os.path.join(ART, f'{ds}_qids.json'), 'w'))
        print(ds, 'encoded:', len(ids), 'docs,', len(qs), 'queries')


# =====================================================================
# Shared: prep (BM25 + LSA + metadata tensors)
# =====================================================================
def prep(ds):
    import joblib
    import scipy.sparse as sp
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    out = os.path.join(ART, f'{ds}_prep')
    os.makedirs(out, exist_ok=True)
    corpus_path, _, _ = ds_paths(ds)
    docs = load_jsonl(corpus_path)
    doc_ids = [str(d['_id']) for d in docs]
    texts = [(d.get('title') or '') + ' ' + (d.get('text') or '') for d in docs]
    print(ds, 'docs:', len(docs), flush=True)

    # step 1: BM25 + doc lengths (checkpointed)
    bm25_fp = os.path.join(out, 'bm25.joblib')
    lens_fp = os.path.join(out, 'lens.npy')
    if os.path.exists(bm25_fp) and os.path.exists(lens_fp):
        bm25 = joblib.load(bm25_fp)
        lens = np.load(lens_fp)
        print(ds, 'bm25 loaded', flush=True)
    else:
        tok = [tokenize(t) for t in texts]
        bm25 = BM25Okapi(tok, k1=1.5, b=0.75, epsilon=0.25)
        lens = np.array([len(t) for t in tok], dtype=np.float64)
        joblib.dump(bm25, bm25_fp)
        np.save(lens_fp, lens)
        print(ds, 'bm25 done', flush=True)

    # step 2: TF-IDF matrix (checkpointed)
    X_fp = os.path.join(out, 'tfidf_X.npz')
    vec_fp = os.path.join(out, 'tfidf.joblib')
    if os.path.exists(X_fp) and os.path.exists(vec_fp):
        X = sp.load_npz(X_fp)
        vec = joblib.load(vec_fp)
        print(ds, 'tfidf loaded', flush=True)
    else:
        vec = TfidfVectorizer(max_features=50000, stop_words='english', min_df=2)
        X = vec.fit_transform(texts)
        sp.save_npz(X_fp, X)
        joblib.dump(vec, vec_fp)
        print(ds, 'tfidf done', flush=True)

    # step 3: SVD -> LSA (checkpointed)
    lsa_fp = os.path.join(out, 'lsa.npy')
    svd_fp = os.path.join(out, 'svd.joblib')
    if os.path.exists(lsa_fp) and os.path.exists(svd_fp):
        svd = joblib.load(svd_fp)
        print(ds, 'svd loaded', flush=True)
    else:
        svd = TruncatedSVD(n_components=384, random_state=RANDOM_STATE)
        L = svd.fit_transform(X).astype(np.float32)
        L /= (np.linalg.norm(L, axis=1, keepdims=True) + 1e-9)
        np.save(lsa_fp, L)
        joblib.dump(svd, svd_fp)
        print(ds, 'svd done', flush=True)
    del X

    # step 4: metadata tensors (checkpointed)
    C_fp = os.path.join(out, 'C.npy')
    if os.path.exists(C_fp) and os.path.exists(os.path.join(out, 'R.npy')) \
            and os.path.exists(os.path.join(out, 'V.npy')):
        print(ds, 'metadata tensors loaded', flush=True)
    else:
        meta = json.load(open(os.path.join(META_DIR, f'{ds}_metadata.json'), encoding='utf-8'))
        cits, years, venues = [], [], []
        for did in doc_ids:
            m = meta.get(did)
            if m is None:
                cits.append(0); years.append(None); venues.append('')
            else:
                cits.append(m.get('citations') or 0)
                years.append(m.get('year'))
                venues.append(m.get('venue') or '')
        med_year = int(np.median([y for y in years if y]))
        years = [y if y else med_year for y in years]
        C = np.log1p(np.array(cits, dtype=np.float64))
        C = C / (C.max() + 1e-9)
        R = np.exp(-LAMBDA_RECENCY * (REF_YEAR - np.array(years, dtype=np.float64)))
        from collections import defaultdict
        vs, vc = defaultdict(float), defaultdict(int)
        for c, v in zip(cits, venues):
            vs[v] += c; vc[v] += 1
        vmean = {v: vs[v] / vc[v] for v in vs}
        V = minmax(np.array([vmean[v] for v in venues], dtype=np.float64))
        np.save(C_fp, C)
        np.save(os.path.join(out, 'R.npy'), R)
        np.save(os.path.join(out, 'V.npy'), V)
        print(ds, 'metadata tensors done', flush=True)

    json.dump(doc_ids, open(os.path.join(out, 'doc_ids.json'), 'w'))
    return out


def corpus_embeddings(ds):
    d = os.path.join(ART, f'{ds}_emb')
    ids = json.load(open(os.path.join(d, 'ids.json')))
    chunks = []
    starts = sorted(int(f[6:-4]) for f in os.listdir(d) if f.startswith('chunk_'))
    for s in starts:
        chunks.append(np.load(os.path.join(d, f'chunk_{s}.npy')))
    E = np.vstack(chunks)
    assert len(E) == len(ids)
    return ids, E


def per_query_metrics(rank_idx, rel_list, gains_list, ks=(1, 5, 10)):
    res = {f'R@{k}': [] for k in ks}
    res['N@10'] = []
    res['MRR'] = []
    for qi in range(len(rel_list)):
        rel = rel_list[qi]
        gains = gains_list[qi]
        order = rank_idx[qi]
        n_rel = float(rel.sum())
        ranked_rel = rel[order]
        ranked_gain = gains[order]
        for k in ks:
            res[f'R@{k}'].append(ranked_rel[:k].sum() / n_rel if n_rel else 0.0)
        dcg = float(np.sum(ranked_gain[:10] / np.log2(np.arange(2, 12))))
        ideal = np.sort(gains)[::-1][:10]
        idcg = float(np.sum(ideal / np.log2(np.arange(2, 2 + len(ideal)))))
        res['N@10'].append(dcg / idcg if idcg > 0 else 0.0)
        rr = 0.0
        for rank, r in enumerate(ranked_rel[:100], start=1):
            if r:
                rr = 1.0 / rank
                break
        res['MRR'].append(rr)
    return res


def build_eval_arrays(qrels, qids, didx, ndocs):
    rel_list, gains_list = [], []
    for q in qids:
        rel = np.zeros(ndocs, dtype=np.float64)
        gains = np.zeros(ndocs, dtype=np.float64)
        for d, s in qrels[q].items():
            if d in didx:
                gains[didx[d]] = s
                if s >= 1:
                    rel[didx[d]] = 1
        rel_list.append(rel)
        gains_list.append(gains)
    return rel_list, gains_list


# =====================================================================
# Stage: retrieval (all seven methods, per-query metrics)
# =====================================================================
def stage_retrieval(datasets=('scidocs', 'scifact')):
    import joblib
    os.makedirs(RES, exist_ok=True)
    for ds in datasets:
        out = prep(ds)
        doc_ids = json.load(open(os.path.join(out, 'doc_ids.json')))
        didx = {d: i for i, d in enumerate(doc_ids)}
        L = np.load(os.path.join(out, 'lsa.npy'))
        C = np.load(os.path.join(out, 'C.npy'))
        Rr = np.load(os.path.join(out, 'R.npy'))
        V = np.load(os.path.join(out, 'V.npy'))
        lens = np.load(os.path.join(out, 'lens.npy'))
        bm25 = joblib.load(os.path.join(out, 'bm25.joblib'))
        vec = joblib.load(os.path.join(out, 'tfidf.joblib'))
        svd = joblib.load(os.path.join(out, 'svd.joblib'))
        emb_ids, E = corpus_embeddings(ds)
        assert emb_ids == doc_ids, 'embedding id mismatch'

        _, queries_path, qrels_path = ds_paths(ds)
        queries = {str(q['_id']): q.get('text') or '' for q in load_jsonl(queries_path)}
        qrels = load_qrels(qrels_path)
        qids = sorted(qrels.keys())
        print(ds, 'test queries:', len(qids))
        q_emb = np.load(os.path.join(ART, f'{ds}_qemb.npy'))
        q_emb_ids = json.load(open(os.path.join(ART, f'{ds}_qids.json')))
        qemb_map = {q: q_emb[i] for i, q in enumerate(q_emb_ids)}
        rel_list, gains_list = build_eval_arrays(qrels, qids, didx, len(doc_ids))

        qtexts = [queries[q] for q in qids]
        qtok = [tokenize(t) for t in qtexts]
        Qlsa = svd.transform(vec.transform(qtexts)).astype(np.float32)
        Qlsa /= (np.linalg.norm(Qlsa, axis=1, keepdims=True) + 1e-9)
        Qsbert = np.stack([qemb_map[q] for q in qids])

        # base score matrices (checkpointed)
        sm_path = os.path.join(ART, f'{ds}_scoremats.npz')
        if os.path.exists(sm_path):
            z = np.load(sm_path)
            S_bm, S_sb, S_lsa = z['S_bm'], z['S_sb'], z['S_lsa']
        else:
            t0 = time.time()
            S_bm = np.stack([np.asarray(bm25.get_scores(t), dtype=np.float32) for t in qtok])
            bm25_ms = (time.time() - t0) / len(qids) * 1000
            S_sb = (Qsbert @ E.T).astype(np.float32)
            S_lsa = (Qlsa @ L.T).astype(np.float32)
            np.savez_compressed(sm_path, S_bm=S_bm, S_sb=S_sb, S_lsa=S_lsa)
            print(ds, 'BM25 scoring: %.1f ms/query' % bm25_ms)

        all_metrics = {}
        for m in METHODS:
            orders = []
            for qi in range(len(qids)):
                s_bm = S_bm[qi].astype(np.float64)
                s_sb = S_sb[qi].astype(np.float64)
                s_lsa = S_lsa[qi].astype(np.float64)
                bm_n, sb_n = minmax(s_bm), minmax(s_sb)
                if m == 'BM25':
                    sc = s_bm
                elif m == 'LSA-Dense':
                    sc = s_lsa
                elif m == 'SBERT-Dense':
                    sc = s_sb
                elif m == 'Neural-Hybrid':
                    sc = 0.5 * bm_n + 0.5 * sb_n
                else:
                    hyb = ALPHA * bm_n + (1 - ALPHA) * sb_n
                    if m == 'UMA-RAG':
                        sc = hyb + DELTA * V + EPSILON * C
                    elif m == 'LP-RAG':
                        sc = hyb * (1 + ETA * np.exp(-lens / MU))
                    elif m == 'CA-HR':
                        cand = np.argpartition(-hyb, TOPK)[:TOPK]
                        sc = np.full_like(hyb, -1e18)
                        sc[cand] = hyb[cand] + BETA * C[cand] + GAMMA * Rr[cand]
                orders.append(np.argsort(-sc)[:100])
            all_metrics[m] = per_query_metrics(orders, rel_list, gains_list)
            print(ds, m, 'avg N@10 = %.4f' % float(np.mean(all_metrics[m]['N@10'])), flush=True)

        np.savez_compressed(os.path.join(RES, f'{ds}_perquery.npz'),
                            qids=np.array(qids),
                            **{f'{m}||{k}': np.array(v)
                               for m, mm in all_metrics.items() for k, v in mm.items()})
        print(ds, 'retrieval stage saved')


# =====================================================================
# Stage: tables + significance
# =====================================================================
def load_perquery(ds):
    z = np.load(os.path.join(RES, f'{ds}_perquery.npz'), allow_pickle=True)
    qids = z['qids'].tolist()
    d = {m: {k: z[f'{m}||{k}'] for k in METRICS} for m in METHODS}
    return qids, d


def cohend(a, b):
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 1e-12 else 0.0


def wilcoxon_greater(a, b):
    from scipy import stats as sstats
    diff = a - b
    if np.all(np.abs(diff) < 1e-15):
        return 1.0
    return float(sstats.wilcoxon(a, b, alternative='greater').pvalue)


def stage_tables(datasets=('scidocs', 'scifact')):
    out = {}
    for ds in datasets:
        qids, d = load_perquery(ds)
        avg = {m: {k: float(np.mean(v)) for k, v in d[m].items()} for m in METHODS}
        tests = {}
        for base in METHODS:
            if base == 'CA-HR':
                continue
            for k in METRICS:
                a, b = d['CA-HR'][k], d[base][k]
                tests[f'CA-HR vs {base} | {k}'] = {
                    'cahr': float(np.mean(a)), 'base': float(np.mean(b)),
                    'p_one_sided': wilcoxon_greater(a, b), 'd': cohend(a, b)}
        routed = ['UMA-RAG', 'LP-RAG', 'CA-HR']
        oracle = {k: float(np.mean(np.max(np.stack([d[m][k] for m in routed]), axis=0)))
                  for k in METRICS}
        out[ds] = {'n_queries': len(qids), 'avg': avg,
                   'oracle_routed': oracle, 'tests_vs_cahr': tests}
        print('===', ds, len(qids), 'queries')
        for m in METHODS:
            print('  %-14s R@10=%.4f N@10=%.4f MRR=%.4f' % (
                m, avg[m]['R@10'], avg[m]['N@10'], avg[m]['MRR']))
    json.dump(out, open(os.path.join(RES, 'tables.json'), 'w'), indent=1)


# =====================================================================
# Stage: ablation
# =====================================================================
def eval_orders(orders, qrels, qids, didx, ndocs):
    rel_list, gains_list = build_eval_arrays(qrels, qids, didx, ndocs)
    res = {'R@10': [], 'N@10': [], 'MRR': []}
    for qi in range(len(qids)):
        rel, gains = rel_list[qi], gains_list[qi]
        o = orders[qi]
        n = rel.sum()
        res['R@10'].append(float(rel[o][:10].sum() / n) if n else 0.0)
        g = gains[o][:10]
        dcg = float(np.sum(g / np.log2(np.arange(2, 2 + len(g)))))
        ideal = np.sort(gains)[::-1][:10]
        idcg = float(np.sum(ideal / np.log2(np.arange(2, 2 + len(ideal)))))
        res['N@10'].append(dcg / idcg if idcg > 0 else 0.0)
        rr = 0.0
        for rank, idx in enumerate(o[:100], start=1):
            if rel[idx]:
                rr = 1.0 / rank
                break
        res['MRR'].append(rr)
    return res


def load_common(ds):
    import joblib
    out = os.path.join(ART, f'{ds}_prep')
    doc_ids = json.load(open(os.path.join(out, 'doc_ids.json')))
    didx = {d: i for i, d in enumerate(doc_ids)}
    C = np.load(os.path.join(out, 'C.npy'))
    Rr = np.load(os.path.join(out, 'R.npy'))
    bm25 = joblib.load(os.path.join(out, 'bm25.joblib'))
    _, E = corpus_embeddings(ds)
    _, queries_path, qrels_path = ds_paths(ds)
    queries = {str(q['_id']): q.get('text') or '' for q in load_jsonl(queries_path)}
    qrels = load_qrels(qrels_path)
    qids = sorted(qrels.keys())
    q_emb = np.load(os.path.join(ART, f'{ds}_qemb.npy'))
    q_emb_ids = json.load(open(os.path.join(ART, f'{ds}_qids.json')))
    qemb_map = {q: q_emb[i] for i, q in enumerate(q_emb_ids)}
    return doc_ids, didx, bm25, E, C, Rr, queries, qrels, qids, qemb_map


def stage_ablation(datasets=('scidocs', 'scifact')):
    variants = {
        'full': dict(alpha=ALPHA, beta=BETA, gamma=GAMMA),
        '-citation': dict(alpha=ALPHA, beta=0.0, gamma=GAMMA),
        '-recency': dict(alpha=ALPHA, beta=BETA, gamma=0.0),
        '-dense (alpha=1)': dict(alpha=1.0, beta=BETA, gamma=GAMMA),
        '-sparse (alpha=0)': dict(alpha=0.0, beta=BETA, gamma=GAMMA),
        '-rerank (plain hybrid)': dict(alpha=ALPHA, beta=0.0, gamma=0.0),
    }
    for ds in datasets:
        (doc_ids, didx, bm25, E, C, Rr, queries, qrels, qids, qemb_map) = load_common(ds)
        sm_path = os.path.join(ART, f'{ds}_scoremats.npz')
        S_bm = S_sb = None
        if os.path.exists(sm_path):
            z = np.load(sm_path)
            S_bm, S_sb = z['S_bm'], z['S_sb']
        done = {}
        for name, prm in variants.items():
            orders = []
            for qi, q in enumerate(qids):
                if S_bm is not None:
                    s_bm = S_bm[qi].astype(np.float64)
                    s_sb = S_sb[qi].astype(np.float64)
                else:
                    s_bm = np.asarray(bm25.get_scores(tokenize(queries[q])), dtype=np.float64)
                    s_sb = E @ qemb_map[q]
                hyb = prm['alpha'] * minmax(s_bm) + (1 - prm['alpha']) * minmax(s_sb)
                cand = np.argpartition(-hyb, TOPK)[:TOPK]
                sc = np.full_like(hyb, -1e18)
                sc[cand] = hyb[cand] + prm['beta'] * C[cand] + prm['gamma'] * Rr[cand]
                orders.append(np.argsort(-sc)[:100])
            res = eval_orders(orders, qrels, qids, didx, len(doc_ids))
            done[name] = {k: float(np.mean(v)) for k, v in res.items()}
            print(ds, name, {k: round(v, 4) for k, v in done[name].items()}, flush=True)
        json.dump(done, open(os.path.join(RES, f'{ds}_ablation.json'), 'w'), indent=1)


# =====================================================================
# Stage: robustness
# =====================================================================
def stage_robust(datasets=('scidocs', 'scifact')):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_DIR, device='cpu')
    for ds in datasets:
        (doc_ids, didx, bm25, E, C, Rr, queries, qrels, qids, qemb_map) = load_common(ds)
        if ds == 'scidocs':
            # fixed 300-query subsample for the noise study (disclosed in paper)
            sub_rng = np.random.RandomState(7)
            qids = sorted(sub_rng.choice(qids, size=300, replace=False).tolist())
        rng = np.random.RandomState(RANDOM_STATE)
        done = {}
        for noise in [0.1, 0.2, 0.3, 0.4]:
            noisy = []
            for q in qids:
                toks = queries[q].split()
                keep = [t for t in toks if rng.rand() > noise] or toks[:1]
                noisy.append(' '.join(keep))
            Qn = model.encode(noisy, batch_size=64, show_progress_bar=False,
                              normalize_embeddings=True).astype(np.float32)
            row = {}
            # score once per query, reuse across methods (identical results,
            # 3x less BM25 work than scoring inside each method branch)
            per_q = []
            for qi, q in enumerate(qids):
                s_bm = np.asarray(bm25.get_scores(tokenize(noisy[qi])), dtype=np.float64)
                s_sb = E @ Qn[qi]
                per_q.append((s_bm, minmax(s_bm), minmax(s_sb)))
            for m in ['BM25', 'Neural-Hybrid', 'CA-HR']:
                orders = []
                for s_bm, bm_n, sb_n in per_q:
                    if m == 'BM25':
                        sc = s_bm
                    elif m == 'Neural-Hybrid':
                        sc = 0.5 * bm_n + 0.5 * sb_n
                    else:
                        hyb = ALPHA * bm_n + (1 - ALPHA) * sb_n
                        cand = np.argpartition(-hyb, TOPK)[:TOPK]
                        sc = np.full_like(hyb, -1e18)
                        sc[cand] = hyb[cand] + BETA * C[cand] + GAMMA * Rr[cand]
                    orders.append(np.argsort(-sc)[:100])
                res = eval_orders(orders, qrels, qids, didx, len(doc_ids))
                row[m] = {k: float(np.mean(v)) for k, v in res.items()}
            done[str(noise)] = row
            print(ds, 'noise', noise,
                  {k: round(v['N@10'], 4) for k, v in row.items()},
                  flush=True)
        json.dump(done, open(os.path.join(RES, f'{ds}_robust.json'), 'w'), indent=1)


# =====================================================================
# Stage: router (PAV-Agent)
# =====================================================================
def query_features(text):
    toks = text.split()
    n = len(toks)
    low = text.lower()
    uniq = len(set(t.lower() for t in toks))
    stops = sum(1 for t in toks if t.lower() in
                ('a', 'an', 'the', 'of', 'in', 'on', 'for', 'and', 'or', 'to', 'is', 'are',
                 'with', 'by'))
    return [
        n, len(text),
        float(np.mean([len(t) for t in toks])) if n else 0.0,
        sum(c.isdigit() for c in text),
        sum(1 for t in toks if t.isupper() and len(t) > 1),
        float(bool(re.search(r'\b(review|survey|overview|state.of.the.art|advances)\b', low))),
        float(bool(re.search(r'\b(how|what|which|why|compare|versus|vs)\b', low))),
        uniq / n if n else 0.0,
        float(bool(re.search(r'\b(19|20)\d{2}\b', text))),
        stops / n if n else 0.0,
        sum(1 for t in toks if t[:1].isupper()) / n if n else 0.0,
        sum(1 for t in toks if '-' in t),
    ]


def stage_router(datasets=('scidocs', 'scifact')):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
    routed = ['UMA-RAG', 'LP-RAG', 'CA-HR']
    for ds in datasets:
        qids, d = load_perquery(ds)
        _, queries_path, _ = ds_paths(ds)
        queries = {str(q['_id']): q.get('text') or '' for q in load_jsonl(queries_path)}
        X = np.array([query_features(queries[q]) for q in qids])
        N = np.stack([d[m]['N@10'] for m in routed])
        y = N.argmax(axis=0)
        for j in range(len(qids)):
            if d['CA-HR']['N@10'][j] >= N[:, j].max() - 1e-9:
                y[j] = 2
        counts = np.bincount(y, minlength=3)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        yhat = np.zeros(len(qids), dtype=int)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            yhat[te] = clf.predict(sc.transform(X[te]))
        sysres = {}
        for k in METRICS:
            M = np.stack([d[m][k] for m in routed])
            sysres[k] = float(np.mean(M[yhat, np.arange(len(qids))]))
        orc = {k: float(np.mean(np.max(np.stack([d[m][k] for m in routed]), axis=0)))
               for k in METRICS}
        best_single = max(routed, key=lambda m: np.mean(d[m]['N@10']))
        out = {'n': len(qids),
               'label_counts': {routed[i]: int(counts[i]) for i in range(3)},
               'cv_accuracy': float(accuracy_score(y, yhat)),
               'macro_f1': float(f1_score(y, yhat, average='macro')),
               'kappa': float(cohen_kappa_score(y, yhat)),
               'routed_system': sysres, 'oracle': orc,
               'best_single': best_single,
               'best_single_N@10': float(np.mean(d[best_single]['N@10']))}
        json.dump(out, open(os.path.join(RES, f'{ds}_router.json'), 'w'), indent=1)
        print(ds, 'router: acc=%.3f kappa=%.3f routed N@10=%.4f oracle=%.4f' % (
            out['cv_accuracy'], out['kappa'], sysres['N@10'], orc['N@10']))


# =====================================================================
# Stage: figures
# =====================================================================
def stage_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    tables = json.load(open(os.path.join(RES, 'tables.json')))
    COLORS = ['#8c8c8c', '#bdbdbd', '#2171b5', '#6baed6', '#74c476', '#fd8d3c', '#cb181d']

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, ds, title in zip(axes, ['scidocs', 'scifact'],
                             ['SCIDOCS (Computer Science)', 'SciFact (Biomedical)']):
        vals = [tables[ds]['avg'][m]['N@10'] for m in METHODS]
        bars = ax.bar(range(len(METHODS)), vals, color=COLORS)
        best = int(np.argmax(vals))
        bars[best].set_edgecolor('black'); bars[best].set_linewidth(1.6)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, rotation=30, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('NDCG@10' if ds == 'scidocs' else '')
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.01, f'{v:.3f}', ha='center', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'Fig1.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    abl_s = json.load(open(os.path.join(RES, 'scidocs_ablation.json')))
    abl_f = json.load(open(os.path.join(RES, 'scifact_ablation.json')))
    names = list(abl_s.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, abl, title in zip(axes, [abl_s, abl_f], ['SCIDOCS', 'SciFact']):
        vals = [abl[n]['N@10'] for n in names]
        ax.bar(range(len(names)), vals, color='#4292c6')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(['Full', '-Cite', '-Recency', '-Dense', '-Sparse', '-Rerank'],
                           rotation=25, ha='right', fontsize=8)
        ax.set_title(f'CA-HR Ablation — {title} (NDCG@10)', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(min(vals) * 0.97, max(vals) * 1.02)
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'Fig2.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    rob_s = json.load(open(os.path.join(RES, 'scidocs_robust.json')))
    rob_f = json.load(open(os.path.join(RES, 'scifact_robust.json')))
    levels = ['0.1', '0.2', '0.3', '0.4']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, rob, title in zip(axes, [rob_s, rob_f],
                              ['SCIDOCS (300-query subsample)', 'SciFact (300 test queries)']):
        for m, c in zip(['BM25', 'Neural-Hybrid', 'CA-HR'], ['#8c8c8c', '#6baed6', '#cb181d']):
            ax.plot([float(l) * 100 for l in levels], [rob[l][m] for l in levels],
                    marker='o', label=m, color=c)
        ax.set_xlabel('Query word-drop noise (%)')
        ax.set_ylabel('Recall@10')
        ax.set_title(f'Noise Robustness — {title}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'Fig3.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    rt_s = json.load(open(os.path.join(RES, 'scidocs_router.json')))
    rt_f = json.load(open(os.path.join(RES, 'scifact_router.json')))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, r, title in zip(axes, [rt_s, rt_f], ['SCIDOCS', 'SciFact']):
        cats = ['Best single\nstrategy', 'PAV-Agent\n(5-fold CV)', 'Oracle\nrouting']
        vals = [r['best_single_N@10'], r['routed_system']['N@10'], r['oracle']['N@10']]
        bars = ax.bar(cats, vals, color=['#6baed6', '#fd8d3c', '#238b45'])
        ax.set_title(f'Query-Level Routing Ceiling — {title} (NDCG@10)', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(min(vals) * 0.95, max(vals) * 1.05)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.4f}',
                    ha='center', va='bottom', fontsize=8)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'Fig4.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('figures saved to', FIG)


STAGES = {
    'download': stage_download,
    'metadata': stage_metadata,
    'encode': stage_encode,
    'retrieval': stage_retrieval,
    'tables': stage_tables,
    'ablation': stage_ablation,
    'robust': stage_robust,
    'router': stage_router,
    'figures': stage_figures,
}
ORDER = ['download', 'metadata', 'encode', 'retrieval', 'tables',
         'ablation', 'robust', 'router', 'figures']

if __name__ == '__main__':
    os.makedirs(ART, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage == 'all':
        for s in ORDER:
            print('\n######## STAGE:', s, '########')
            STAGES[s]()
    else:
        # optional extra arg: restrict to one dataset, e.g.
        #   python reproduce.py robust scidocs
        if len(sys.argv) > 2 and stage in ('encode', 'retrieval', 'tables',
                                           'ablation', 'robust', 'router'):
            STAGES[stage](datasets=(sys.argv[2],))
        else:
            STAGES[stage]()
    print('DONE')
