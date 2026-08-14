# -*- coding: utf-8 -*-
"""Generation-side evaluation: does CA-HR context improve RAG answers?

Paired design: for the same 100 test queries (50 SCIDOCS + 50 SciFact,
seed-fixed), we build top-5 context with two retrieval backends
(Neural-Hybrid vs CA-HR), generate a cited answer with DeepSeek
(deepseek-chat), then score:
  - answer relevance (1-5, LLM judge)
  - faithfulness to context (1-5, LLM judge)
  - citation precision: fraction of [n] citations pointing at a
    qrels-relevant passage
Results: per-query JSONL checkpoint + summary with Wilcoxon signed-rank.

The DeepSeek key is read from the PaperPilot project .env at runtime;
it is never printed or copied.

Usage:  python gen_eval.py            # resume-safe; run until DONE
        python gen_eval.py summary    # only aggregate finished records
"""
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
import urllib.error

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(ROOT, os.pardir, 'exp_v2')
RES = os.path.join(ROOT, 'results')
CKPT = os.path.join(RES, 'gen_eval_ckpt.jsonl')
SUMMARY = os.path.join(RES, 'gen_eval_summary.json')
ENV_PATH = os.path.join(os.path.expanduser('~'), 'Desktop',
                        'paperpilot-src', 'paperpilot', '.env')

ALPHA, BETA, GAMMA = 0.6, 0.15, 0.10
TOPK = 100
SEED = 20260813
N_PER_DS = 50
TOP_CONTEXT = 5
API_URL = 'https://api.deepseek.com/v1/chat/completions'
MODEL = 'deepseek-chat'

DATASETS = ['scidocs', 'scifact', 'nfcorpus', 'trec-covid']
V3DATA = os.path.join(ROOT, 'data')
V3ART = os.path.join(ROOT, 'artifacts')


def ds_locations(ds):
    """Return (base_dir_with_beir_files, prep_dir, scoremats_npz)."""
    if ds in ('scidocs', 'scifact'):
        return (os.path.join(V2, ds), os.path.join(V2, f'{ds}_prep'),
                os.path.join(V2, f'{ds}_scoremats.npz'))
    return (os.path.join(V3DATA, ds), os.path.join(V3ART, f'{ds}_prep'),
            os.path.join(V3ART, f'{ds}_scoremats.npz'))

GEN_SYS = ('You are an academic writing assistant. Answer the question using '
           'ONLY the provided passages. Cite supporting passages inline as '
           '[1], [2], ... Be concise (<= 150 words).')
GEN_USER = ('Question: {q}\n\nPassages:\n{ctx}\n\nAnswer with citations:')

JUDGE_SYS = ('You are a strict evaluator of RAG answers. Return JSON only.')
JUDGE_USER = ('Question: {q}\n\nRetrieved passages:\n{ctx}\n\n'
              'Candidate answer:\n{ans}\n\n'
              'Rate on 1-5 integer scales:\n'
              '- relevance: does the answer address the question?\n'
              '- faithfulness: is every claim supported by the passages '
              '(no hallucination)?\n'
              'Return exactly: {{"relevance": X, "faithfulness": Y}}')


def load_key():
    local = os.path.join(ROOT, '.deepseek_key')
    if os.path.exists(local):
        return open(local, encoding='utf-8').read().strip()
    with open(ENV_PATH, encoding='utf-8') as f:
        for line in f:
            if line.startswith('DEEPSEEK_API_KEY='):
                return line.strip().split('=', 1)[1].strip().strip('"').strip("'")
    raise RuntimeError('DEEPSEEK_API_KEY not found in .env')


def api_call(key, sys_prompt, user_prompt, temperature, max_tokens=600):
    body = json.dumps({
        'model': MODEL,
        'messages': [{'role': 'system', 'content': sys_prompt},
                     {'role': 'user', 'content': user_prompt}],
        'temperature': temperature,
        'max_tokens': max_tokens,
    }).encode('utf-8')
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                API_URL, data=body,
                headers={'Content-Type': 'application/json',
                         'Authorization': 'Bearer ' + key})
            with urllib.request.urlopen(req, timeout=90) as r:
                out = json.loads(r.read().decode('utf-8'))
            return out['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 5:
                time.sleep(min(2 ** attempt * 3, 40))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < 5:
                time.sleep(min(2 ** attempt * 3, 40))
                continue
            raise


def minmax(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f]


def load_qrels(path):
    qrels = {}
    with open(path, encoding='utf-8') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) >= 3:
                qrels.setdefault(p[0], {})[p[1]] = int(float(p[2]))
    return qrels


def build_tasks():
    """Recompute top-5 rankings for both systems; sample 50 queries/dataset."""
    rng = random.Random(SEED)
    tasks = []
    for ds in DATASETS:
        base, prep, sm_path = ds_locations(ds)
        doc_ids = json.load(open(os.path.join(prep, 'doc_ids.json')))
        didx = {d: i for i, d in enumerate(doc_ids)}
        C = np.load(os.path.join(prep, 'C.npy'))
        Rr = np.load(os.path.join(prep, 'R.npy'))
        z = np.load(sm_path)
        S_bm, S_sb = z['S_bm'], z['S_sb']
        qrels = load_qrels(os.path.join(base, 'qrels', 'test.tsv'))
        qids = sorted(qrels.keys())
        assert len(qids) == S_bm.shape[0], f'{ds} qid/scoremat mismatch'
        queries = {str(q['_id']): q.get('text') or ''
                   for q in load_jsonl(os.path.join(base, 'queries.jsonl'))}
        corpus = {str(d['_id']): d for d in
                  load_jsonl(os.path.join(base, 'corpus.jsonl'))}
        n_take = min(N_PER_DS, len(qids))
        sample = rng.sample(qids, n_take)
        for q in sample:
            qi = qids.index(q)
            bm_n = minmax(S_bm[qi].astype(np.float64))
            sb_n = minmax(S_sb[qi].astype(np.float64))
            nh = 0.5 * bm_n + 0.5 * sb_n
            top_nh = np.argsort(-nh)[:TOP_CONTEXT]
            hyb = ALPHA * bm_n + (1 - ALPHA) * sb_n
            cand = np.argpartition(-hyb, TOPK)[:TOPK]
            sc = np.full_like(hyb, -1e18)
            sc[cand] = hyb[cand] + BETA * C[cand] + GAMMA * Rr[cand]
            top_ca = np.argsort(-sc)[:TOP_CONTEXT]
            for system, top in (('Neural-Hybrid', top_nh), ('CA-HR', top_ca)):
                docs = [doc_ids[i] for i in top]
                ctx = []
                for rank, did in enumerate(docs, 1):
                    d = corpus.get(did) or {}
                    text = ((d.get('title') or '') + '. ' +
                            (d.get('text') or ''))[:1200]
                    ctx.append(f'[{rank}] {text}')
                rel_docs = {d for d, s in qrels[q].items() if s >= 1}
                tasks.append({
                    'ds': ds, 'qid': q, 'question': queries.get(q, ''),
                    'system': system, 'doc_ids': docs,
                    'n_rel_context': sum(1 for d in docs if d in rel_docs),
                    'context': '\n\n'.join(ctx),
                })
    return tasks


def parse_judge(text):
    m = re.search(r'\{[^{}]*\}', text or '', re.S)
    if not m:
        return None, None
    try:
        j = json.loads(m.group(0))
        return int(j.get('relevance')), int(j.get('faithfulness'))
    except (ValueError, TypeError):
        return None, None


def citation_precision(answer, doc_ids, rel_docs_by_ds_qid, ds, qid):
    cited = sorted(set(int(n) for n in re.findall(r'\[(\d+)\]', answer or '')
                       if 1 <= int(n) <= len(doc_ids)))
    if not cited:
        return None, 0
    rel = rel_docs_by_ds_qid[(ds, qid)]
    good = sum(1 for n in cited if doc_ids[n - 1] in rel)
    return good / len(cited), len(cited)


def main():
    os.makedirs(RES, exist_ok=True)
    done = set()
    if os.path.exists(CKPT):
        with open(CKPT, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                done.add((r['ds'], r['qid'], r['system']))
    tasks = [t for t in build_tasks()
             if (t['ds'], t['qid'], t['system']) not in done]
    print(f'todo: {len(tasks)} generations', flush=True)
    if not tasks:
        return
    key = load_key()

    # qrels lookup for citation precision
    rel_docs = {}
    for ds in DATASETS:
        base, _, _ = ds_locations(ds)
        qr = load_qrels(os.path.join(base, 'qrels', 'test.tsv'))
        for q, dd in qr.items():
            rel_docs[(ds, q)] = {d for d, s in dd.items() if s >= 1}

    lock = threading.Lock()
    out_f = open(CKPT, 'a', encoding='utf-8')
    counter = {'n': 0}

    def worker():
        while True:
            with lock:
                if not tasks:
                    return
                t = tasks.pop()
            rec = dict(t)
            rec.pop('context', None)
            try:
                ans = api_call(key, GEN_SYS,
                               GEN_USER.format(q=t['question'], ctx=t['context']),
                               temperature=0.2)
                judge = api_call(key, JUDGE_SYS,
                                 JUDGE_USER.format(q=t['question'],
                                                   ctx=t['context'], ans=ans),
                                 temperature=0.0, max_tokens=60)
                rel, faith = parse_judge(judge)
                cp, n_cit = citation_precision(ans, t['doc_ids'], rel_docs,
                                               t['ds'], t['qid'])
                rec.update({'answer': ans, 'relevance': rel,
                            'faithfulness': faith, 'citation_precision': cp,
                            'n_citations': n_cit})
            except Exception as e:  # noqa: BLE001 - record failure, continue
                rec.update({'error': f'{type(e).__name__}: {e}'})
            with lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out_f.flush()
                counter['n'] += 1
                if counter['n'] % 10 == 0:
                    print(f'finished {counter["n"]}', flush=True)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    out_f.close()
    print('generation pass complete', flush=True)


def summarize():
    recs = [json.loads(l) for l in open(CKPT, encoding='utf-8')]
    ok = [r for r in recs if 'error' not in r and r.get('relevance')]
    from scipy import stats as sstats
    by_q = {}
    for r in ok:
        by_q.setdefault((r['ds'], r['qid']), {})[r['system']] = r
    paired = [v for v in by_q.values() if 'CA-HR' in v and 'Neural-Hybrid' in v]
    out = {'n_records': len(recs), 'n_scored': len(ok), 'n_paired': len(paired)}

    def agg(sysname, key):
        vals = [r[key] for r in ok
                if r['system'] == sysname and r.get(key) is not None]
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                'n': len(vals)} if vals else None

    for sysname in ('CA-HR', 'Neural-Hybrid'):
        out[sysname] = {
            'relevance': agg(sysname, 'relevance'),
            'faithfulness': agg(sysname, 'faithfulness'),
            'citation_precision': agg(sysname, 'citation_precision'),
            'rel_docs_in_context': agg(sysname, 'n_rel_context'),
            'n_citations': agg(sysname, 'n_citations'),
        }
    for metric in ('relevance', 'faithfulness', 'citation_precision',
                   'n_rel_context'):
        a = [p['CA-HR'].get(metric) for p in paired]
        b = [p['Neural-Hybrid'].get(metric) for p in paired]
        ab = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if len(ab) >= 10:
            x, y = np.array([v[0] for v in ab]), np.array([v[1] for v in ab])
            diff = x - y
            nz = diff != 0
            p = float(sstats.wilcoxon(x[nz], y[nz]).pvalue) if nz.sum() else 1.0
            out[f'paired_{metric}'] = {
                'n': len(ab), 'mean_CA-HR': float(x.mean()),
                'mean_Neural-Hybrid': float(y.mean()),
                'wilcoxon_p_two_sided': p}
    json.dump(out, open(SUMMARY, 'w'), indent=1)
    # per-dataset breakdown (the study's claim is four-domain)
    by_ds = {}
    for r in ok:
        by_ds.setdefault(r['ds'], []).append(r)
    for ds, recs_ds in by_ds.items():
        sub = {}
        for sysname in ('CA-HR', 'Neural-Hybrid'):
            rr = [r for r in recs_ds if r['system'] == sysname]
            rel = [r['relevance'] for r in rr if r.get('relevance')]
            fai = [r['faithfulness'] for r in rr if r.get('faithfulness')]
            cp = [r['citation_precision'] for r in rr
                  if r.get('citation_precision') is not None]
            sub[sysname] = {
                'n': len(rr),
                'relevance_mean': float(np.mean(rel)) if rel else None,
                'faithfulness_mean': float(np.mean(fai)) if fai else None,
                'citation_precision_mean': float(np.mean(cp)) if cp else None,
                'rel_docs_in_context_mean': float(np.mean(
                    [r['n_rel_context'] for r in rr])) if rr else None}
        out.setdefault('by_dataset', {})[ds] = sub
    json.dump(out, open(SUMMARY, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summarize()
    else:
        main()
        summarize()
