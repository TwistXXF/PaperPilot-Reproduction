"""boost_citations.py — second pass: fill citation counts for trec-covid docs
that had no DOI, using s2_id (S2 batch API) and pubmed_id (OpenAlex) from the
CORD-19 metadata.csv. Checkpointed and idempotent."""
import csv
import json
import os
import time

import requests

import _layout as L

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(L.PARENT, 'data') if L.REPO_LAYOUT \
    else os.path.join(ROOT, 'data')
META_DIR = os.path.join(DATA, 'metadata')
TIME_BUDGET = 235
_t0 = time.time()

meta_path = os.path.join(META_DIR, 'trec-covid_metadata.json')
meta = json.load(open(meta_path, encoding='utf-8'))
need = {i for i, v in meta.items() if v['citations'] is None}
print('docs still without citations:', len(need), flush=True)

# --- pass 0: collect s2_id / pubmed_id for those docs from cord csv
ckpt_ids = os.path.join(META_DIR, 'trec-covid_extids_ckpt.json')
if os.path.exists(ckpt_ids):
    ext = json.load(open(ckpt_ids, encoding='utf-8'))
    print('ext ids loaded from ckpt:', len(ext), flush=True)
else:
    ext = {}
    with open(os.path.join(DATA, 'cord19_metadata.csv'), encoding='utf-8',
              errors='ignore', newline='') as f:
        r = csv.reader(f)
        hdr = next(r)
        ix = {n: i for i, n in enumerate(hdr)}
        iu, is2, ipm = ix['cord_uid'], ix['s2_id'], ix['pubmed_id']
        for row in r:
            if len(row) <= max(iu, is2, ipm):
                continue
            uid = row[iu]
            if uid in need:
                cur = ext.get(uid, {'s2': '', 'pmid': ''})
                s2, pmid = row[is2].strip(), row[ipm].strip()
                if (s2 and not cur['s2']) or (pmid and not cur['pmid']):
                    ext[uid] = {'s2': s2 or cur['s2'], 'pmid': pmid or cur['pmid']}
    json.dump(ext, open(ckpt_ids, 'w', encoding='utf-8'))
    print('ext ids collected:', len(ext), flush=True)

# --- pass 1: S2 batch by paper sha (s2_id)
ckpt_s2 = os.path.join(META_DIR, 'trec-covid_s2cite_ckpt.json')
ck = json.load(open(ckpt_s2, encoding='utf-8')) if os.path.exists(ckpt_s2) else {}
todo = []  # S2 batch API persistently 429s from this IP; OpenAlex pmid path only
print('s2 lookups todo:', len(todo), flush=True)
B = 400
for s in range(0, len(todo), B):
    if time.time() - _t0 > TIME_BUDGET:
        print('budget; resume later', flush=True)
        break
    batch = todo[s:s + B]
    recs = None
    for attempt in range(20):
        try:
            r = requests.post('https://api.semanticscholar.org/graph/v1/paper/batch',
                              params={'fields': 'citationCount,year,venue'},
                              json={'ids': [ext[u]['s2'] for u in batch]}, timeout=30)
            if r.status_code == 200:
                recs = r.json()
                break
            time.sleep(45 if r.status_code == 429 else 10)
        except Exception:
            time.sleep(10)
        if time.time() - _t0 > TIME_BUDGET:
            break
    if recs is None:
        print('paused at', s, flush=True)
        break
    for u, rec in zip(batch, recs):
        ck[u] = None if rec is None else {'citations': rec.get('citationCount'),
                                          'year': rec.get('year'),
                                          'venue': rec.get('venue') or ''}
    json.dump(ck, open(ckpt_s2, 'w', encoding='utf-8'))
    print(f'  s2 ckpt {len(ck)}', flush=True)
    time.sleep(4)

# --- pass 2: OpenAlex by pubmed_id for the rest
ckpt_oa = os.path.join(META_DIR, 'trec-covid_pmid_ckpt.json')
oa = json.load(open(ckpt_oa, encoding='utf-8')) if os.path.exists(ckpt_oa) else {}
todo2 = [u for u, e in ext.items()
         if e['pmid'] and u not in oa and (u not in ck or ck[u] is None)]
print('pmid lookups todo:', len(todo2), flush=True)
sess = requests.Session()
B2 = 50
starts = list(range(0, len(todo2), B2))
import concurrent.futures

def fetch(s):
    batch = todo2[s:s + B2]
    pm = [ext[u]['pmid'] for u in batch]
    for _ in range(3):
        try:
            r = sess.get('https://api.openalex.org/works',
                         params={'filter': 'pmid:' + '|'.join(pm), 'per-page': 50,
                                 'select': 'ids,cited_by_count,publication_year,primary_location',
                                 'mailto': '3353854381@qq.com'}, timeout=20)
            if r.status_code == 200:
                m = {}
                for w in r.json()['results']:
                    pmid = (w['ids'].get('pmid') or '').rstrip('/').split('/')[-1]
                    src = ((w.get('primary_location') or {}).get('source') or {})
                    m[pmid] = {'citations': w.get('cited_by_count'),
                               'year': w.get('publication_year'),
                               'venue': src.get('display_name') or ''}
                return batch, m
        except Exception:
            pass
        time.sleep(2)
    return batch, {}

rounds = 0
with concurrent.futures.ThreadPoolExecutor(8) as ex:
    for batch, m in ex.map(fetch, starts):
        for u in batch:
            oa[u] = m.get(ext[u]['pmid'])
        rounds += 1
        if rounds % 100 == 0:
            json.dump(oa, open(ckpt_oa, 'w', encoding='utf-8'))
            print(f'  oa ckpt {len(oa)}', flush=True)
        if time.time() - _t0 > TIME_BUDGET:
            break
json.dump(oa, open(ckpt_oa, 'w', encoding='utf-8'))

# --- merge back (never downgrade an existing value)
for u in need:
    rec = ck.get(u) or oa.get(u)
    if rec:
        cur = meta[u]
        meta[u] = {'citations': rec.get('citations'),
                   'year': cur['year'] or rec.get('year'),
                   'venue': cur['venue'] or rec.get('venue') or ''}
json.dump(meta, open(meta_path, 'w', encoding='utf-8'))
cit = sum(1 for v in meta.values() if v['citations'] is not None)
print(f'citation coverage now: {cit}/{len(meta)} ({100*cit/len(meta):.1f}%)')
print('DONE')
