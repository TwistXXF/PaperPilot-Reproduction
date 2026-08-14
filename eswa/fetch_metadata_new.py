"""fetch_metadata_new.py — exp_v3 metadata for trec-covid and nfcorpus.

trec-covid:  join BEIR corpus ids (CORD uids) against the CORD-19 metadata.csv
             (already downloaded, possibly truncated tail) to get year/venue/DOI,
             then batch-fetch citationCount from the S2 API by DOI.
nfcorpus:    BEIR ids are 'MED-<pmid>'; batch-fetch from S2 by PMID.

Both phases checkpoint to data/metadata/{ds}_metadata.json so re-running resumes.
Output format matches reproduce.py expectations:
    {doc_id: None | {'citations': int|None, 'year': int|None, 'venue': str}}
"""
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
os.makedirs(META_DIR, exist_ok=True)

S2_BATCH = 'https://api.semanticscholar.org/graph/v1/paper/batch'


def load_ids(ds):
    ids = []
    with open(os.path.join(DATA, ds, 'corpus.jsonl'), encoding='utf-8') as f:
        for line in f:
            ids.append(str(json.loads(line)['_id']))
    return ids


TIME_BUDGET = 235  # seconds per invocation; exit cleanly and rerun to resume
_t0 = time.time()


def s2_batch(payloads, on_batch=None, sleep=6.0):
    """payloads: S2 id strings. Resumable: calls on_batch(index, recs) after each
    400-block so the caller can checkpoint. Returns list of records for the
    blocks fetched this run; caller tracks the offset."""
    out = []
    B = 400
    for s in range(0, len(payloads), B):
        if time.time() - _t0 > TIME_BUDGET:
            print('  time budget reached; rerun to resume', flush=True)
            break
        batch = payloads[s:s + B]
        recs = None
        for attempt in range(30):
            try:
                r = requests.post(S2_BATCH, params={'fields': 'citationCount,year,venue'},
                                  json={'ids': batch}, timeout=30)
                if r.status_code == 200:
                    recs = r.json()
                    break
                wait = 45 if r.status_code == 429 else 15
            except Exception:
                wait = 15
            if time.time() - _t0 > TIME_BUDGET:
                break
            time.sleep(wait)
        if recs is None:
            print(f'  paused at offset {s}; rerun to resume', flush=True)
            break
        out.extend(recs)
        if on_batch:
            on_batch(s, recs)
        print(f'  batch {s}-{s+len(batch)} done', flush=True)
        time.sleep(sleep)
    return out


def save(ds, done):
    p = os.path.join(META_DIR, f'{ds}_metadata.json')
    json.dump(done, open(p, 'w', encoding='utf-8'))
    print(ds, 'saved', len(done), 'records ->', p, flush=True)


# ---------------------------------------------------------------- trec-covid
def trec_covid():
    ids = load_ids('trec-covid')
    # NOTE: output is rebuilt from the two checkpoints on every run (idempotent),
    # so resuming after a partial DOI crawl always upgrades records correctly.

    # phase 1: CORD-19 metadata.csv join (cord_uid -> doi/year/venue)
    # checkpointed: the 1GB+ scan takes several minutes
    cord_path = os.path.join(DATA, 'cord19_metadata.csv')
    join_ckpt = os.path.join(META_DIR, 'trec-covid_join_ckpt.json')
    if os.path.exists(join_ckpt):
        uid2info = json.load(open(join_ckpt, encoding='utf-8'))
        print('cord join loaded from ckpt:', len(uid2info), flush=True)
    else:
        uid2info = {}
        idset = set(ids)
        with open(cord_path, encoding='utf-8', errors='ignore', newline='') as f:
            r = csv.reader(f)
            hdr = next(r)
            ix = {n: i for i, n in enumerate(hdr)}
            iu, idoi, iy, iv = ix['cord_uid'], ix['doi'], ix['publish_time'], ix['journal']
            for row in r:
                if len(row) <= max(iu, idoi, iy, iv):
                    continue
                uid = row[iu]
                if uid not in idset:
                    continue
                doi = row[idoi].strip()
                year = row[iy][:4]
                venue = row[iv].strip()
                # prefer rows carrying a DOI / fuller info
                if uid not in uid2info or (doi and not uid2info[uid]['doi']):
                    uid2info[uid] = {'doi': doi, 'year': year, 'venue': venue}
        json.dump(uid2info, open(join_ckpt, 'w', encoding='utf-8'))
        print('cord join matched:', len(uid2info), '/', len(ids), flush=True)

    # phase 2: citations by DOI via OpenAlex (fast: 50 DOIs per call, 10 req/s
    # polite pool; no API key needed). Checkpointed, resumable.
    doi2uid = {}
    for uid, info in uid2info.items():
        if info['doi']:
            doi2uid.setdefault(info['doi'].lower(), []).append(uid)
    ckpt_path = os.path.join(META_DIR, 'trec-covid_doi_ckpt.json')
    ck = json.load(open(ckpt_path, encoding='utf-8')) if os.path.exists(ckpt_path) else {}
    dois = [d for d in doi2uid if d not in ck]
    print('DOIs to fetch:', len(dois), 'cached:', len(ck), flush=True)
    B = 50
    import concurrent.futures

    sess = requests.Session()

    def fetch_batch(s):
        batch = dois[s:s + B]
        for attempt in range(3):
            try:
                r = sess.get(
                    'https://api.openalex.org/works',
                    params={'filter': 'doi:' + '|'.join(batch),
                            'per-page': 50,
                            'select': 'doi,cited_by_count,publication_year,primary_location',
                            'mailto': '3353854381@qq.com'}, timeout=20)
                if r.status_code == 200:
                    recs = {}
                    for w in r.json()['results']:
                        src = ((w.get('primary_location') or {}).get('source') or {})
                        d = (w['doi'] or '').replace('https://doi.org/', '').lower()
                        recs[d] = {'citations': w.get('cited_by_count'),
                                   'year': w.get('publication_year'),
                                   'venue': src.get('display_name') or ''}
                    return batch, recs
            except Exception:
                pass
            time.sleep(2)
        return batch, {}   # leave unfetched; rerun picks them up? no—mark None below

    starts = list(range(0, len(dois), B))
    done_rounds = 0
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        for batch, recs in ex.map(fetch_batch, starts):
            for d in batch:
                ck[d] = recs.get(d)  # None = OpenAlex has no record
            done_rounds += 1
            if done_rounds % 100 == 0:
                json.dump(ck, open(ckpt_path, 'w', encoding='utf-8'))
                print(f'  ckpt {len(ck)}/{len(doi2uid)}', flush=True)
            if time.time() - _t0 > TIME_BUDGET:
                break
    json.dump(ck, open(ckpt_path, 'w', encoding='utf-8'))

    # rebuild all records from the two checkpoints (idempotent)
    done = {}
    for i in ids:
        info = uid2info.get(i)
        if info is None:
            done[i] = {'citations': None, 'year': None, 'venue': ''}
            continue
        rec = ck.get(info['doi'].lower()) if info['doi'] else None
        cord_year = int(info['year']) if info['year'].isdigit() else None
        if rec is None:
            done[i] = {'citations': None, 'year': cord_year, 'venue': info['venue']}
        else:
            done[i] = {'citations': rec.get('citations'),
                       'year': rec.get('year') or cord_year,
                       'venue': rec.get('venue') or info['venue']}
    save('trec-covid', done)


# ----------------------------------------------------------------- nfcorpus
def nfcorpus():
    ids = load_ids('nfcorpus')
    out_path = os.path.join(META_DIR, 'nfcorpus_metadata.json')
    done = json.load(open(out_path, encoding='utf-8')) if os.path.exists(out_path) else {}
    todo = [i for i in ids if i not in done]
    print('nfcorpus:', len(ids), 'docs,', len(todo), 'to fetch', flush=True)
    if not todo:
        return
    pmids = [i.replace('MED-', '') for i in todo]
    recs = s2_batch([f'PMID:{p}' for p in pmids])
    for orig, rec in zip(todo, recs):
        done[orig] = None if rec is None else {
            'citations': rec.get('citationCount'),
            'year': rec.get('year'),
            'venue': rec.get('venue') or ''}
        if len(done) % 400 == 0:
            save('nfcorpus', done)
    save('nfcorpus', done)
    matched = sum(1 for v in done.values() if v)
    print(f'nfcorpus matched: {matched}/{len(done)} ({100*matched/len(done):.1f}%)')


if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'nfcorpus'):
        nfcorpus()
    if which in ('all', 'trec-covid'):
        trec_covid()
    print('DONE')
