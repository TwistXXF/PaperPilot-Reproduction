# -*- coding: utf-8 -*-
"""Layout resolver: the same scripts run in two layouts.

1. Development layout (this machine):
       IEEE/exp_v3/            <- scripts here (ROOT)
       IEEE/exp_v2/            <- scidocs/scifact raw data, prep, scoremats,
                                  per-query results, metadata jsons
       IEEE/exp_v3/data/       <- nfcorpus/trec-covid raw data + metadata dir

2. Public repository layout (fresh clone of PaperPilot-Reproduction):
       repo/eswa/              <- scripts here (ROOT)
       repo/data/{ds}/         <- raw BEIR data (all four datasets)
       repo/data/metadata/     <- fetched citation metadata jsons
       repo/artifacts/         <- scidocs/scifact prep + scoremats (regenerated)
       repo/results/           <- scidocs/scifact per-query + ablation etc.
       repo/eswa/artifacts/    <- nfcorpus/trec-covid prep, BGE embeddings
       repo/eswa/results/      <- published ESWA result files

Detection: if <parent>/data/scidocs exists we are in the repository layout.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(ROOT)
REPO_LAYOUT = os.path.isdir(os.path.join(PARENT, 'data', 'scidocs'))
V2 = os.path.join(PARENT, 'exp_v2')          # development-layout anchor


def raw_ds(ds):
    """Raw BEIR dataset directory (corpus.jsonl / queries.jsonl / qrels)."""
    if ds in ('scidocs', 'scifact'):
        return os.path.join(PARENT, 'data', ds) if REPO_LAYOUT \
            else os.path.join(V2, ds)
    return os.path.join(PARENT, 'data', ds) if REPO_LAYOUT \
        else os.path.join(ROOT, 'data', ds)


def meta_file(ds):
    """Path of {ds}_metadata.json."""
    if REPO_LAYOUT:
        return os.path.join(PARENT, 'data', 'metadata', f'{ds}_metadata.json')
    if ds in ('scidocs', 'scifact'):
        return os.path.join(V2, f'{ds}_metadata.json')
    return os.path.join(ROOT, 'data', 'metadata', f'{ds}_metadata.json')


def prep_dir(ds):
    """Prep directory with doc_ids.json / C.npy / R.npy / bm25.joblib."""
    if ds in ('scidocs', 'scifact'):
        return os.path.join(PARENT, 'artifacts', f'{ds}_prep') if REPO_LAYOUT \
            else os.path.join(V2, f'{ds}_prep')
    return os.path.join(ROOT, 'artifacts', f'{ds}_prep')


def scoremats(ds):
    if ds in ('scidocs', 'scifact'):
        return os.path.join(PARENT, 'artifacts', f'{ds}_scoremats.npz') \
            if REPO_LAYOUT else os.path.join(V2, f'{ds}_scoremats.npz')
    return os.path.join(ROOT, 'artifacts', f'{ds}_scoremats.npz')


def v2_perquery(ds):
    """MiniLM-family per-query npz for scidocs/scifact."""
    return os.path.join(PARENT, 'results', f'{ds}_perquery.npz') \
        if REPO_LAYOUT else os.path.join(V2, f'{ds}_perquery.npz')


def v2_json(name):
    """exp_v2-era aggregate json (ablation / robust / router / latency /
    tables) for scidocs/scifact."""
    return os.path.join(PARENT, 'results', name) if REPO_LAYOUT \
        else os.path.join(V2, name)


def manuscript_docx():
    """Path of the manuscript under audit."""
    cand = [os.path.join(PARENT, 'ESWA_submission', '01_Manuscript_ESWA.docx'),
            os.path.join(PARENT, 'manuscript', '01_Manuscript_ESWA.docx')]
    for c in cand:
        if os.path.exists(c):
            return c
    return cand[0]


def art_base(ds):
    """Artifacts directory for a dataset (embeddings, query embeddings,
    score matrices)."""
    if ds in ('scidocs', 'scifact') and REPO_LAYOUT:
        return os.path.join(PARENT, 'artifacts')
    return os.path.join(ROOT, 'artifacts')


def art_path(ds, name):
    return os.path.join(art_base(ds), name)


def emb_dir(ds, bge=False):
    return art_path(ds, f'{ds}_{"bge_" if bge else ""}emb')
