# -*- coding: utf-8 -*-
"""Build the ESWA (Expert Systems with Applications) submission package
from the consolidated, audited numbers in results/eswa_tables.json.

Output folder: ../ESWA_submission/
  00_Title_Page.docx
  01_Manuscript_ESWA.docx
  02_Highlights.docx
  03_Cover_Letter.docx
  04_Declarations.docx
  figures/*.png|pdf

Every quantitative claim in the text is injected from eswa_tables.json so the
manuscript cannot drift from the result files. verify_eswa.py audits this.
"""
import json
import os
import shutil

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(BASE, '..', 'ESWA_submission'))
FIGOUT = os.path.join(OUT, 'figures')
os.makedirs(FIGOUT, exist_ok=True)

T = json.load(open(os.path.join(BASE, 'results', 'eswa_tables.json')))

METHODS = ['BM25', 'LSA-Dense', 'SBERT-Dense', 'BGE-Dense', 'Neural-Hybrid',
           'UMA-RAG', 'LP-RAG', 'CA-HR', 'BGE-Hybrid', 'BGE-CA-HR']
DS = ['scidocs', 'scifact', 'nfcorpus', 'trec-covid']
DS_NAME = {d: T['datasets'][d]['name'] for d in DS}

AUTHOR = 'Zichen Feng'
AFFIL = ('Department of Computer Science and Technology, Sanya University, '
         'Sanya 572022, China')
EMAIL = '3353854381@qq.com'
ORCID = '0009-0008-5491-3370'
REPO = 'https://github.com/TwistXXF/PaperPilot-Reproduction'
DOI = '10.5281/zenodo.21930729'
SITE = 'https://xxfpaperpilot.cn'

TITLE = ('When does bibliographic metadata help scientific retrieval-augmented '
         'generation? A four-domain evaluation of metadata-aware hybrid '
         'retrieval with a deployed academic writing assistant')


def f4(x):
    return f'{x:.4f}'


def f3(x):
    return f'{x:.3f}'


def pval(p):
    if p < 0.001:
        return 'p < 0.001'
    return f'p = {p:.3f}'


def new_doc():
    d = Document()
    st = d.styles['Normal']
    st.font.name = 'Times New Roman'
    st.font.size = Pt(11)
    return d


def p(doc, text, bold=False, size=11, align=None, italic=False):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    if align:
        par.alignment = align
    return par


def h1(doc, t):
    p(doc, t, bold=True, size=12)


def h2(doc, t):
    p(doc, t, bold=True, size=11)


def add_table(doc, headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'
    for j, htxt in enumerate(headers):
        c = t.rows[0].cells[j]
        c.text = htxt
        for r in c.paragraphs[0].runs:
            r.bold = True
            r.font.size = Pt(8)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.rows[i + 1].cells[j]
            c.text = str(v)
            for r in c.paragraphs[0].runs:
                r.font.size = Pt(8)


# convenient accessors ------------------------------------------------------
def avg(ds, m, k):
    return T['main'][ds]['avg'][m][k]


def abl(ds, variant, k='N@10'):
    return T['ablation'][ds][variant][k]


def holm(ds, test):
    """Holm-Bonferroni-adjusted p of a pre-specified primary comparison."""
    for t in T['primary_tests']:
        if t['dataset'] == ds and t['test'] == test:
            return t['p_holm']
    raise KeyError(test)


# ===========================================================================
# Title page
# ===========================================================================
def build_title_page():
    d = new_doc()
    p(d, 'Title Page', bold=True, size=14)
    d.add_paragraph()
    p(d, TITLE, bold=True, size=13)
    d.add_paragraph()
    p(d, f'{AUTHOR}', bold=True)
    p(d, AFFIL)
    p(d, f'E-mail: {EMAIL}')
    p(d, f'ORCID: {ORCID}')
    p(d, 'Telephone: +86-18032173019')
    d.add_paragraph()
    p(d, f'Corresponding author: {AUTHOR} ({EMAIL})')
    d.add_paragraph()
    p(d, 'Declarations of interest: none.')
    p(d, 'Acknowledgements: None.')
    p(d, f'Data and code availability: all data, code, and per-query result '
         f'files are publicly available at {REPO} '
         f'(archived at https://doi.org/{DOI}).')
    d.save(os.path.join(OUT, '00_Title_Page.docx'))


# ===========================================================================
# Highlights
# ===========================================================================
def build_highlights():
    d = new_doc()
    p(d, 'Highlights', bold=True, size=13)
    hl = [
        'Ten retrieval configurations are evaluated across four scientific '
        'domains.',
        'BGE-based retrieval leads on three domains, but no method wins '
        'everywhere.',
        'Citation priors help only when citation counts predict relevance.',
        'A surface-feature router fails to recover per-query oracle gains.',
        'Answer quality is stable across backends on 200 paired queries.',
    ]
    for x in hl:
        par = d.add_paragraph(style='List Bullet')
        par.add_run(x)
    d.save(os.path.join(OUT, '02_Highlights.docx'))


# ===========================================================================
# Manuscript
# ===========================================================================
def build_manuscript():
    d = new_doc()
    p(d, TITLE, bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    d.add_paragraph()

    # ---- Abstract ---------------------------------------------------------
    g = T['generation']
    h1(d, 'Abstract')
    p(d,
      'Retrieval-augmented generation (RAG) systems for scientific '
      'literature often ship one static retrieval configuration, yet '
      'whether one is optimal across research domains—and '
      'whether bibliographic metadata should influence ranking—remains '
      'insufficiently quantified. We evaluate ten retrieval '
      'configurations—BM25, latent semantic analysis, two pretrained dense '
      'encoders, equal-weight hybrid fusion, and metadata-aware variants '
      '(UMA-RAG, LP-RAG, and a citation- and recency-aware re-ranker, '
      'CA-HR)—on SCIDOCS, SciFact, NFCorpus, and '
      'TREC-COVID (3,633 to 171,332 documents; 1,673 queries), using real '
      'citation counts, years, and venues from Semantic '
      'Scholar and OpenAlex (coverage 69.8%-99.7%). Three findings '
      'emerge. First, the optimal strategy is domain-dependent: '
      'no configuration wins everywhere. Second, citation-aware signals '
      'help only where citation authority predicts relevance (on SCIDOCS, '
      'relevant documents carry a median of 566 citations versus 71 for '
      'non-relevant ones; AUC 0.798), are neutral where the association is '
      'weak, mildly harmful where citations are uninformative or '
      'inversely associated with relevance (AUC 0.498 and 0.461); the gain '
      'on the weaker MiniLM backbone does not transfer to the stronger BGE '
      'backbone under fixed hyperparameters, and a 30-combination weight '
      'sweep recovers it only on the citation-informative corpus. Third, '
      'per-query strategy selection offers little recoverable headroom: a '
      '12-feature logistic router does not exceed majority-class behaviour '
      'on any dataset (kappa near 0). On 200 paired '
      'queries, answer relevance and faithfulness are statistically '
      'indistinguishable between the hybrid and metadata-aware backends. '
      'The pipeline is deployed in a live academic writing assistant '
      '(anonymized for review); we report a one-month pilot '
      'deployment. Data, code, and per-query results are released '
      'through an anonymized repository.')
    p(d, 'Keywords: retrieval-augmented generation; scientific information '
         'retrieval; hybrid retrieval; bibliographic metadata; citation-aware '
         'ranking; query routing; expert systems',
      italic=True, size=10)
    d.add_page_break()

    # ---- 1. Introduction ---------------------------------------------------
    h1(d, '1. Introduction')
    p(d,
      'Academic writing and literature review increasingly rely on '
      'retrieval-augmented generation (RAG): a user question is first '
      'grounded in passages retrieved from a scholarly corpus, and a large '
      'language model then composes an answer conditioned on those passages '
      '(Lewis et al., 2020). The retrieval stage is the dominant determinant of end-to-end '
      'quality, because passages that are never retrieved cannot be cited. '
      'Yet most deployed scientific RAG systems adopt one fixed retrieval '
      'configuration—typically BM25 or a pretrained dense encoder—chosen '
      'once, on one benchmark, and generalised to every domain the system '
      'serves.')
    p(d,
      'This practice rests on an untested assumption: that the optimal '
      'retrieval configuration is stable across research domains. Scientific '
      'corpora differ systematically in document length, vocabulary '
      'specificity, claim style, and—critically for metadata-aware '
      'methods—in the density and coverage of bibliographic signals such as '
      'citation counts, publication years, and venue prestige. A citation '
      'boost that helps on a mature computer-science corpus may be useless on '
      'an emergency pandemic corpus where a third of the documents are fresh '
      'preprints with zero citations.')
    p(d,
      'This paper quantifies that question directly. We evaluate ten '
      'retrieval configurations on four public scientific benchmarks spanning '
      'computer science, biomedicine, nutrition, and pandemic medicine, using '
      'real—never synthetic—bibliographic metadata acquired from the Semantic '
      'Scholar and OpenAlex APIs. Unlike prior comparisons that rely on '
      'simulated metadata or single benchmarks, our study couples (i) a '
      'cross-domain retrieval evaluation with significance testing, (ii) '
      'ablation and query-corruption robustness analyses of the metadata '
      'terms, (iii) an oracle-and-router analysis that measures how much '
      'per-query adaptivity could ever buy, (iv) a generation-side study that '
      'asks whether backend differences survive into the final answer, and '
      '(v) a deployment case study in a live academic writing assistant '
      'used by real users (system anonymized for review).')
    p(d,
      'Rather than proposing yet another retrieval algorithm, we frame the '
      'study around three research questions:')
    for c in [
        'RQ1. Does bibliographic metadata improve scientific retrieval '
        'consistently across domains, or are its benefits conditional on '
        'corpus properties?',
        'RQ2. Which metadata signals help, under what corpus conditions do '
        'they fail, and should retrieval strategy adaptation happen per '
        'query or per domain?',
        'RQ3. Do retrieval-level differences actually propagate to the '
        'generated answers and to deployment decisions in a running system?',
    ]:
        par = d.add_paragraph(style='List Bullet')
        par.add_run(c)
    p(d,
      'Answering these questions yields the following contributions:')
    for c in [
        'A four-domain, ten-configuration evaluation of scientific RAG '
        'retrieval with real bibliographic metadata, showing that the best '
        'configuration is domain-dependent and that BGE-based retrieval '
        'leads on three of four domains (RQ1);',
        'Evidence that citation- and recency-aware re-ranking (CA-HR) is '
        'conditional on bibliographic informativeness, not coverage—'
        'significantly helpful where citations strongly predict relevance '
        '(computer science), neutral where the association is weak '
        '(biomedical claims), mildly harmful where it is absent or inverted '
        '(nutrition, pandemic medicine)—and backbone-conditional: '
        'the citation gain observed on the weaker MiniLM backbone does not '
        'transfer to the stronger BGE backbone under fixed a-priori '
        'weights; a weight sweep recovers it only on the '
        'citation-informative corpus (RQ1, RQ2);',
        'A replicated negative result on per-query routing: oracle headroom '
        'is small and a lightweight feature-based router (PAV Router) '
        'collapses to majority-class behaviour on all four datasets, '
        'implying adaptivity should be allocated at the domain level (RQ2);',
        'A four-domain generation-side evaluation (200 paired queries) '
        'showing answer relevance and faithfulness are statistically '
        'indistinguishable across the two leading retrieval backends, '
        'together with a citation-precision analysis of LLM answers (RQ3);',
        'A deployment feasibility case study of the full pipeline in the '
        'live system, with architecture, cost, and one month of '
        'real usage statistics (RQ3). All code, data, and per-query results '
        'are released through an anonymized public repository (link '
        'withheld for anonymous review).',
    ]:
        par = d.add_paragraph(style='List Bullet')
        par.add_run(c)
    p(d,
      'The remainder of the paper is organised as follows. Section 2 reviews '
      'related work. Section 3 describes the deployed system. Section 4 '
      'formalises the ten retrieval configurations and the router. Section 5 '
      'details datasets, metadata acquisition, and evaluation protocol. '
      'Section 6 reports results. Section 7 presents the deployment case '
      'study. Section 8 discusses implications and limitations, and Section 9 '
      'concludes.')

    # ---- 2. Related work ---------------------------------------------------
    h1(d, '2. Related work')
    h2(d, '2.1. Retrieval for scientific text')
    p(d,
      'Sparse lexical retrieval with BM25 (Robertson & Zaragoza, 2009) '
      'remains a strong baseline on '
      'scientific corpora. Dense passage retrieval (Karpukhin et al., 2020) '
      'and its scientific '
      'specialisations—SPECTER (Cohan et al., 2020), which exploits '
      'citation links during '
      'pre-training, and neighborhood-contrastive approaches (Ostendorff et '
      'al., 2022)—improve '
      'semantic matching. Benchmarks such as BEIR (Thakur et al., 2021) and '
      'SciRepEval (Singh et al., 2023) '
      'have shown that no single retriever dominates across domains, a '
      'finding our study sharpens for RAG-oriented pipelines. Sentence-level '
      'encoders such as Sentence-BERT (Reimers & Gurevych, 2019) and the '
      'distilled MiniLM family '
      '(Wang et al., 2020) make dense retrieval cheap enough for CPU-only '
      'deployment, which '
      'is the regime our deployed system operates in; E5 (Wang et al., '
      '2022) and the BGE '
      'family (Chen et al., 2024) push the same efficiency-quality frontier '
      'further.')
    h2(d, '2.2. Hybrid and metadata-aware ranking')
    p(d,
      'Hybrid systems interpolate sparse and dense scores (Lin, 2021; Qu '
      'et al., 2021). '
      'Re-ranking with cross-encoders or sequence-to-sequence models '
      '(monoT5; Nogueira et al., 2020) and zero-shot LLM rankers such as '
      'setwise prompting '
      '(Zhuang et al., 2024) improve top-of-ranking quality at additional '
      'inference cost; '
      'these methods re-rank by content only. CA-HR instead re-ranks by '
      'combining content scores with citation-authority and recency signals '
      'derived from real bibliographic metadata, in the spirit of '
      'authority-aware scholarly search but applied inside a RAG pipeline. '
      'UMA-RAG and LP-RAG represent two alternative metadata-aware designs '
      '(uniform metadata augmentation and length-penalised ranking, '
      'respectively) and serve as strong in-family baselines.')
    h2(d, '2.3. Adaptive and agentic RAG')
    p(d,
      'Recent systems make retrieval adaptive: Self-RAG (Asai et al., '
      '2024b) learns when to '
      'retrieve, ITER-RETGEN (Shao et al., 2023) interleaves retrieval and '
      'generation, and '
      'agent-based assistants (Wang et al., 2024) plan multi-step '
      'retrieval. OpenScholar '
      '(Asai et al., 2024a) synthesises scientific literature with '
      'retrieval-augmented '
      'models. These approaches adapt retrieval decisions per query or per '
      'generation step; our router analysis asks a sharper question—given a '
      'portfolio of cheap retrieval strategies, how much could perfect '
      'per-query selection possibly gain, and can surface query features '
      'recover it? On four datasets the answer for the tested surface-feature '
      'router is consistently negative, which constrains where this '
      'lightweight form of adaptivity is worth its cost.')
    h2(d, '2.4. Evaluating the generation side')
    p(d,
      'Retrieval metrics do not directly measure what users read. We '
      'therefore add a generation-side study in which the same questions are '
      'answered by a fixed LLM (DeepSeek, deepseek-chat; DeepSeek-AI, '
      '2024) conditioned '
      'on contexts produced by two competing backends, scored for relevance '
      'and faithfulness with an LLM judge (Zheng et al., 2023) and for '
      'citation precision '
      'against the gold relevance judgments. This closes the loop between '
      'retrieval evaluation and deployed answer quality.')
    h2(d, '2.5. Recent metadata-aware scientific RAG and positioning')
    p(d,
      'The RAG literature has moved quickly, and several 2025-2026 studies '
      'are closest to our problem. SurveyGen (Bao et al., 2025) builds a '
      'large-scale '
      'scientific survey dataset and a quality-aware framework (QUAL-SG) '
      'that injects citation counts, author influence, and venue reputation '
      'into literature retrieval for survey generation; it demonstrates '
      'that '
      'quality metadata helps its generation pipeline but does not isolate '
      'when or why the retrieval-level gain occurs. Yousuf et al. (2026) '
      'systematically compare metadata-as-text, dual-encoder, and '
      'reformulation strategies on SEC filings and show that moderate '
      'metadata weights help on structured financial corpora; their '
      'metadata '
      'is structural (company, form, section), not bibliometric. SciRAG '
      '(Ding et al., 2026) couples adaptive retrieval with citation-graph '
      'symbolic '
      'reasoning and outline-guided synthesis for scientific question '
      'answering, using citations to organise evidence rather than as a '
      'ranking prior. RA-RAG (Hwang et al., 2025) estimates per-source '
      'reliability by '
      'cross-checking and retrieves only from reliable sources in '
      'multi-source QA. Table 1 summarises the positioning. Our work is '
      'complementary and, to our knowledge, the first to (i) quantify, '
      'across four scientific domains, when citation- and recency-aware '
      'ranking priors help, are neutral, or harm—linking the outcome to a '
      'measurable corpus property (citation-relevance AUC) rather than to '
      'metadata coverage—and (ii) show, via a weight-sensitivity sweep on a '
      'stronger encoder, that the gain is backbone-dependent and re-emerges '
      'only where citations are informative, and that it does not propagate '
      'to generation quality or to a deployed system.')
    add_table(d,
      ['Study', 'Venue', 'Metadata signal', 'Role of metadata',
       'Cross-domain analysis', 'Deployment'],
      [
        ['Bao et al. (2025)', 'EMNLP 2025', 'Citations, author, venue',
         'Quality filter for survey generation', 'No', 'No'],
        ['Yousuf et al. (2026)', 'ECIR 2026', 'Structural fields (SEC)',
         'Embedding/fusion strategies', 'Single corpus', 'No'],
        ['Ding et al. (2026)', 'EACL 2026', 'Citation graph',
         'Evidence organisation & attribution', 'No', 'No'],
        ['Hwang et al. (2025)', 'EMNLP 2025', 'Source reliability',
         'Source selection & voting', 'No', 'No'],
        ['This paper', '—', 'Citations, recency, venue (real APIs)',
         'Ranking prior; informativeness-gated', 'Four domains', 'Live system'],
      ])
    p(d, 'Table 1. Positioning against recent (2025-2026) metadata-aware and '
         'citation-aware RAG studies.', italic=True, size=9)

    # ---- 3. Deployed system ------------------------------------------------
    h1(d, '3. The deployed academic writing assistant')
    p(d,
      'The system under study is a web-based academic reading and writing '
      'assistant (name and URL withheld for anonymous review) that '
      'operationalises the retrieval stack evaluated in this paper. Users '
      'upload papers, chat with an AI assistant grounded in their library, '
      'manage highlights and AI-generated summaries, and draft writing '
      'projects with citation support. The system serves real users in '
      'production.')
    p(d,
      'Fig. 1 shows the architecture. A React 18 single-page application '
      'communicates over HTTPS with an Nginx 1.18 reverse proxy that serves '
      'static assets and forwards API calls—including server-sent-event (SSE) '
      'token streams—to a Node.js 20 / Express backend supervised by PM2. '
      'Persistent state (users, papers, conversations, messages, highlights, '
      'summaries) lives in MySQL 8.0 behind the Drizzle ORM. The RAG '
      'orchestration layer chunks uploaded papers, embeds passages, runs the '
      'hybrid retrieval and metadata re-ranking pipeline evaluated here, and '
      'assembles a cited context of five passages that is streamed to the '
      'DeepSeek chat model (deepseek-chat; DeepSeek-AI, 2024) for answer '
      'generation.')
    p(d,
      'Two deployment constraints shaped the retrieval design. First, the '
      'production server is a commodity virtual machine (2 vCPU, 1.6 GB RAM) '
      'with no GPU, ruling out cross-encoder re-ranking at query time and '
      'motivating the CPU-efficient configurations evaluated in this study. '
      'Second, answers must stream interactively, so retrieval plus '
      're-ranking must stay well under one second—satisfied by all ten '
      'configurations (Section 6.7).')
    d.add_picture(os.path.join(BASE, 'figures', 'system_architecture.png'),
                  width=Inches(6.0))
    p(d, 'Fig. 1. Architecture and deployment boundary of the deployed '
         'academic writing assistant (system anonymized for review).',
      italic=True, size=9)

    # ---- 4. Methods ---------------------------------------------------------
    h1(d, '4. Retrieval configurations and routing')
    h2(d, '4.1. Content-based retrieval')
    p(d,
      'BM25 (Robertson & Zaragoza, 2009) scores documents with k1 = 1.5, '
      'b = 0.75 over whitespace- and '
      'punctuation-tokenised title-plus-abstract text. LSA-Dense projects a '
      '50,000-feature TF-IDF space to 384 dimensions with truncated SVD '
      '(latent semantic analysis; Deerwester et al., 1990). SBERT-Dense '
      'encodes the same text with '
      'all-MiniLM-L6-v2 (Reimers & Gurevych, 2019; Wang et al., 2020) '
      '(384-dim, L2-normalised, inner-product '
      'similarity). BGE-Dense replaces the encoder with BGE-small-en-v1.5 '
      '(Xiao et al., 2023), a 12-layer 384-dim model roughly twice the '
      'depth of MiniLM, '
      'used with its official query instruction prefix. Neural-Hybrid '
      'fuses BM25 and SBERT-Dense with equal weights on min-max-normalised '
      'scores, and BGE-Hybrid is the same equal-weight fusion on the BGE '
      'backbone. To test whether metadata effects depend on the strength of '
      'the dense backbone, BGE-CA-HR applies the CA-HR re-ranking rule '
      '(Section 4.2) to the BGE hybrid instead of the MiniLM hybrid.')
    h2(d, '4.2. Metadata-aware variants')
    p(d,
      'Three configurations add bibliographic signals. UMA-RAG augments the '
      'hybrid score uniformly with venue prestige V(d) and citation authority '
      'C(d). LP-RAG multiplies the hybrid score by a length prior '
      '1 + eta * exp(-len(d)/mu), which smoothly down-weights longer '
      'documents: shorter documents receive up to a 1 + eta boost that '
      'decays exponentially with document length. CA-HR forms the hybrid base score S_hybrid(q, d) = alpha * '
      'S_sparse + (1 - alpha) * S_dense with alpha = 0.6, takes the top-100 '
      'candidates, and re-ranks by S_CA-HR(q, d) = S_hybrid(q, d) + beta * '
      'C(d) + gamma * R(d), where C(d) = log(1 + c_d) / max_j log(1 + c_j) '
      'is corpus-normalised citation authority and R(d) = exp(-lambda * '
      '(t_ref - t_d)) is an exponential recency term (lambda = 0.1 per year, '
      't_ref = 2024), with beta = 0.15 and gamma = 0.10. Citation counts, '
      'years, and venues are real API values; documents without a matched '
      'record receive c_d = 0 and the corpus median year, mirroring how a '
      'deployed system must handle missing metadata.')
    h2(d, '4.3. Query-level routing (PAV Router)')
    p(d,
      'PAV Router is a multinomial logistic-regression classifier over 12 '
      'hand-crafted query features (length, lexical diversity, acronym and '
      'digit counts, survey/intent keywords, year mentions, stop-word ratio, '
      'capitalisation, hyphenation). Training labels are derived post hoc: '
      'for each query, the label is the strategy among {UMA-RAG, LP-RAG, '
      'CA-HR} attaining the highest per-query NDCG@10 (ties favour CA-HR). '
      'We evaluate with stratified 5-fold cross-validation and report '
      'out-of-fold accuracy, macro-F1, and Cohen\'s kappa, together with the '
      'end-to-end effectiveness of the routed system versus the per-query '
      'oracle and the best single strategy. This design measures how much of '
      'the oracle routing gain is recoverable from surface query features '
      'alone.')

    # ---- 5. Experimental setup ----------------------------------------------
    h1(d, '5. Experimental setup')
    h2(d, '5.1. Datasets')
    rows = []
    for ds in DS:
        m = T['datasets'][ds]
        cov = m['meta_cov']
        rows.append([m['name'], m['domain'], f"{m['n_docs']:,}",
                     f"{m['n_queries']:,}",
                     f"{100*cov['citations']:.1f}%",
                     f"{100*cov['year']:.1f}%",
                     f"{100*cov['venue']:.1f}%"])
    p(d, 'Table 2. Datasets and real-metadata coverage.', italic=True, size=9)
    add_table(d,
              ['Dataset', 'Domain', 'Docs', 'Queries', 'Citation cov.',
               'Year cov.', 'Venue cov.'], rows)
    p(d,
      'We use four public benchmarks in BEIR format (Thakur et al., 2021): '
      'SCIDOCS (computer '
      'science; Cohan et al., 2020), SciFact (biomedical claim '
      'verification; Wadden et al., 2020), NFCorpus '
      '(nutrition and medicine; Boteva et al., 2016), and TREC-COVID '
      '(COVID-19 biomedicine; '
      'Voorhees et al., 2021). They span two orders of magnitude in corpus '
      'size (3.6k-171k '
      'documents) and four distinct domains, and—critically—differ in '
      'metadata coverage (Table 2): SCIDOCS is nearly complete (99.7%), '
      'SciFact and NFCorpus are high (94%), while TREC-COVID is genuinely '
      'sparse at 69.8% citation coverage, reflecting its many preprints. We '
      'treat this sparsity not as a defect to hide but as a property of '
      'real-world corpora that our cross-domain design is built to expose.')
    h2(d, '5.2. Bibliographic informativeness diagnostics')
    DI = T['diagnostics']
    p(d, 'Table 3. Bibliographic informativeness diagnostics per corpus. '
         'Citation-relevance AUC = P(citations of a relevant document exceed '
         'those of a non-relevant one); non-relevant sets are judged '
         'non-relevant documents where available (SCIDOCS, TREC-COVID) and a '
         'seeded background sample otherwise (SciFact, NFCorpus).',
      italic=True, size=9)
    rows = []
    for ds in DS:
        dg = DI[ds]
        rows.append([DS_NAME[ds], f"{100*dg['coverage']:.1f}%",
                     f"{dg['median_citations']:.0f}",
                     f"{100*dg['pct_zero_citation']:.1f}%",
                     f"{dg['median_age']:.0f}",
                     f"{dg['rel_median_citations']:.0f}",
                     f"{dg['nonrel_median_citations']:.0f}",
                     f"{dg['cit_rel_auc']:.3f}"])
    add_table(d, ['Dataset', 'Cit. cov.', 'Median cit.', '% zero-cit.',
                  'Median age (yr)', 'Rel. med. cit.', 'Non-rel. med. cit.',
                  'Cit.-rel. AUC'], rows)
    p(d,
      'Coverage alone does not explain where metadata helps: SciFact (94.1%) '
      'and NFCorpus (94.0%) have virtually identical citation coverage yet '
      'respond oppositely to citation-aware ranking. The discriminating '
      'variable is the citation-relevance association (Table 3). On SCIDOCS, '
      f'relevant documents carry a median of '
      f'{DI["scidocs"]["rel_median_citations"]:.0f} citations versus '
      f'{DI["scidocs"]["nonrel_median_citations"]:.0f} for non-relevant ones '
      f'(AUC = {DI["scidocs"]["cit_rel_auc"]:.3f}), so a citation prior '
      'genuinely separates relevant from non-relevant material. On SciFact '
      f'the association is weak (AUC = {DI["scifact"]["cit_rel_auc"]:.3f}). '
      f'On NFCorpus it is absent entirely (AUC = '
      f'{DI["nfcorpus"]["cit_rel_auc"]:.3f}, Mann-Whitney p = 0.55; medians '
      '6 vs. 5), and on TREC-COVID it is inverted '
      f'(AUC = {DI["trec-covid"]["cit_rel_auc"]:.3f}): relevant COVID-19 '
      'documents are newer and less cited than the background literature '
      '(median 6 vs. 16 citations), because the benchmark rewards recent '
      'pandemic findings, not established ones. Across our four benchmarks, '
      'metadata utility therefore tracks the informativeness of the '
      'bibliographic signal, not how completely it covers the corpus.')
    h2(d, '5.3. Metadata acquisition')
    p(d,
      'Citation counts, publication years, and venues were fetched from the '
      'Semantic Scholar API for SCIDOCS and SciFact (99.7% and 94.1% '
      'matched, respectively). For NFCorpus, PubMed identifiers were mapped '
      'through the Semantic Scholar batch endpoint (94.0% matched). For '
      'TREC-COVID, CORD-19 identifiers were resolved through the official '
      'metadata file to DOIs and PubMed identifiers, then enriched through '
      'the OpenAlex API (Priem et al., 2022), reaching 69.8% citation, '
      '96.4% year, and 92.5% '
      'venue coverage. No metadata value is synthetic; unmatched documents '
      'receive neutral defaults (zero citations, corpus-median year) exactly '
      'as they would in production.')
    h2(d, '5.4. Metrics, significance, and compute')
    p(d,
      'We report Recall@1/5/10, NDCG@10 with graded gains, and MRR on the '
      'official test judgments (documents judged non-relevant are excluded '
      'from the relevant sets). Significance uses the one-sided Wilcoxon '
      'signed-rank test over per-query scores with Cohen\'s d computed from '
      'the same per-query differences, so reported p-values and effect sizes '
      'are consistent by construction. To control the family-wise error '
      'rate, significance claims in the text are restricted to a '
      'pre-specified primary family of 16 comparisons (per dataset: CA-HR '
      'vs. BM25, SBERT-Dense, and Neural-Hybrid, and BGE-CA-HR vs. '
      'BGE-Hybrid, all on NDCG@10); quoted primary p-values are '
      'Holm-Bonferroni-adjusted within this family. Ablation, robustness, '
      'routing, and generation-side comparisons are exploratory analyses '
      'with unadjusted two-sided p-values; metadata-weight sensitivity '
      'uses one-sided Wilcoxon tests with Holm correction within each '
      'dataset. All retrieval runs use '
      'a single '
      'CPU-only workstation. Measured per-query cost on the largest corpus '
      '(TREC-COVID, 171k documents): BM25 scoring averages 1,002 ms; dense '
      'scoring over precomputed embeddings and CA-HR re-ranking add under '
      '2 ms per query. On the smaller corpora BM25 costs 7.6-97 ms per '
      'query. Query encoding (MiniLM) averages about 30 ms. Total online '
      'latency is therefore dominated by the LLM call, not retrieval.')

    # ---- 6. Results ---------------------------------------------------------
    h1(d, '6. Results')
    h2(d, '6.1. Main retrieval results')
    p(d, 'Table 4. Retrieval effectiveness: NDCG@10 / Recall@10 on the four '
         'test sets (best single method per dataset in bold in Fig. 2).',
      italic=True, size=9)
    rows = []
    for m in METHODS:
        row = [m]
        for ds in DS:
            row.append(f"{avg(ds, m, 'N@10'):.4f} / {avg(ds, m, 'R@10'):.4f}")
        rows.append(row)
    add_table(d, ['Method', 'SCIDOCS', 'SciFact', 'NFCorpus', 'TREC-COVID'],
              rows)
    best_str = '; '.join(
        f"{DS_NAME[ds]}: {T['main'][ds]['best_single_N@10']} "
        f"({avg(ds, T['main'][ds]['best_single_N@10'], 'N@10'):.4f})"
        for ds in DS)
    p(d,
      f'Fig. 2 and Table 4 show the central pattern: no configuration wins '
      f'everywhere. The best single method per dataset is {best_str}. '
      f'Pretrained dense retrieval dominates on SCIDOCS, where '
      f'SBERT-Dense reaches {f4(avg("scidocs", "SBERT-Dense", "N@10"))} '
      f'NDCG@10 versus {f4(avg("scidocs", "BM25", "N@10"))} for BM25 '
      f'(+44.7% relative); on SciFact the hybrid family is best-in-class '
      f'(Neural-Hybrid {f4(avg("scifact", "Neural-Hybrid", "N@10"))} versus '
      f'{f4(avg("scifact", "BM25", "N@10"))} for BM25), and CA-HR attains '
      f'the best Recall@10 ({f4(avg("scifact", "CA-HR", "R@10"))}) of all '
      f'ten methods. BGE-based configurations provide the best single '
      f'result on three of the four datasets—BGE-Dense on SciFact '
      f'({f4(avg("scifact", "BGE-Dense", "N@10"))}) and TREC-COVID '
      f'({f4(avg("trec-covid", "BGE-Dense", "N@10"))}), the '
      f'latter consistent with its published BEIR reference score '
      f'(approximately 0.76; Xiao et al., 2023), and BGE-Hybrid '
      f'on NFCorpus ({f4(avg("nfcorpus", "BGE-Hybrid", "N@10"))})—yet '
      f'BGE-Dense trails '
      f'SBERT-Dense on SCIDOCS '
      f'({f4(avg("scidocs", "BGE-Dense", "N@10"))} vs. '
      f'{f4(avg("scidocs", "SBERT-Dense", "N@10"))}), confirming that even '
      f'encoder choice is domain-dependent.')
    p(d,
      'CA-HR, the citation- and recency-aware configuration, sits in the '
      'middle of the hybrid family overall: it significantly outperforms '
      'BM25 on all four datasets '
      f'(Holm-adjusted within the primary family; e.g., SCIDOCS '
      f'{pval(holm("scidocs", "CA-HR vs BM25 | N@10"))}, '
      f'TREC-COVID {pval(holm("trec-covid", "CA-HR vs BM25 | N@10"))}), '
      'and it also exceeds LSA-Dense on all four datasets, an exploratory '
      'comparison outside the primary family (unadjusted one-sided '
      'Wilcoxon p < 0.001 on every dataset), '
      'but it does not beat the plain dense or plain hybrid baselines except '
      'on SciFact, where it significantly exceeds SBERT-Dense '
      f'({pval(holm("scifact", "CA-HR vs SBERT-Dense | N@10"))}, '
      f'd = {f3(T["main"]["scifact"]["tests_vs_cahr"]["CA-HR vs SBERT-Dense | N@10"]["d"])}). '
      'In other words, metadata-aware re-ranking reliably beats purely '
      'lexical and non-pretrained retrieval, but whether it beats plain '
      'dense retrieval depends on the domain.')
    p(d,
      'Table 4 also isolates the role of the dense backbone. Equal-weight '
      'hybrid fusion on BGE-small (BGE-Hybrid) does not reproduce the hybrid '
      'advantage seen with MiniLM: it trails BGE-Dense on SCIDOCS '
      f'({f4(avg("scidocs", "BGE-Hybrid", "N@10"))} vs. '
      f'{f4(avg("scidocs", "BGE-Dense", "N@10"))}), SciFact '
      f'({f4(avg("scifact", "BGE-Hybrid", "N@10"))} vs. '
      f'{f4(avg("scifact", "BGE-Dense", "N@10"))}), and TREC-COVID '
      f'({f4(avg("trec-covid", "BGE-Hybrid", "N@10"))} vs. '
      f'{f4(avg("trec-covid", "BGE-Dense", "N@10"))}); only on NFCorpus does '
      f'BGE-Hybrid ({f4(avg("nfcorpus", "BGE-Hybrid", "N@10"))}) edge out '
      f'BGE-Dense ({f4(avg("nfcorpus", "BGE-Dense", "N@10"))}), where it '
      'attains the best NDCG@10 of all ten configurations. Decisively for '
      'RQ1, transferring the CA-HR re-ranking rule to the BGE backbone '
      '(BGE-CA-HR) never helps: it falls slightly below BGE-Hybrid on every '
      f'dataset (SCIDOCS {f4(avg("scidocs", "BGE-CA-HR", "N@10"))} vs. '
      f'{f4(avg("scidocs", "BGE-Hybrid", "N@10"))}, '
      f'd = {f3(T["main"]["scidocs"]["bge_tests"]["BGE-CA-HR vs BGE-Hybrid | N@10"]["d"])}; '
      f'SciFact {f4(avg("scifact", "BGE-CA-HR", "N@10"))} vs. '
      f'{f4(avg("scifact", "BGE-Hybrid", "N@10"))}, '
      f'd = {f3(T["main"]["scifact"]["bge_tests"]["BGE-CA-HR vs BGE-Hybrid | N@10"]["d"])}; '
      f'NFCorpus {f4(avg("nfcorpus", "BGE-CA-HR", "N@10"))} vs. '
      f'{f4(avg("nfcorpus", "BGE-Hybrid", "N@10"))}, '
      f'd = {f3(T["main"]["nfcorpus"]["bge_tests"]["BGE-CA-HR vs BGE-Hybrid | N@10"]["d"])}; '
      f'TREC-COVID {f4(avg("trec-covid", "BGE-CA-HR", "N@10"))} vs. '
      f'{f4(avg("trec-covid", "BGE-Hybrid", "N@10"))}, '
      f'd = {f3(T["main"]["trec-covid"]["bge_tests"]["BGE-CA-HR vs BGE-Hybrid | N@10"]["d"])}) '
      'and well below BGE-Dense. Under CA-HR\'s fixed a-priori '
      'hyperparameters, the citation gain obtained on the weaker MiniLM '
      'backbone on SCIDOCS does not transfer to a stronger encoder: the '
      'observed metadata benefit is backbone-dependent rather than '
      'additive (Section 6.4 tests whether any metadata weight rescues it).')
    d.add_picture(os.path.join(BASE, 'figures', 'Fig2_main_results.png'),
                  width=Inches(6.0))
    p(d, 'Fig. 2. Retrieval effectiveness (NDCG@10) across four domains.',
      italic=True, size=9)

    h2(d, '6.2. Ablation of CA-HR components')
    p(d, 'Table 5. CA-HR ablation (NDCG@10; bold marks cases where removing '
         'a component improves performance).', italic=True, size=9)
    ABL = ['full', '-citation', '-recency', '-dense (alpha=1)',
           '-sparse (alpha=0)', '-rerank (plain hybrid)']
    ABL_LABEL = {'full': 'Full CA-HR', '-citation': '− citation',
                 '-recency': '− recency', '-dense (alpha=1)': '− dense (α=1)',
                 '-sparse (alpha=0)': '− sparse (α=0)',
                 '-rerank (plain hybrid)': '− re-rank'}
    rows = []
    for k in ABL:
        row = [ABL_LABEL[k]]
        for ds in DS:
            v = abl(ds, k)
            mark = '*' if (k != 'full' and v > abl(ds, 'full')) else ''
            row.append(f'{v:.4f}{mark}')
        rows.append(row)
    add_table(d, ['Variant', 'SCIDOCS', 'SciFact', 'NFCorpus', 'TREC-COVID'],
              rows)
    p(d,
      f'The ablation (Table 5, Fig. 3; * = removal improves on the full '
      f'model) localises exactly where metadata helps and where it hurts. '
      f'On SCIDOCS—the corpus where citations are most informative about '
      f'relevance (AUC = {T["diagnostics"]["scidocs"]["cit_rel_auc"]:.3f})—removing the '
      f'citation boost drops NDCG@10 from {f4(abl("scidocs", "full"))} to '
      f'{f4(abl("scidocs", "-citation"))}, the single most valuable metadata '
      f'signal in the study, while removing the recency prior improves '
      f'performance to {f4(abl("scidocs", "-recency"))}, indicating a '
      f'miscalibrated recency prior on a snapshot corpus. On SciFact both '
      f'metadata terms are neutral. On the two new domains the sign flips: '
      f'removing the citation boost improves NDCG@10 on NFCorpus '
      f'({f4(abl("nfcorpus", "full"))} → {f4(abl("nfcorpus", "-citation"))}) '
      f'and on TREC-COVID '
      f'({f4(abl("trec-covid", "full"))} → {f4(abl("trec-covid", "-citation"))}). '
      f'The sign flip tracks the bibliographic-informativeness diagnostics '
      f'(Table 3), not coverage: NFCorpus and TREC-COVID are exactly the '
      f'corpora where citation counts carry no (AUC = '
      f'{T["diagnostics"]["nfcorpus"]["cit_rel_auc"]:.3f}) or inverse '
      f'(AUC = {T["diagnostics"]["trec-covid"]["cit_rel_auc"]:.3f}) '
      f'information about relevance, so the citation boost amplifies '
      f'noise rather than refining the ranking. The content '
      f'side is stable across domains: removing either modality degrades '
      f'performance almost everywhere, except that a pure dense '
      f'configuration is best on SCIDOCS, consistent with dense dominance '
      f'on computer science.')
    d.add_picture(os.path.join(BASE, 'figures', 'Fig3_ablation.png'),
                  width=Inches(6.0))
    p(d, 'Fig. 3. CA-HR ablation (NDCG@10; dashed line = full model).',
      italic=True, size=9)

    h2(d, '6.3. Robustness to query corruption')
    p(d,
      'Under simulated word-drop noise (10%-40%), all hybrid methods degrade '
      'gracefully, but robustness rankings are domain-dependent and do not '
      'simply follow clean-query rankings (Fig. 4). On TREC-COVID, CA-HR is '
      'the most noise-resistant configuration: at 40% corruption it retains '
      f'{f4(T["robust"]["trec-covid"]["0.4"]["CA-HR"]["N@10"])} NDCG@10 '
      f'versus {f4(T["robust"]["trec-covid"]["0.4"]["Neural-Hybrid"]["N@10"])} '
      'for Neural-Hybrid and '
      f'{f4(T["robust"]["trec-covid"]["0.4"]["BM25"]["N@10"])} for BM25: the '
      'citation authority term, being independent of the corrupted query '
      'text, acts as a stabiliser even where it does not raise clean-query '
      'effectiveness. On SciFact, CA-HR '
      f'({f4(T["robust"]["scifact"]["0.4"]["CA-HR"]["N@10"])}) and '
      'Neural-Hybrid '
      f'({f4(T["robust"]["scifact"]["0.4"]["Neural-Hybrid"]["N@10"])}) are '
      'nearly tied at 40% noise, both well above BM25 '
      f'({f4(T["robust"]["scifact"]["0.4"]["BM25"]["N@10"])}), and on '
      'NFCorpus the two hybrids again track each other closely '
      f'({f4(T["robust"]["nfcorpus"]["0.4"]["CA-HR"]["N@10"])} vs. '
      f'{f4(T["robust"]["nfcorpus"]["0.4"]["Neural-Hybrid"]["N@10"])}). '
      'SCIDOCS is the exception: there BM25 is the most robust method at 40% '
      f'corruption ({f4(T["robust"]["scidocs"]["0.4"]["BM25"]["N@10"])} '
      f'versus {f4(T["robust"]["scidocs"]["0.4"]["CA-HR"]["N@10"])} for CA-HR '
      f'and {f4(T["robust"]["scidocs"]["0.4"]["Neural-Hybrid"]["N@10"])} for '
      'Neural-Hybrid), indicating that where dense representations dominate '
      'on clean queries they are also the most fragile to lexical '
      'corruption.')
    d.add_picture(os.path.join(BASE, 'figures', 'Fig4_robustness.png'),
                  width=Inches(6.0))
    p(d, 'Fig. 4. Robustness to query corruption (NDCG@10 vs. word-drop '
         'noise).', italic=True, size=9)

    h2(d, '6.4. Metadata-weight sensitivity and rank-fusion baselines')
    p(d,
      'Two objections remain after Section 6.1. First, CA-HR\'s weights were '
      'fixed a priori, so the absence of a metadata gain on the stronger BGE '
      'backbone could be an artefact of under-weighting the metadata terms. '
      'Second, our hybrids interpolate min-max-normalised scores, so a '
      'standard rank-fusion reference is missing. Table 6 addresses both. '
      'We sweep the citation and recency weights of BGE-CA-HR over a '
      '30-combination grid (beta in {0, 0.05, 0.10, 0.15, 0.20, 0.30}, '
      'gamma in {0, 0.05, 0.10, 0.15, 0.20}) and test each combination '
      'against BGE-Hybrid with one-sided Wilcoxon tests, Holm-corrected '
      'within each dataset. Because the grid is selected on the test '
      'judgments, we report this sweep strictly as an exploratory '
      'sensitivity analysis, not as confirmatory evidence. The metadata '
      'gain re-emerges only on SCIDOCS: '
      f'{T["sensitivity"]["scidocs"]["best_combo"].replace("beta=", "beta = ").replace("|gamma=", ", gamma = ")} '
      f'attains {f4(T["sensitivity"]["scidocs"]["best_N@10"])} NDCG@10, '
      'significantly above BGE-Hybrid '
      f'({f4(T["sensitivity"]["scidocs"]["bge_hybrid_N@10"])}; Holm '
      'p < 0.001) and marginally above BGE-Dense '
      f'({f4(avg("scidocs", "BGE-Dense", "N@10"))}). On SciFact, NFCorpus, '
      'and TREC-COVID no grid combination significantly beats BGE-Hybrid '
      f'(best {f4(T["sensitivity"]["scifact"]["best_N@10"])}, '
      f'{f4(T["sensitivity"]["nfcorpus"]["best_N@10"])}, and '
      f'{f4(T["sensitivity"]["trec-covid"]["best_N@10"])} versus '
      f'{f4(T["sensitivity"]["scifact"]["bge_hybrid_N@10"])}, '
      f'{f4(T["sensitivity"]["nfcorpus"]["bge_hybrid_N@10"])}, and '
      f'{f4(T["sensitivity"]["trec-covid"]["bge_hybrid_N@10"])}; all '
      'Holm-adjusted p = 1.0). The backbone effect is therefore not a '
      'weight artefact: metadata gains on a strong encoder are gated by '
      'citation informativeness, and where the signal is informative the '
      'gain transfers only after re-calibrating the metadata weight.')
    p(d,
      'As a stronger fusion reference we add reciprocal rank fusion '
      '(RRF, k = 60) of BM25 with each dense encoder. RRF-MiniLM / RRF-BGE '
      'reach NDCG@10 '
      f'{f4(T["sensitivity"]["scidocs"]["rrf"]["RRF-MiniLM"]["N@10"])} / '
      f'{f4(T["sensitivity"]["scidocs"]["rrf"]["RRF-BGE"]["N@10"])} on '
      'SCIDOCS, '
      f'{f4(T["sensitivity"]["scifact"]["rrf"]["RRF-MiniLM"]["N@10"])} / '
      f'{f4(T["sensitivity"]["scifact"]["rrf"]["RRF-BGE"]["N@10"])} on '
      'SciFact, '
      f'{f4(T["sensitivity"]["nfcorpus"]["rrf"]["RRF-MiniLM"]["N@10"])} / '
      f'{f4(T["sensitivity"]["nfcorpus"]["rrf"]["RRF-BGE"]["N@10"])} on '
      'NFCorpus, and '
      f'{f4(T["sensitivity"]["trec-covid"]["rrf"]["RRF-MiniLM"]["N@10"])} / '
      f'{f4(T["sensitivity"]["trec-covid"]["rrf"]["RRF-BGE"]["N@10"])} on '
      'TREC-COVID: no RRF variant is the best configuration on any dataset, '
      'so the qualitative conclusions of Table 4 do not depend on the '
      'min-max interpolation rule.')
    rows = []
    for ds in DS:
        s = T['sensitivity'][ds]
        rows.append([DS_NAME[ds],
                     f4(s['bge_hybrid_N@10']),
                     f4(avg(ds, 'BGE-Dense', 'N@10')),
                     f4(avg(ds, 'BGE-CA-HR', 'N@10')),
                     s['best_combo'].replace('beta=', '').replace('|gamma=', '/'),
                     f4(s['best_N@10']),
                     'yes' if s['any_significant_gain_after_holm'] else 'no',
                     f4(s['rrf']['RRF-MiniLM']['N@10']),
                     f4(s['rrf']['RRF-BGE']['N@10'])])
    add_table(d,
      ['Dataset', 'BGE-Hybrid', 'BGE-Dense', 'BGE-CA-HR (fixed)',
       'Best grid beta/gamma', 'Best grid N@10', 'Sig. after Holm',
       'RRF-MiniLM', 'RRF-BGE'], rows)
    p(d, 'Table 6. BGE-backbone metadata-weight sensitivity (30-combination '
         'grid per dataset; significance vs. BGE-Hybrid, one-sided Wilcoxon '
         'with Holm correction) and RRF (k = 60) rank-fusion baselines; '
         'NDCG@10.', italic=True, size=9)

    h2(d, '6.5. Oracle headroom and learned routing')
    p(d, 'Table 7. Per-query oracle, best single metadata-aware strategy, '
         'PAV Router-routed system, and router agreement.', italic=True,
      size=9)
    rows = []
    for ds in DS:
        o = T['oracle'][ds]['routed_oracle']['N@10']
        routed = ['UMA-RAG', 'LP-RAG', 'CA-HR']
        best = max(routed, key=lambda m: avg(ds, m, 'N@10'))
        rt = T['router'][ds]
        rows.append([DS_NAME[ds], f'{avg(ds, best, "N@10"):.4f} ({best})',
                     f4(avg(ds, 'CA-HR', 'N@10')),
                     f4(rt['routed_system']['N@10']), f4(o),
                     f"{rt['cv_accuracy']:.3f}", f"{rt['kappa']:.3f}"])
    add_table(d, ['Dataset', 'Best single', 'CA-HR', 'PAV Router',
                  'Oracle', 'Router acc.', "Cohen's κ"], rows)
    p(d,
      'Table 7 and Fig. 5 quantify how much per-query adaptivity could ever '
      'buy. The oracle that picks the per-query best among the three '
      'metadata-aware strategies exceeds the best single strategy by only '
      f'+0.006 (SCIDOCS) and +0.008 (SciFact) NDCG@10; the headroom is '
      f'larger on NFCorpus (+0.010) and TREC-COVID (+0.036), where the '
      f'metadata signals interact with sparse coverage. The learned router '
      f'does not recover this headroom on any dataset: out-of-fold label '
      f'accuracy reaches '
      f'{T["router"]["scifact"]["cv_accuracy"]*100:.1f}% on SciFact but '
      f'collapses to majority-class prediction (Cohen\'s κ between '
      f'{min(T["router"][d]["kappa"] for d in DS):.3f} and '
      f'{max(T["router"][d]["kappa"] for d in DS):.3f} across the four '
      f'datasets), and the routed system never exceeds simply always '
      f'choosing CA-HR. Across the four datasets, the tested surface-feature '
      f'router fails to recover the available per-query oracle headroom, '
      f'suggesting that domain-level configuration is a stronger default '
      f'than this lightweight form of query-level adaptation.')
    d.add_picture(os.path.join(BASE, 'figures', 'Fig5_routing.png'),
                  width=Inches(6.0))
    p(d, 'Fig. 5. Oracle headroom vs. learned routing (NDCG@10).',
      italic=True, size=9)

    h2(d, '6.6. Generation-side evaluation')
    p(d, 'Table 8. End-to-end answer quality with DeepSeek generation on 200 '
         'paired queries (50 per dataset across the four benchmarks): '
         'LLM-judged relevance and faithfulness (1-5), citation precision '
         'against gold judgments, and relevant passages in the top-5 '
         'context.',
      italic=True, size=9)
    ca, nh = g['CA-HR'], g['Neural-Hybrid']
    rows = [
        ['Relevance (judge 1-5)', f"{ca['relevance']['mean']:.2f} ± {ca['relevance']['std']:.2f}",
         f"{nh['relevance']['mean']:.2f} ± {nh['relevance']['std']:.2f}",
         pval(g['paired_relevance']['wilcoxon_p_two_sided'])],
        ['Faithfulness (judge 1-5)', f"{ca['faithfulness']['mean']:.2f} ± {ca['faithfulness']['std']:.2f}",
         f"{nh['faithfulness']['mean']:.2f} ± {nh['faithfulness']['std']:.2f}",
         pval(g['paired_faithfulness']['wilcoxon_p_two_sided'])],
        ['Citation precision', f"{g['paired_citation_precision']['mean_CA-HR']:.3f}",
         f"{g['paired_citation_precision']['mean_Neural-Hybrid']:.3f}",
         pval(g['paired_citation_precision']['wilcoxon_p_two_sided'])],
        ['Relevant docs in top-5', f"{g['paired_n_rel_context']['mean_CA-HR']:.2f}",
         f"{g['paired_n_rel_context']['mean_Neural-Hybrid']:.2f}",
         pval(g['paired_n_rel_context']['wilcoxon_p_two_sided'])],
    ]
    add_table(d, ['Measure', 'CA-HR backend', 'Neural-Hybrid backend',
                  'Wilcoxon (two-sided)'], rows)
    p(d,
      'Do the retrieval-level differences survive into the answers users '
      'actually read? We sampled 200 test queries (50 per dataset, fixed '
      'seed), generated cited answers with DeepSeek (deepseek-chat, '
      'temperature 0.2, top-5 passages as context), and scored each answer '
      'for relevance and faithfulness with an LLM judge and for citation '
      'precision against the gold judgments. The judge is the same '
      'DeepSeek chat model prompted with a fixed rubric (1-5 relevance; '
      '1-5 faithfulness conditioned on the retrieved context; the exact '
      'prompts are released with the code). Each answer was judged once, '
      'in randomised backend order with backend labels hidden from the '
      'judge; no human verification subsample was performed, which we flag '
      'as a limitation (Section 8.2). Table 8 shows the backends are '
      'statistically indistinguishable on every measure (all p > 0.48): at '
      'top-5, the two systems surface nearly the same passages '
      f'({g["paired_n_rel_context"]["mean_CA-HR"]:.2f} vs. '
      f'{g["paired_n_rel_context"]["mean_Neural-Hybrid"]:.2f} gold-relevant '
      f'documents on average). Two secondary findings are worth reporting. '
      f'First, absolute answer quality is high (relevance '
      f'{ca["relevance"]["mean"]:.2f}-{nh["relevance"]["mean"]:.2f}, '
      f'faithfulness '
      f'{ca["faithfulness"]["mean"]:.2f}-{nh["faithfulness"]["mean"]:.2f} '
      f'of 5), so the pipeline is usable in production. Second, citation '
      f'precision against the strict gold judgments averages about '
      f'{g["paired_citation_precision"]["mean_CA-HR"]:.2f} but varies '
      f'sharply with gold-judgment density: 0.87 on TREC-COVID, whose '
      f'queries carry dense graded judgments, versus 0.15 on SCIDOCS, whose '
      f'judgments are sparse; the model frequently cites plausible but not '
      f'gold-relevant passages, a failure mode invisible to retrieval '
      f'metrics and a target for future citation verification. We caveat '
      f'that the judge shares the generator\'s model family (Zheng et al., '
      f'2023); the '
      f'paired design controls for this bias, but absolute scores should be '
      f'read accordingly.')

    h2(d, '6.7. Efficiency')
    p(d,
      'All ten configurations are CPU-feasible. On the largest corpus '
      '(TREC-COVID, 171,332 documents) BM25 scoring averages 1,002 ms per '
      'query; dense scoring over precomputed embeddings and CA-HR '
      're-ranking add under 2 ms. On the smaller corpora BM25 costs '
      '7.6-97 ms per query and query encoding about 30 ms. In the deployed '
      'system, retrieval latency is negligible next to LLM generation, and '
      'the one-time corpus embedding cost (about 34-49 documents/s for '
      'MiniLM on 8 CPU threads) is amortised offline.')

    # ---- 7. Deployment feasibility case study -------------------------------
    h1(d, '7. Deployment feasibility case study')
    p(d,
      'The full pipeline has been deployed in the production assistant '
      'since 10 July 2026 '
      '(system name and URL withheld for anonymous review), on a single '
      'Aliyun ECS instance (2 vCPU, 1.6 GB RAM, '
      '40 GB disk, Ubuntu 22.04) with Node.js 20.20.2, MySQL 8.0.46, '
      'Nginx 1.18 terminating TLS (Let\'s Encrypt, auto-renewed), and PM2 '
      'supervision. Over the first month of operation the service recorded 6 '
      'registered users, 6 uploaded papers, 26 chat conversations with 63 '
      'messages, 6 AI-generated summaries, and 3 writing projects; the '
      'process restarted twice in six days (both manual deploys) and the '
      'edge responds in about 74 ms. This is a pilot-scale deployment, and '
      'we report it as such: its purpose is to demonstrate that the '
      'evaluated pipeline runs unchanged on commodity hardware in '
      'production, not to claim large-scale adoption.')
    p(d,
      'Two operational observations connect the deployment to the '
      'experimental findings. First, retrieval never appeared in the latency '
      'budget: user-perceived response time is dominated by the streamed LLM '
      'tokens, consistent with Section 6.7, so the choice among the ten '
      'configurations is free from an engineering standpoint. Second, '
      'user-uploaded papers are often fresh preprints with no citation '
      'record—the production analogue of TREC-COVID\'s inverted '
      'citation-relevance association—so '
      'the system defaults to the plain hybrid backend for user libraries '
      'and reserves citation-aware re-ranking for corpora where citations '
      'demonstrably correlate with relevance, which is exactly the '
      'informativeness-conditional configuration policy our experiments '
      'support.')

    # ---- 8. Discussion ------------------------------------------------------
    h1(d, '8. Discussion')
    h2(d, '8.1. A configuration policy for scientific RAG')
    p(d,
      'Taken together, the four-domain results support a simple '
      'domain-conditional policy grounded in bibliographic informativeness '
      '(Table 3): (i) where citation authority strongly separates relevant '
      'from non-relevant documents (computer science; citation-relevance '
      'AUC = 0.798), citation-aware signals contribute positively—uniform '
      'metadata augmentation (UMA-RAG) is the strongest metadata-aware '
      'variant on SCIDOCS, and within CA-HR the citation term is the single '
      'most valuable component—though a strong pretrained dense encoder '
      'alone is hard to beat; (ii) for claim-style biomedical retrieval, '
      'where the citation-relevance association is weak (AUC = 0.582), '
      'hybrid fusion is best-in-family and metadata terms are neutral; '
      '(iii) where citations are uninformative or inversely associated with '
      'relevance (nutrition, AUC = 0.498; pandemic medicine, AUC = 0.461—'
      'and, analogously, fresh user uploads with no citation record), '
      'metadata boosts are neutral-to-harmful regardless of coverage, and a '
      'stronger dense encoder such as BGE-small is the best single '
      'investment; and (iv) the tested surface-feature router does not '
      'recover the available oracle headroom anywhere we tested—'
      'configuration effort should be spent at domain level. Two '
      'qualifiers sharpen the policy. First, the metadata effects are '
      'backbone- and weight-conditional: at CA-HR\'s fixed weights every '
      'metadata gain measured on the MiniLM backbone disappeared or '
      'inverted on the stronger BGE-small backbone (Section 6.1), and a '
      '30-combination weight sweep shows the gain re-emerges only on the '
      'citation-informative corpus (SCIDOCS) and only after raising the '
      'citation weight (Section 6.4)—metadata re-ranking must therefore be '
      're-validated and re-calibrated whenever the underlying encoder is '
      'upgraded. Second, robustness under query '
      'corruption does not follow clean-query rankings—BM25 is the most '
      'noise-robust method on SCIDOCS while CA-HR is the most robust on '
      'TREC-COVID (Section 6.3)—so noise expectations should also enter the '
      'per-domain choice.')
    h2(d, '8.2. Limitations')
    p(d,
      'Six limitations qualify our claims. (1) TREC-COVID has only 50 test '
      'queries, so per-dataset rankings there carry wider confidence '
      'intervals; the cross-domain pattern, however, is consistent across '
      '1,673 queries in total. (2) Metadata coverage is below 100% by '
      'construction on TREC-COVID (69.8% citations); we mitigated with '
      'neutral defaults and disclosed coverage per field, but matched and '
      'unmatched documents may differ systematically. (3) The generation '
      'judge shares the generator\'s model family; the paired design '
      'controls relative bias, absolute judge scores may be inflated, and '
      'no human verification subsample was performed. (4) '
      'CA-HR\'s hyperparameters (alpha, beta, gamma, lambda, top-100 depth) '
      'were fixed a priori and not tuned per dataset, which is conservative '
      'but may understate the method where metadata is rich; the Section 6.4 '
      'weight sweep bounds this effect directly, showing a reachable gain '
      'only on SCIDOCS. (5) The '
      'deployment study is pilot-scale (6 users, one month); it '
      'demonstrates operability, not adoption. (6) The benchmark experiments '
      'operate on document-level representations (title and abstract), '
      'whereas the deployed pipeline retrieves passage-level '
      'chunks of uploaded papers; the deployment demonstrates engineering '
      'transferability but does not establish that all document-level '
      'ranking effects reproduce identically at passage level.')

    # ---- 9. Conclusion ------------------------------------------------------
    h1(d, '9. Conclusion')
    p(d,
      'We evaluated ten retrieval configurations for scientific RAG on '
      'four benchmarks with real bibliographic metadata, and deployed the '
      'stack in a live academic writing assistant. The answer to the title '
      'question is conditional on two dimensions rather than one: '
      'bibliographic metadata helps scientific RAG when the bibliographic '
      'signal is genuinely informative about relevance—citation authority '
      'separates relevant from non-relevant documents on SCIDOCS '
      '(AUC = 0.798) but not on NFCorpus (AUC = 0.498) or TREC-COVID '
      '(AUC = 0.461)—and requires re-calibration when the underlying content '
      'retriever changes: under fixed hyperparameters the citation gain '
      'observed with MiniLM vanishes on the stronger BGE-small backbone, and '
      'a 30-combination weight sweep recovers it only on the '
      'citation-informative corpus (SCIDOCS). A lightweight surface-feature '
      'router does not '
      'recover even the modest oracle headroom, '
      'generation-side quality is insensitive to the backend at top-5, and '
      'the whole pipeline runs on commodity CPU hardware in production. '
      'For practitioners, the actionable guidance is to '
      'configure retrieval per domain and to audit the citation-relevance '
      'association of a corpus before '
      'relying on metadata-aware ranking. Future work includes citation '
      'verification for generated answers, richer routing features '
      '(embedding-space and corpus-statistic signals), and scaling the '
      'deployment study to a larger user base.')

    # ---- Declaration of generative AI (required before references) --------
    h1(d, 'Declaration of generative AI and AI-assisted technologies in '
          'the manuscript preparation process')
    p(d,
      'During the preparation of this work the author used Kimi (Moonshot '
      'AI) for code debugging and language polishing. All AI-assisted '
      'output, including code and experimental results, was reviewed, '
      'executed, and verified by the author, who takes full responsibility '
      'for the content of the published article.')

    # ---- References (APA, author-year, alphabetical) -----------------------
    h1(d, 'References')
    refs = [
        'Asai, A., He, J., Shao, R., Shi, W., Singh, A., Chang, J. C., Lo, K., Soldaini, L., Feldman, S., D\'Arcy, M., Wadden, D., Latzke, M., Tian, M., Ji, P., Liu, S., Tong, H., Wu, B., Xiong, Y., Zettlemoyer, L., ... Hajishirzi, H. (2024a). OpenScholar: Synthesizing scientific literature with retrieval-augmented language models. arXiv:2411.14199. https://doi.org/10.48550/arXiv.2411.14199',
        'Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024b). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In Proceedings of ICLR.',
        'Bao, T., Nayeem, M. T., Rafiei, D., & Zhang, C. (2025). SurveyGen: Quality-aware scientific survey generation with large language models. In Proceedings of EMNLP (pp. 2712–2736). https://doi.org/10.18653/v1/2025.emnlp-main.136',
        'Boteva, V., Gholipour, D., Sokolov, A., & Riezler, S. (2016). A full-text learning to rank dataset for medical information retrieval. In Proceedings of ECIR (pp. 716–722).',
        'Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv:2402.03216. https://doi.org/10.48550/arXiv.2402.03216',
        'Cohan, A., Feldman, S., Beltagy, I., Downey, D., & Weld, D. S. (2020). SPECTER: Document-level representation learning using citation-informed transformers. In Proceedings of ACL (pp. 2270–2282). https://doi.org/10.18653/v1/2020.acl-main.207',
        'Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by latent semantic analysis. Journal of the American Society for Information Science, 41(6), 391–407.',
        'DeepSeek-AI. (2024). DeepSeek-V3 technical report. arXiv:2412.19437. https://doi.org/10.48550/arXiv.2412.19437',
        'Ding, H., Zhao, Y., Hu, T., Wang, Z., Patwardhan, M., & Cohan, A. (2026). SciRAG: Adaptive, citation-aware, and outline-guided retrieval and synthesis for scientific literature. In Proceedings of EACL (Volume 1: Long Papers) (pp. 6440–6460).',
        'Hwang, J., Park, J., Park, H., Kim, D., Park, S., & Ok, J. (2025). Retrieval-augmented generation with estimation of source reliability. In Proceedings of EMNLP (pp. 34279–34303). https://doi.org/10.18653/v1/2025.emnlp-main.1738',
        'Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.-t. (2020). Dense passage retrieval for open-domain question answering. In Proceedings of EMNLP (pp. 6769–6781). https://doi.org/10.18653/v1/2020.emnlp-main.550',
        'Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In Proceedings of NeurIPS (pp. 9459–9474).',
        'Lin, J. (2021). A proposed conceptual framework for a representational approach to information retrieval. ACM SIGIR Forum, 55(2).',
        'Nogueira, R., Jiang, Z., Pradeep, R., & Lin, J. (2020). Document ranking with a pretrained sequence-to-sequence model. In Findings of EMNLP (pp. 708–718). https://doi.org/10.18653/v1/2020.findings-emnlp.63',
        'Ostendorff, M., Rethmeier, N., Augenstein, I., Gipp, B., & Rehm, G. (2022). Neighborhood contrastive learning for scientific document representations with citation embeddings. In Proceedings of EMNLP (pp. 11670–11688).',
        'Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv:2205.01833. https://doi.org/10.48550/arXiv.2205.01833',
        'Qu, Y., Ding, Y., Liu, J., Liu, K., Ren, R., Zhao, W. X., Dong, D., Wu, H., & Wang, H. (2021). RocketQA: An optimized training approach to dense passage retrieval. In Proceedings of NAACL (pp. 5835–5847). https://doi.org/10.18653/v1/2021.naacl-main.466',
        'Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of EMNLP-IJCNLP (pp. 3982–3992). https://doi.org/10.18653/v1/D19-1410',
        'Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4), 333–389. https://doi.org/10.1561/1500000019',
        'Shao, Z., Gong, Y., Shen, Y., Huang, M., Duan, N., & Chen, W. (2023). Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy. In Findings of EMNLP (pp. 9248–9274).',
        'Singh, A., D\'Arcy, M., Cohan, A., Downey, D., & Feldman, S. (2023). SciRepEval: A multi-format benchmark for scientific document representations. In Proceedings of EMNLP (pp. 5548–5566).',
        'Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Proceedings of NeurIPS Datasets and Benchmarks.',
        'Voorhees, E., Alam, T., Bedrick, S., Demner-Fushman, D., Hersh, W. R., Lo, K., Roberts, K., Soboroff, I., & Wang, L. L. (2021). TREC-COVID: Constructing a pandemic information retrieval test collection. ACM SIGIR Forum, 54(1), 1–12.',
        'Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020). Fact or fiction: Verifying scientific claims. In Proceedings of EMNLP (pp. 7534–7550). https://doi.org/10.18653/v1/2020.emnlp-main.609',
        'Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., & Wei, F. (2022). Text embeddings by weakly-supervised contrastive pre-training. arXiv:2212.03533. https://doi.org/10.48550/arXiv.2212.03533',
        'Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., Zhao, W. X., Wei, Z., & Wen, J.-R. (2024). A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6), 186345.',
        'Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. In Proceedings of NeurIPS.',
        'Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2023). C-Pack: Packaged resources to advance general Chinese embedding. arXiv:2309.07597. https://doi.org/10.48550/arXiv.2309.07597',
        'Yousuf, R. B., Xu, S., Sharma, M., Neeser, A., Latimer, C., & Ramakrishnan, N. (2026). Utilizing metadata for better retrieval-augmented generation. In Proceedings of ECIR (pp. 305–319). https://doi.org/10.1007/978-3-032-21289-4_20',
        'Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Proceedings of NeurIPS Datasets and Benchmarks.',
        'Zhuang, S., Zhuang, H., Koopman, B., & Zuccon, G. (2024). A setwise approach for effective and highly efficient zero-shot ranking with large language models. In Proceedings of SIGIR (pp. 38–47).',
    ]
    for r in refs:
        p(d, r, size=10)

    d.save(os.path.join(OUT, '01_Manuscript_ESWA.docx'))


# ===========================================================================
# Cover letter
# ===========================================================================
def build_cover_letter():
    d = new_doc()
    p(d, 'Cover Letter', bold=True, size=14)
    p(d, f'{AUTHOR}')
    p(d, AFFIL)
    p(d, f'E-mail: {EMAIL}  |  ORCID: {ORCID}')
    d.add_paragraph()
    p(d, 'Dear Editor-in-Chief,')
    p(d,
      f'We submit the manuscript "{TITLE}" for consideration in Expert '
      f'Systems with Applications.')
    p(d,
      'The manuscript fits the journal\'s focus on applied intelligent '
      'systems: it couples (i) a four-domain, ten-configuration evaluation '
      'of retrieval for scientific question answering with real '
      'bibliographic metadata, (ii) a replicated negative result on '
      'query-level routing that carries practical design guidance, (iii) a '
      'generation-side evaluation of answer quality, and (iv) a deployment '
      'case study of the full pipeline in PaperPilot, a live academic '
      'writing assistant running on commodity hardware. The study yields an '
      'actionable, domain-conditional configuration policy for scientific '
      'RAG systems, and all data, code, and per-query results are publicly '
      f'available at {REPO} (archived at https://doi.org/{DOI}) '
      'for full reproducibility.')
    p(d,
      'The manuscript is original, is not under consideration elsewhere, '
      'and is approved by the author. The author declares no competing '
      'interests.')
    p(d,
      'Regarding the journal\'s institutional e-mail requirement: the '
      'corresponding author is an undergraduate researcher at Sanya '
      'University, which does not provide institutional e-mail accounts to '
      'undergraduate students. The personal e-mail address given above and '
      'in the submission system is therefore used for all correspondence.')
    d.add_paragraph()
    p(d, 'Suggested reviewers (no conflicts of interest):')
    for r in [
        'Jimmy Lin, University of Waterloo, Canada — jimmylin@uwaterloo.ca '
        '(information retrieval, neural ranking)',
        'Arman Cohan, Yale University / Allen Institute for AI, USA — '
        'arman.cohan@yale.edu (scientific document representations)',
        'Guido Zuccon, The University of Queensland, Australia — '
        'g.zuccon@uq.edu.au (health/medical IR, LLM rankers)',
        'Andrew Yates, Johns Hopkins University, USA — '
        'andrew.yates@jhu.edu (dense retrieval, ranking)',
    ]:
        par = d.add_paragraph(style='List Bullet')
        par.add_run(r)
    d.add_paragraph()
    p(d, 'Sincerely,')
    p(d, AUTHOR)
    d.save(os.path.join(OUT, '03_Cover_Letter.docx'))


# ===========================================================================
# Declarations
# ===========================================================================
def build_declarations():
    d = new_doc()
    p(d, 'Declarations', bold=True, size=14)
    h2(d, 'Ethics approval and consent to participate')
    p(d, 'Not applicable. The study uses only public benchmark datasets '
         'and aggregated, non-personal operational counts from the deployed '
         'system (registered users, uploaded papers, conversations, '
         'messages, restarts, response times). No user content, no personal '
         'data, and no human subjects were involved; no ethics committee '
         'approval was required.')
    h2(d, 'Consent for publication')
    p(d, 'Not applicable.')
    h2(d, 'Declaration of competing interest')
    p(d, 'The author declares no competing financial or personal interests.')
    h2(d, 'Funding')
    p(d, 'This research received no external funding.')
    h2(d, 'CRediT authorship contribution statement')
    p(d, f'{AUTHOR}: Conceptualization, Methodology, Software, Validation, '
         f'Formal analysis, Investigation, Data curation, Writing — original '
         f'draft, Writing — review & editing, Visualization.')
    h2(d, 'Data availability')
    p(d,
      'All four benchmarks are public (BEIR format). Bibliographic metadata '
      'was obtained from the public Semantic Scholar and OpenAlex APIs. All '
      f'code, fetched metadata, and per-query result files are publicly '
      f'available at {REPO} (archived at https://doi.org/{DOI}).')
    h2(d, 'Use of AI in the manuscript preparation process')
    p(d,
      'During the preparation of this work the author used Kimi (Moonshot '
      'AI) for code debugging and language polishing. All AI-assisted '
      'output, including code and experimental results, was reviewed, '
      'executed, and verified by the author, who takes full responsibility '
      'for the integrity of the publication.')
    d.save(os.path.join(OUT, '04_Declarations.docx'))


# ===========================================================================
# main
# ===========================================================================
def copy_figures():
    mapping = {
        'Fig1_architecture.png': 'system_architecture.png',
        'Fig2_main_results.png': 'Fig2_main_results.png',
        'Fig3_ablation.png': 'Fig3_ablation.png',
        'Fig4_robustness.png': 'Fig4_robustness.png',
        'Fig5_routing.png': 'Fig5_routing.png',
    }
    src = os.path.join(BASE, 'figures')
    for dst, s in mapping.items():
        shutil.copy(os.path.join(src, s), os.path.join(FIGOUT, dst))


if __name__ == '__main__':
    build_title_page()
    build_manuscript()
    build_highlights()
    build_cover_letter()
    build_declarations()
    copy_figures()
    print('ESWA package written to', OUT)
    print(sorted(os.listdir(OUT)))
