# -*- coding: utf-8 -*-
"""System architecture figure for the ESWA revision (PaperPilot deployment)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(10.5, 6.2))
ax.set_xlim(0, 105)
ax.set_ylim(0, 62)
ax.axis('off')

C_CLIENT = '#dbeafe'
C_EDGE = '#e0e7ff'
C_APP = '#fef3c7'
C_RAG = '#dcfce7'
C_DB = '#f3e8ff'
C_EXT = '#fee2e2'
EDGE = '#334155'


def box(x, y, w, h, label, fc, fs=9.5, weight='normal', ec=EDGE):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.35,rounding_size=1.2',
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
            fontsize=fs, weight=weight, linespacing=1.45)
    return p


def arrow(x1, y1, x2, y2, label='', fs=8, style='-|>', color=EDGE, rad=0.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, lw=1.4,
                        color=color, mutation_scale=14,
                        connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.6, label, ha='center',
                fontsize=fs, color='#0f172a')


# ---- deployment boundary ------------------------------------------------
bound = FancyBboxPatch((24, 3), 78, 56, boxstyle='round,pad=0.6,rounding_size=2',
                       fc='#f8fafc', ec='#64748b', lw=1.6, linestyle='--')
ax.add_patch(bound)
ax.text(63, 61.2, 'Aliyun ECS  (2 vCPU, 1.6 GB RAM, Ubuntu 22.04)  —  PM2 process manager, Let\'s Encrypt TLS',
        ha='center', fontsize=9, color='#475569', style='italic')

# ---- client --------------------------------------------------------------
box(2, 24, 20, 14, 'User / Browser\nReact 18 SPA (Vite)\nreader · chat · writing', C_CLIENT, weight='bold')

# ---- edge -----------------------------------------------------------------
box(30, 46, 30, 9, 'Nginx 1.18 reverse proxy  (:443 HTTPS)\nstatic assets · API proxy · SSE streaming', C_EDGE)

# ---- app tier --------------------------------------------------------------
box(30, 27, 68, 14, '', '#ffffff', ec=EDGE)
ax.text(64, 39.2, 'Node.js 20 / Express API  (PM2: paperpilot)', ha='center',
        fontsize=10, weight='bold')
box(33, 28.5, 30, 8.5, 'REST + SSE endpoints\nauth · papers · chat · writing', C_APP, fs=8.5)
box(66, 28.5, 29, 8.5, 'RAG orchestration\nchunk · embed · retrieve · generate', C_APP, fs=8.5)

# ---- RAG pipeline ----------------------------------------------------------
box(30, 9, 68, 14, '', '#ffffff', ec=EDGE)
ax.text(64, 21.2, 'CA-HR retrieval pipeline', ha='center', fontsize=10, weight='bold')
box(33, 10.5, 20, 8, 'Hybrid scoring\n0.6·BM25 + 0.4·dense\n(MiniLM-L6, 384-d)', C_RAG, fs=8.5)
box(55.5, 10.5, 20, 8, 'Metadata re-rank\ntop-100: + 0.15·citation\n+ 0.10·recency', C_RAG, fs=8.5)
box(78, 10.5, 17, 8, 'top-5 passages\nas cited context\n[1]…[5]', C_RAG, fs=8.5)

# ---- db ---------------------------------------------------------------------
box(2, 6, 20, 12, 'MySQL 8.0\nusers · papers ·\nconversations · messages\nhighlights · summaries', C_DB, fs=8.5)

# ---- external llm ------------------------------------------------------------
box(78, 46, 20, 9, 'DeepSeek API\ndeepseek-chat\n(answer generation)', C_EXT, fs=9)

# ---- arrows -------------------------------------------------------------------
arrow(22, 34, 30, 48, 'HTTPS')
arrow(45, 46, 45, 41, 'proxy_pass :3000')
arrow(63, 32, 66, 32, '')
arrow(80.5, 28.5, 80.5, 23, '')
arrow(43, 14.5, 55.5, 14.5, '')
arrow(75.5, 14.5, 78, 14.5, '')
arrow(88, 37, 88, 46, 'prompt + context /\nSSE token stream', style='<|-|>', fs=7.5)
arrow(22, 12, 30, 12, 'SQL (Drizzle ORM)', style='<|-|>')

fig.savefig(os.path.join(OUT, 'system_architecture.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'system_architecture.png'), dpi=300, bbox_inches='tight')
print('saved:', os.listdir(OUT))
