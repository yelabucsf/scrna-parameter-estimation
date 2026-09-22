"""Summarize and plot the completed full-panel fibroblast benchmark."""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent
out=p/'results_fibroblasts'
r=json.loads((out/'report.json').read_text())
cpu=pd.read_csv(out/'cpu_seed5.csv').set_index('gene')
gpu=pd.read_csv(out/'gpu_seed5.csv').set_index('gene').loc[cpu.index]
fig,axes=plt.subplots(2,2,figsize=(9,8),layout='constrained')
for row,kind in enumerate(('de','dv')):
    for col,stat in enumerate(('coef','se')):
        ax=axes[row,col];x=cpu[f'{kind}_{stat}'];y=gpu[f'{kind}_{stat}']
        valid=np.isfinite(x)&np.isfinite(y)
        ax.scatter(x[valid],y[valid],s=3,alpha=.25,rasterized=True)
        lo=min(x[valid].min(),y[valid].min());hi=max(x[valid].max(),y[valid].max())
        ax.plot([lo,hi],[lo,hi],color='black',lw=.8)
        ax.set(xlabel='Compressed CPU',ylabel='GPU',title=f'{kind.upper()} {stat}: {valid.sum():,} genes')
fig.suptitle('Fibroblasts: bladder versus subcutaneous adipose\n10,000 bootstrap draws; independent CPU/GPU sampling')
fig.savefig(out/'cpu_gpu_agreement.png',dpi=180);fig.savefig(out/'cpu_gpu_agreement.svg')
runs=r['runs'];cr=next(v for v in runs if v['backend']=='cpu');gr=[v for v in runs if v['backend']=='gpu']
lines=['# Fibroblasts: full-panel CPU/GPU validation','',
'Raw integer counts from `ts_stromal.h5ad` (`raw/X`), exact `cell_type="fibroblast"`, 10X only. Bladder is the treatment and subcutaneous adipose the reference. Five donors have at least 30 cells in both tissues: '+', '.join(r['donors'])+'. Groups are donor × tissue; donor indicators are covariates.','',
f"There are **{r['cells']:,} cells and {r['genes']:,} tested genes** after the standard filters (`min_perc_group=0.7`). All filtered genes were tested on both backends, with **10,000 draws**, hyper-relative moments, normal ASL, and no donor resampling. Capture rate q=0.07 is an assumed benchmark setting, not a tissue-specific estimate. This validates computational agreement, not biological calibration of that assumption.",'',
'| Backend | Seed | Seconds | DE FDR < .05 | DV FDR < .05 |','|---|---:|---:|---:|---:|']
for v in runs:lines.append(f"| {v['backend']} | {v['seed']} | {v['seconds']:.2f} | {v['de_fdr_05']} | {v['dv_fdr_05']} |")
lines+=['',f"Full CPU/GPU speed ratio: **{cr['seconds']/gr[0]['seconds']:.1f}×** using the first GPU run. CPU uses ten joblib threads with one BLAS thread each; process affinity is CPUs 0–9. Threads avoid ten separate scientific-Python processes in this 8 GiB WSL instance. GPU is an RTX 3060 12 GB with the automatic memory policy; peak live tensor memory **{max(v['peak_allocated_gib'] for v in gr):.2f} GiB**. Public-call timings exclude file loading, preprocessing, and result/FDR export. Preparation took {r['preparation_seconds']:.2f} seconds. These are measurements on this machine, not universal speed claims.",'','## Agreement','']
for kind in ('de','dv'):
    c=r['gpu_vs_cpu'][kind];q=c['se_ratio_quantiles'];a=r[kind+'_fdr_agreement']
    lines.append(f"- {kind.upper()}: GPU/CPU SE ratio median **{q[2]:.4f}**, central 90% **{q[1]:.4f}–{q[3]:.4f}**, range {q[0]:.4f}–{q[4]:.4f}; {c['nan_se_mismatch']} NaN-SE mismatches. FDR discoveries: {a['both']} shared, {a['cpu_only']} CPU-only, {a['gpu_only']} GPU-only.")
lines+=['','Bootstrap draws differ across backends. Borderline calls can differ with Monte Carlo noise; the independent GPU repeat is included in the JSON report. BH correction is performed separately for DE and DV across the full panel. Inference uses the existing cell-bootstrap model conditional on these donors. This experiment covers mean and variability tests, not correlations.','',
'![CPU/GPU agreement](results_fibroblasts/cpu_gpu_agreement.png)','',
'## Reproduce and inspect','','```bash','conda activate torch','python experimental/gpu_acceleration/run_fibroblasts.py','python experimental/gpu_acceleration/analyze_fibroblasts.py','```','',
'The runner reads only selected raw-count rows and checkpoints completed backend runs, reusing them after an interruption. Source files are not modified.','',
'- [Full CPU results](results_fibroblasts/cpu_seed5.csv)','- [Full GPU results](results_fibroblasts/gpu_seed5.csv)','- [GPU repeat](results_fibroblasts/gpu_seed6.csv)','- [Donor cell counts and exclusions](results_fibroblasts/donor_cell_counts.csv)','- [Settings, timings and comparisons](results_fibroblasts/report.json)','']
(p/'FIBROBLAST_RESULTS.md').write_text('\n'.join(lines))
print('\n'.join(lines[:25]))
