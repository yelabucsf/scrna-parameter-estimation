"""Plot measured crossover and full-call results as standalone artifacts."""
import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('experimental/gpu_acceleration/results_cell_matmul')
sweep=json.loads((root/'sweep.json').read_text())
full=json.loads((root/'full.json').read_text())
cpu_full=json.loads((root.parent/'results_cell_matmul_cpu/full.json').read_text())
full['runs']=[r for r in full['runs'] if r['mode']!='cpu_matrix']+cpu_full['runs']
fig,axes=plt.subplots(1,3,figsize=(14,4.2))
for ax,prefix,title in zip(axes[:2],('cpu','gpu'),('CPU: at most two cores','GPU: RTX 3060, IEEE fp32')):
    for kind,label,color in [('compressed','Compressed states','#777777'),('matrix','Shared cell weights','#167d9a')]:
        rows=[r for r in sweep['runs'] if r['mode']==prefix+'_'+kind]
        cells=sorted(set(r['cells'] for r in rows))
        vals=[[r['seconds'] for r in rows if r['cells']==n] for n in cells]
        center=np.array([np.median(v) for v in vals])
        ax.plot(cells,center,'o-',label=label,color=color)
        ax.fill_between(cells,[min(v) for v in vals],[max(v) for v in vals],alpha=.15,color=color)
    ax.set(xscale='log',yscale='log',xlabel='Cells per computational group',ylabel='Bootstrap seconds',title=title)
    ax.grid(alpha=.2);ax.legend(fontsize=8)
ax=axes[2]
modes=['gpu_compressed','gpu_matrix','cpu_matrix']
labels=['GPU\ncompressed','GPU\ncell weights','CPU\ncell weights']
vals=[[r['seconds'] for r in full['runs'] if r['mode']==m] for m in modes]
centers=[np.median(v) for v in vals]
ax.bar(labels,centers,color=['#777777','#167d9a','#75b5b5'])
for i,(center,v) in enumerate(zip(centers,vals)):
    ax.text(i,center+.4,f'{center:.1f} s',ha='center')
    ax.scatter([i]*len(v),v,color='black',s=12,zorder=3)
ax.set(title='Full ht_1d_moments call',ylabel='Seconds')
ax.spines[['top','right']].set_visible(False)
fig.suptitle('Shared cell bootstrap versus per-gene state compression',fontsize=14)
fig.text(.5,.005,'Sweep: 128 genes, 10,000 bootstraps; pooled/resampled cells are computational workloads. Full call: 1,742 genes, 16 real groups.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,.93))
for ext in ('svg','png'):fig.savefig(root/f'comparison.{ext}',dpi=180)
print(root/'comparison.svg')
