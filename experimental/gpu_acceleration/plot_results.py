"""Render saved benchmark results; no expensive analysis or GPU work."""
import os
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('report',type=Path)
    args=parser.parse_args()
    report=json.loads(args.report.read_text())
    root=args.report.parent
    n,b=report['tested_genes'],report['args']['boots']
    arrays={mode:np.load(root/f'{mode}_{n}g_{b}b.npz') for mode in report['runs']}
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    modes=list(report['runs'])
    labels={'cpu':'CPU (2 workers)','cpu_repeat':'CPU repeat (2 workers)',
            'gpu_cpu':'GPU + reference CPU regression',
            'gpu_cpu_map':'GPU + factored CPU regression','gpu':'GPU + resident regression'}
    wall=[report['runs'][m]['wall_seconds'] for m in modes]
    ax=axes[0,0]
    ax.barh([labels[m] for m in modes],wall,color=['#777777' if m.startswith('cpu') else '#167d9a' for m in modes])
    for i,t in enumerate(wall): ax.text(t+max(wall)*.02,i,f'{t:.1f}s',va='center')
    ax.set_xlim(0,max(wall)*1.22); ax.invert_yaxis(); ax.set_xlabel('Public-call wall time (seconds)')
    ax.set_title('End-to-end timing under workstation load')
    for ax,key,title in zip((axes[0,1],axes[1,0]),('mean_se','var_se'),('Mean standard errors','Variability standard errors')):
        for mode,label,color in [('gpu','GPU / CPU','#167d9a'),('cpu_repeat','CPU repeat / CPU','#d18f21')]:
            if mode in arrays:
                ratio=arrays[mode][key]/arrays['cpu'][key]
                ax.hist(ratio,bins=np.linspace(.93,1.07,29),alpha=.6,label=label,color=color)
        ax.axvline(1,color='black',lw=1); ax.set_title(title)
        ax.set_xlabel('SE ratio'); ax.set_ylabel('Genes'); ax.legend(frameon=False)
    ax=axes[1,1]
    stages=report['runs']['gpu']['stages_seconds']
    ax.barh(list(stages),list(stages.values()),color='#167d9a')
    ax.set_xlabel('Seconds'); ax.set_title('Resident GPU path: measured stages'); ax.invert_yaxis()
    fig.suptitle(f'CD14+ Monocytes · {n} genes · {b:,} bootstraps · 16 donor/condition groups')
    fig.savefig(root/'comparison.png',dpi=160)
    fig.savefig(root/'comparison.svg')
    print(root/'comparison.png')


if __name__=='__main__': main()
