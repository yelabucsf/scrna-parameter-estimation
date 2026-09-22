"""Summarize measured timings and independent-run output agreement."""
import os
import json
from pathlib import Path
from real_data_bench import np,comparison

root=Path('experimental/gpu_acceleration/results_cell_matmul')
base=np.load(root/'0_gpu_compressed.npz')
keys=('mean_coef','mean_se','mean_asl','var_coef','var_se','var_asl')
reference={k:base[k] for k in keys}
report={}
for folder in (root,root.parent/'results_cell_matmul_cpu'):
    for file in sorted(folder.glob('*_*.npz')):
        if file.name=='0_gpu_compressed.npz':continue
        z=np.load(file)
        np.testing.assert_array_equal(z['genes'],base['genes'])
        report[str(file.relative_to(root.parent))]=comparison(reference,{k:z[k] for k in keys})
(root/'accuracy.json').write_text(json.dumps(report,indent=2))
print('Independent-run comparisons:')
for name,r in report.items():
    print(name, {k:r[k]['ratio_quantiles'] for k in ('mean_se','var_se')},flush=True)
sweep=json.loads((root/'sweep.json').read_text())
print('Sweep medians: cells, CPU compressed/matrix, GPU compressed/matrix')
for cells in sorted(set(r['cells'] for r in sweep['runs'])):
    print(cells,[round(float(np.median([r['seconds'] for r in sweep['runs'] if r['cells']==cells and r['mode']==mode])),4)
                 for mode in ('cpu_compressed','cpu_matrix','gpu_compressed','gpu_matrix')])
