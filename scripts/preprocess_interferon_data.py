"""
	preprocess_interferon_data.py
	Preprocess the publicly available interferon dataset through the scanpy pipeline.
	https://www.nature.com/articles/nbt.4042
"""


import pandas as pd

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Computing pseudotime metrics for 2 clusters.')
	parser.add_argument('--data_path', type=str, metavar='P', help='path to the data')
	args = parser.parse_args()

	MERGE_DEMUXLET = True
	FILTER_SINGLETS = True
	BASIC_FILTERING = True
	NORMALIZATION = True
	DIMENSION_REDUCTION = True

	if MERGE_DEMUXLET:

		print('Merging demuxlet output...')
		demuxlet_output = read_demuxlet_output('/ye/yelabstore3/mincheol/crohns_pilot/demuxlet_imputed/lane_1_output/demuxlet.best', 1)

		# demuxlet_output = pd.concat([
		# 	read_demuxlet_output('/ye/yelabstore3/mincheol/crohns_pilot/demuxlet_imputed/lane_1_output/demuxlet.best', 1),
		# 	read_demuxlet_output('/ye/yelabstore3/mincheol/crohns_pilot/demuxlet_raw/Lane2_demuxlet_out/demuxlet.best', 2)])

		adata = sc.read_10x_h5(args.data_path + 'single_cell/raw_gene_bc_matrices_h5.h5', genome='hg19')
		adata.obs['BARCODE'] = adata.obs.index.tolist()
		adata.obs = adata.obs.merge(demuxlet_output, on='BARCODE', how='left')

		adata.write(args.data_path + 'single_cell/raw.demuxlet.h5ad')

	if FILTER_SINGLETS:

		print('Filtering for singlets...')
		adata = sc.read(args.data_path + 'single_cell/raw.demuxlet.h5ad')
		adata = adata[adata.obs['BEST'].str.startswith('SNG'), :].copy()
		print(adata.shape)
		adata.write(args.data_path + 'single_cell/raw.demuxlet.sng.h5ad')

	if BASIC_FILTERING:

		print('Filtering for UMI and SNP...')
		adata = sc.read(args.data_path + 'single_cell/raw.demuxlet.sng.h5ad')
		sc.pp.filter_cells(adata, min_genes=50)
		sc.pp.filter_genes(adata, min_cells=3)

		mito_genes = adata.var_names.str.startswith('MT-')
		adata.obs['percent_mito'] = np.sum(
		    adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
		adata.obs['n_counts'] = adata.X.sum(axis=1).A1
		adata = adata[(adata.obs['n_counts'] > 50), :].copy()
		print(adata.shape)
		adata.write(args.data_path + 'single_cell/raw.demuxlet.sng.numi50.raw.h5ad')

	if NORMALIZATION:

		print('Normalizing...')
		adata = sc.read(args.data_path + 'single_cell/raw.demuxlet.sng.numi50.raw.h5ad')
		sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
		sc.pp.log1p(adata)
		adata.raw = adata
		sc.pp.filter_genes_dispersion(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
		sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])
		sc.pp.regress_out(adata, ['LANE'])
		sc.pp.scale(adata, max_value=10)
		print(adata.shape)
		adata.write(args.data_path + 'single_cell/raw.demuxlet.sng.numi50.norm.h5ad')

	if DIMENSION_REDUCTION:

		print('Downstream analysis...')
		adata = sc.read(args.data_path + 'single_cell/raw.demuxlet.sng.numi50.norm.h5ad')
		sc.tl.pca(adata, svd_solver='arpack')
		sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
		sc.tl.umap(adata)
		sc.tl.louvain(adata, resolution=0.2)
		sc.tl.rank_genes_groups(adata, 'louvain', method='logreg')
		adata.write(args.data_path + 'single_cell/raw.demuxlet.sng.numi50.norm.dr.h5ad')