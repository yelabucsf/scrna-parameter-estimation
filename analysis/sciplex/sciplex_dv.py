import scanpy as sc
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import scipy.stats as stats

import sys
sys.path.append('/home/ssm-user/Github/scrna-parameter-estimation/dist/memento-0.0.6-py3.8.egg')
sys.path.append('/home/ssm-user/Github/misc-seq/miscseq/')
import encode
import memento

import warnings
warnings.filterwarnings("ignore")

def compare_drugs(drug1, drug2, target):
	
	for drug1, drug2 in itertools.combinations(drug_list,2):
    
		print(drug1, '   vs.   ',drug2)

		subset = adata[adata.obs.product_name.isin([drug1, drug2])].copy().copy()
		subset.obs['is_drug1'] = (subset.obs.product_name==drug1).astype(int)
		subset.obs['dose_level'] = 'dose_' + subset.obs['dose'].astype(str)

		memento.create_groups(subset, label_columns=['is_drug1', 'dose_level'])
		memento.compute_1d_moments(subset, min_perc_group=.9)
		memento.ht_1d_moments(
			subset, 
			formula_like='1 + is_drug1 + dose_level',
			cov_column='is_drug1', 
			num_boot=10000, 
			verbose=0,
			num_cpus=94)
		
		return subset
	

if __name__ == '__main__':
	
	data_path = '/data_volume/memento/sciplex/'
	ct = 'A549'
	adata = sc.read(data_path + 'h5ad/{}.h5ad'.format(ct))
	
	target_list = [
		'Aurora Kinase',
		'DNA/RNA Synthesis',
		'HDAC',
		'Histone Methyltransferase',
		'JAK',
		'PARP',
		'Sirtuin']
	
	target_to_dir = {
		'Aurora Kinase':'aurora',
		'DNA/RNA Synthesis':'dna_rna',
		'HDAC':'hdac',
		'Histone Methyltransferase':'histmeth',
		'JAK':'jak',
		'PARP':'parp',
		'Sirtuin':'sirt'}
	
	adata.obs['q'] = 0.05
	memento.setup_memento(adata, q_column='q', filter_mean_thresh=0.07)
	
	for target in target_list:
		print('Starting {}.......'.format(target))
		
		drug_counts = adata.obs.query('target == "{}"'.format(target)).product_name.value_counts()
		drug_list = drug_counts[drug_counts > 0].index.tolist()
		
		for drug1, drug2 in itertools.combinations(drug_list,2):
    
			print(target, drug1, drug2)
			result = compare_drugs(drug1, drug2, target)
			
			result.write(data_path + '{}/{}_vs_{}_stratified.h5ad'.format(target_to_dir[target], drug1, drug2))