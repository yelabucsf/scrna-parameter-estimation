import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import numpy as np
import scanpy as sc
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection
from patsy import dmatrix, dmatrices 
import statsmodels.api as sm
import logging
import pickle as pkl
import os

import sys
sys.path.append('/home/ubuntu/Github/scrna-parameter-estimation/')

import memento

logging.basicConfig(
    format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.captureWarnings(True)

columns = ['coef', 'pval', 'fdr']

canogamez_datasets = [
     'CD4_Memory-Th0',
     'CD4_Memory-Th2',
     'CD4_Memory-Th17',
     'CD4_Memory-iTreg',
     'CD4_Naive-Th0',
     'CD4_Naive-Th2',
     'CD4_Naive-Th17',
     'CD4_Naive-iTreg']
hagai_datasets = [
        'Hagai2018_mouse-lps',
        'Hagai2018_mouse-pic',
        'Hagai2018_pig-lps',
        'Hagai2018_rabbit-lps',
        'Hagai2018_rat-lps',
        'Hagai2018_rat-pic']
bulk_methods = [
    ('deseq2_lrt',['log2FoldChange', 'pvalue', 'padj']),
    ('deseq2_wald',['log2FoldChange', 'pvalue', 'padj']),
    ('edger_lrt',['logFC', 'PValue', 'FDR']),
    ('edger_qlft',['logFC', 'PValue', 'FDR']),
]

sc_methods = [ 
    ('quasiML',['coef', 'pval', 'fdr']),
    ('edger_lrt',['logFC', 'PValue', 'FDR']),
    ('edger_qlft',['logFC', 'PValue', 'FDR']),
    ('deseq2_wald',['log2FoldChange', 'pvalue', 'padj']),
    ('deseq2_lrt',['log2FoldChange', 'pvalue', 'padj']),
    # ('MAST', ['coef', 'Pr(>Chisq)','fdr']),
    ('t',['coef', 'pval', 'fdr']),
    ('MWU',['coef', 'pval', 'fdr']),
]

def concordance_auc(refs, x, k=5):
    count = 0
    for i in range(1, k+1):
        
        ref_total = 0
        for ref in refs:
            ref_total += len(set(x[:i]) & set(ref[:i]))
            
        count += ref_total/len(refs)
        
    return count / (k*(k+1)/2)

    
def hagai_validation():
    
    data_path = '/data_volume/bulkrna/hagai/'
    
    def read_bulk_dataset(dataset, method, cols):

        df = pd.read_csv(data_path + 'bulk_rnaseq/results/{}_{}.csv'.format(dataset, method), index_col=0)[cols]
        df.columns = ['coef','pval', 'fdr']
        # df.index = df['gene']
        return df

    def read_sc_dataset(dataset, method, cols):

        if method == 'quasiML':
            df = pd.read_csv(f'temp/hagai_{dataset}_result.csv', index_col=0)
        else:
            df = pd.read_csv(data_path +  'sc_rnaseq/results/{}_{}.csv'.format(dataset, method), index_col=0)[cols]
            df.columns = ['coef','pval', 'fdr']

        # if method == 'memento':
        #     df['fdr'] = -df['coef'].abs()
        # # df.index = df['gene']
        return df

    def read_sampled_dataset(dataset, method, cols):

        df = pd.read_csv(data_path +  'sc_rnaseq/results/{}_100_{}.csv'.format(dataset, method), index_col=0)[cols]
        df.columns = ['coef','pval', 'fdr']
        # df.index = df['gene']
        return df

    all_results = []
    for dataset in hagai_datasets:

        condition = dataset.split('-')[-1] + '4'

        bulk_results = [read_bulk_dataset(dataset, method, cols) for method, cols in bulk_methods]
        sc_results = [read_sc_dataset(dataset, method, cols) for method, cols in sc_methods]

        gene_list = [set(res.index) for res in sc_results] + [set(res.index) for res in bulk_results] 
        genes = list(functools.reduce(lambda x,y: x & y, gene_list))

        bulk_results = [res.loc[genes].sort_values('fdr') for res in bulk_results]
        sc_results = [res.loc[genes].sort_values('fdr') for res in sc_results]

        scores = [(sc_methods[idx][0], dataset, concordance_auc([b_res.index for b_res in bulk_results], res.index, k=100)) for idx, res in enumerate(sc_results)]
        all_results+=scores
        
    df = pd.DataFrame(all_results, columns=['name', 'dataset', 'auc'])
    
    return df.replace('quasiML', 'memento')


def canogamez_validation():
    
    data_path = '/data_volume/bulkrna/canogamez/'
    conversion = pd.read_csv('conversion.txt', sep='\t', header=None)
    conversion_dict = dict(zip(conversion.iloc[:, 0], conversion.iloc[:, 1]))
    
    def read_bulk_dataset(dataset, method, cols):

        df = pd.read_csv(data_path + 'bulk_results/{}_{}.csv'.format(dataset, method), index_col=0)
        df = df.rename(columns=dict(zip(cols,['logFC','PValue', 'FDR'])))
        df.index = [conversion_dict[x] for x in df.index]
        # df.index = df['gene']
        return df

    def read_sc_dataset(dataset, method, cols, trial):
        if method == 'quasiML':
            df = pd.read_csv(f'temp/canogamez_{dataset}_result.csv', index_col=0)
        else:
            df = pd.read_csv(data_path +  'sc_results/{}_{}_{}.csv'.format(dataset, trial, method), index_col=0)
        df = df.rename(columns=dict(zip(cols,['logFC','PValue', 'FDR'])))
            # df.index = df['gene']
        return df

    def read_sampled_dataset(dataset, method, cols, trial):

        df = pd.read_csv(data_path +  'sc_results/{}_{}_{}.csv'.format(dataset, trial, method), index_col=0)
        df = df.rename(columns=dict(zip(cols,['logFC','PValue', 'FDR'])))
        # df.index = df['gene']
        return df

    all_results = []
    trial = 1
    for dataset in canogamez_datasets:

        bulk_results = [read_bulk_dataset(dataset, method, cols) for method, cols in bulk_methods]

        sc_results = [read_sc_dataset(dataset, method, cols, 1) for method, cols in sc_methods]
        
        gene_list = [set(res.index) for res in sc_results] + [set(res.index) for res in bulk_results]
        genes = list(functools.reduce(lambda x,y: x & y, gene_list))

        bulk_results = [res.loc[genes].sort_values('FDR') for res in bulk_results]
        sc_results = [res.loc[genes].sort_values('FDR') for res in sc_results]

        scores = [(sc_methods[idx][0], dataset, trial, concordance_auc([b_res.index for b_res in bulk_results], res.index, k=100)) for idx, res in enumerate(sc_results)]
        all_results+=scores
        
    df = pd.DataFrame(all_results, columns=['name', 'dataset', 'trial','auc'])
    
    return df[['name', 'dataset','auc']].replace('quasiML', 'memento')
        

def lupus_bulk_validation():

    # Need custom concordance functions since no "ground truth" for lupus dataset
    def concordance_curve(ref1, ref2, ref3, ref4, x, k=300):
        overlap = []
        for i in range(1, k+1):

            a = len(set(x[:i]) & set(ref1[:i]))
            b = len(set(x[:i]) & set(ref2[:i]))
            c = len(set(x[:i]) & set(ref3[:i]))
            d = len(set(x[:i]) & set(ref4[:i]))
            overlap.append((a+b+c+d)/4)

        return np.arange(1, k+1), np.array(overlap)

    def concordance_auc(ref1, ref2, ref3, ref4, x, k=100):
        count = 0
        for i in range(1, k+1):

            a = len(set(x[:i]) & set(ref1[:i]))
            b = len(set(x[:i]) & set(ref2[:i]))
            c = len(set(x[:i]) & set(ref3[:i]))
            d = len(set(x[:i]) & set(ref4[:i]))
            count += (a+b+c+d)/4

        return count / (k*(k+1)/2)
    
    data_path = '/data_volume/bulkrna/lupus/'
    
    all_results = []
    numcells = 100
    for trial in range(50):        
        name_paths = [
            ('bulk_edger_lrt','T4_vs_cM.bulk.edger_lrt.{}.{}.csv'.format(numcells, trial), ['logFC','PValue', 'FDR']), 
            ('bulk_edger_qlft','T4_vs_cM.bulk.edger_qlft.{}.{}.csv'.format(numcells, trial), ['logFC','PValue', 'FDR']),
            ('bulk_deseq2_wald', 'T4_vs_cM.bulk.deseq2_wald.{}.{}.csv'.format(numcells, trial), ['log2FoldChange','pvalue', 'padj']),
            ('bulk_deseq2_lrt', 'T4_vs_cM.bulk.deseq2_lrt.{}.{}.csv'.format(numcells, trial), ['log2FoldChange','pvalue', 'padj']),
            ('edgeR','T4_vs_cM.pseudobulk.edger_lrt.{}.{}.csv'.format(numcells, trial), ['logFC','PValue', 'FDR']),
            # ('edgeR_qlft','T4_vs_cM.pseudobulk.edger_qlft.{}.{}.csv'.format(numcells, trial), ['logFC','PValue', 'FDR']),

            ('DESeq2','T4_vs_cM.pseudobulk.deseq2_wald.{}.{}.csv'.format(numcells, trial), ['log2FoldChange','pvalue', 'padj']),
            ('t-test','{}_{}_t.csv'.format(numcells, trial), ['logFC','PValue', 'FDR']),
            ('MWU','{}_{}_mwu.csv'.format(numcells, trial), ['logFC','PValue', 'FDR']),
            # ('MAST','T4_vs_cM.sc.MAST.{}.{}.csv'.format(numcells, trial), ['coef','Pr(>Chisq)', 'fdr']),
            ('quasiML', '{}_{}_quasiML.csv'.format(numcells, trial), ['coef', 'pval', 'fdr'])
        ]
        results = [pd.read_csv(data_path + path, index_col=0)[cols].rename(columns=dict(zip(cols,['logFC','PValue', 'FDR'])))  for name, path, cols in name_paths]
        gene_lists = [set(res.index) for res in results]
        genes = list(functools.reduce(lambda x,y: x & y, gene_lists))
        results = [res.loc[genes].sort_values('FDR') for res in results]

        scores = [
            (name_paths[idx+4][0].replace('_', '\n'), numcells, trial, concordance_auc(
                results[0].index, 
                results[1].index, 
                results[2].index,
                results[3].index,
                res.index)) for idx, res in enumerate(results[4:])]
        all_results+=scores

        curves = [
            concordance_curve(
                results[0].index, 
                results[1].index, 
                results[2].index,
                results[3].index,
                res.index) for res in results[4:]]
    
    df = pd.DataFrame(all_results, columns=['name', 'numcells','trials', 'auc']).replace('quasiML', 'memento')
    return df