"""Figure 4D - how each regulator correlates with its own targets in wild-type cells.

Port of publication/original/perturbseq/interaction.ipynb cells 25-32.

A regulator is called an activator or a repressor by the sign of its knockout effect:
if knocking the regulator out *raises* a gene (de_coef > 0) it was repressing it. The
published panel shows the activators, ordered by mean correlation, with each
regulator-target pair coloured by the sign of its wild-type correlation.

`2d/wt_one_sample.csv` supplies those correlations directly -- its gene_1 column is the
regulator (all 47 are among the selected regulators) and gene_2 the target.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

import config
import perturbseq_data

WT_CORRELATIONS = config.FIGURE4_DATA + 'panelBCD_effects/wt_one_sample.csv'
DE_FDR = 0.05


def regulator_target_correlations():
    correlations = pd.read_csv(WT_CORRELATIONS)
    per_guide = perturbseq_data.differential_genes()
    per_guide['regulator'] = per_guide['tx'].apply(config.guide_to_gene)

    significant = per_guide.query('de_fdr < @DE_FDR')[['regulator', 'gene', 'de_coef']]
    merged = correlations.merge(
        significant, left_on=['gene_1', 'gene_2'], right_on=['regulator', 'gene'])
    # A knockout that raises the gene means the regulator was repressing it.
    merged['type'] = merged['de_coef'].apply(lambda x: 'repressor' if x > 0 else 'activator')
    return merged


def binomial_test(correlations, kind, positive):
    subset = correlations.query('type == @kind')
    per_regulator = subset.groupby('gene_1')['corr_coef'].mean()
    hits = (per_regulator > 0).sum() if positive else (per_regulator < 0).sum()
    total = per_regulator.shape[0]
    pvalue = stats.binomtest(hits, total).pvalue
    direction = 'positively' if positive else 'negatively'
    print(f'  {kind}s {direction} correlated with targets: {hits}/{total}, binomial p={pvalue:.2e}')
    return pvalue


def main():
    config.set_style()
    correlations = regulator_target_correlations()
    print(f'{correlations.shape[0]} regulator-target pairs, '
          f'{correlations.gene_1.nunique()} regulators')
    binomial_test(correlations, 'activator', positive=True)
    binomial_test(correlations, 'repressor', positive=False)

    mean_corr = (correlations.groupby(['gene_1', 'type'])['corr_coef']
                 .mean().reset_index(name='mean_corr'))
    correlations = correlations.merge(mean_corr, on=['gene_1', 'type'])

    activators = (correlations.query('type == "activator"')
                  .sort_values('mean_corr', ascending=False))
    order = activators['gene_1'].drop_duplicates().tolist()

    fig, ax = plt.subplots(figsize=(9, 2.4))
    sns.boxplot(y='corr_coef', x='gene_1', data=activators, order=order,
                fliersize=0, color='silver', ax=ax)
    sns.stripplot(y='corr_coef', x='gene_1', data=activators.query('corr_coef > 0'),
                  order=order, s=2, color='blue', ax=ax)
    sns.stripplot(y='corr_coef', x='gene_1', data=activators.query('corr_coef < 0'),
                  order=order, s=2, color='red', ax=ax)
    ax.axhline(0, linestyle='--', color='k', lw=1)
    ax.set_xlabel('Transcriptional regulators')
    ax.set_ylabel('Regulator-DEG\ncorrelations in WT')
    ax.tick_params(axis='x', rotation=90, labelsize=6)

    fig.savefig(config.figure_path('figure4D.pdf'), bbox_inches='tight')
    fig.savefig(config.figure_path('figure4D.png'), bbox_inches='tight', dpi=300)
    correlations.to_csv(config.intermediate_path('regulator_target_corr.csv'), index=False)
    print(f'wrote figure4D ({len(order)} activators)')


if __name__ == '__main__':
    main()
