# R
suppressPackageStartupMessages({
    library(ggplot2)
    library(limma)
    library(reshape2)
    library(data.table)

    library(MAST)
})

options(mc.cores = 30)

files = c(
    'Hagai2018_mouse-lps',
    'Hagai2018_mouse-pic',
    'Hagai2018_pig-lps',
    'Hagai2018_rabbit-lps',
    'Hagai2018_rat-lps',
    'Hagai2018_rat-pic'
)

data_path <- '/data_volume/memento/method_comparison/squair/'

for (fname in files) {
	
	stim = strsplit(fname,split='-')[[1]][2]
	
	if (grepl(stim, 'pic', fixed = TRUE)){
		stim = 'pic4'
	}
	if (grepl(stim, 'lps', fixed = TRUE)){
		stim = 'lps4'
	}
	
	print(paste('working on', fname))
	
	expr_fname <- sprintf('%ssc_rnaseq/h5Seurat/replicates/%s_all_expr.csv', data_path, fname)
	obs_fname <- sprintf('%ssc_rnaseq/h5Seurat/replicates/%s_all_obs.csv', data_path, fname)
	var_fname <- sprintf('%ssc_rnaseq/h5Seurat/replicates/%s_all_var.csv', data_path, fname)
	output_fname <- sprintf('%ssc_rnaseq/results/%s_all_MAST.csv', data_path, fname)

	expr = read.csv(expr_fname, row.names = 1)
	obs = read.csv(obs_fname, row.names = 1)
	var = read.csv(var_fname, row.names = 1)

	print('read the data')
	expr_norm<-log(t(apply(expr,1, function(x) x/sum(x)*10000))+1)

	scaRaw <- FromMatrix(t(expr_norm), obs, var)

	freq_expressed <- 0.005
	sca <- scaRaw
	expressed_genes <- freq(sca) > freq_expressed
	sca <- sca[expressed_genes,]

	cond<-factor(colData(sca)$label)
	colData(sca)$ind <- factor(colData(sca)$replicate)
	cond<-relevel(cond,stim)
	colData(sca)$condition<-cond
	print('running zlm')
	zlmCond <- zlm(~condition + ind, sca)

	summaryCond <- summary(zlmCond, doLRT=sprintf('condition%s', 'unst')) 

	summaryDt <- summaryCond$datatable
	fcHurdle <- merge(summaryDt[contrast==sprintf('condition%s', 'unst') & component=='H',.(primerid, `Pr(>Chisq)`)], #hurdle P values
						  summaryDt[contrast==sprintf('condition%s', 'unst') & component=='logFC', .(primerid, coef, ci.hi, ci.lo)], by='primerid') #logFC coefficients

	fcHurdle[,fdr:=p.adjust(`Pr(>Chisq)`, 'fdr')]

	write.csv(fcHurdle, output_fname, row.names = FALSE)
							   
	}