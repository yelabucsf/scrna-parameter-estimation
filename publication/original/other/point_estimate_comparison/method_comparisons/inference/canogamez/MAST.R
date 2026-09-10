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
 'CD4_Memory-Th0',
 'CD4_Memory-Th2',
 'CD4_Memory-Th17',
 'CD4_Memory-iTreg',
 'CD4_Naive-Th0',
 'CD4_Naive-Th2',
 'CD4_Naive-Th17',
 'CD4_Naive-iTreg'
)

data_path <- '/data_volume/memento/method_comparison/canogamez/'

for (fname in files) {
	
	stim = strsplit(fname,split='-')[[1]][2]
	
	print(paste('working on', fname))
	
	expr_fname <- sprintf('%ssingle_cell/%s_1_expr.csv', data_path, fname)
	obs_fname <- sprintf('%ssingle_cell/%s_1_obs.csv', data_path, fname)
	var_fname <- sprintf('%ssingle_cell/%s_1_var.csv', data_path, fname)
	output_fname <- sprintf('%ssc_results/%s_1_MAST.csv', data_path, fname)

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

	cond<-factor(colData(sca)$cytokine.condition)
	colData(sca)$ind <- factor(colData(sca)$donor.id)
	cond<-relevel(cond,stim)
	colData(sca)$condition<-cond
	print('running zlm')
	zlmCond <- zlm(~condition + ind, sca)

	summaryCond <- summary(zlmCond, doLRT=sprintf('condition%s', 'UNS')) 

	summaryDt <- summaryCond$datatable
	fcHurdle <- merge(summaryDt[contrast==sprintf('condition%s', 'UNS') & component=='H',.(primerid, `Pr(>Chisq)`)], #hurdle P values
						  summaryDt[contrast==sprintf('condition%s', 'UNS') & component=='logFC', .(primerid, coef, ci.hi, ci.lo)], by='primerid') #logFC coefficients

	fcHurdle[,fdr:=p.adjust(`Pr(>Chisq)`, 'fdr')]

	write.csv(fcHurdle, output_fname, row.names = FALSE)
							   
	}