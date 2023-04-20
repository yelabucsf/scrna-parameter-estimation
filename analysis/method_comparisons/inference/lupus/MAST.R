# R
suppressPackageStartupMessages({
    library(ggplot2)
    library(limma)
    library(reshape2)
    library(data.table)

    library(MAST)
})

options(mc.cores = 2)


data_path <- '/data_volume/memento/method_comparison/lupus/'
for (numcell in c(200)) {
	
	for (trial in seq(0,49)) {
		print(paste('working on', numcell, trial))

		expr_fname <- sprintf('%sT4_vs_cM.single_cell.%s.%s.expr.csv', data_path, numcell, trial)
		obs_fname <- sprintf('%sT4_vs_cM.single_cell.%s.%s.obs.csv', data_path, numcell, trial)
		var_fname <- sprintf('%sT4_vs_cM.single_cell.%s.%s.var.csv', data_path, numcell, trial)
		output_fname <- sprintf('%sT4_vs_cM.sc.MAST.%s.%s.csv', data_path, numcell, trial)

		expr = read.csv(expr_fname, row.names = 1)
		obs = read.csv(obs_fname, row.names = 1)
		var = read.csv(var_fname, row.names = 1)

		expr_norm<-log(t(apply(expr,1, function(x) x/sum(x)*10000))+1)

		scaRaw <- FromMatrix(t(expr_norm), obs, var)


		freq_expressed <- 0.005
		sca <- scaRaw
		expressed_genes <- freq(sca) > freq_expressed
		sca <- sca[expressed_genes,]
							   
		cond<-factor(colData(sca)$cg_cov)
		colData(sca)$ind <- factor(colData(sca)$ind)
		cond<-relevel(cond,"cM")
		colData(sca)$condition<-cond
		zlmCond <- zlm(~condition + ind, sca)

		summaryCond <- summary(zlmCond, doLRT='conditionT4') 

		summaryDt <- summaryCond$datatable
		fcHurdle <- merge(summaryDt[contrast=='conditionT4' & component=='H',.(primerid, `Pr(>Chisq)`)], #hurdle P values
							  summaryDt[contrast=='conditionT4' & component=='logFC', .(primerid, coef, ci.hi, ci.lo)], by='primerid') #logFC coefficients

		fcHurdle[,fdr:=p.adjust(`Pr(>Chisq)`, 'fdr')]

		write.csv(fcHurdle, output_fname, row.names = FALSE)
							   
		}
	}