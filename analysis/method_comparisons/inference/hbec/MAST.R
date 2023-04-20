# R
suppressPackageStartupMessages({
    library(ggplot2)
    library(limma)
    library(reshape2)
    library(data.table)

    library(MAST)
})

options(mc.cores = 2)


data_path <-'/data_volume/memento/method_comparison/hbec/'
# for (ct in c('BC', 'B', 'C')) {
	
# 	for (tp in c('3', '6', '9', '24', '48')) {
		
# 		for (condition in c('alpha', 'beta')) {
# 			print(paste('working on', ct, tp, condition))

# 			expr_fname <- sprintf('%shbec.single_cell.%s.%s.%s.expr.csv', data_path, ct, tp, condition)
# 			obs_fname <- sprintf('%shbec.single_cell.%s.%s.%s.obs.csv', data_path, ct, tp, condition)
# 			var_fname <- sprintf('%shbec.single_cell.%s.%s.%s.var.csv', data_path, ct, tp, condition)
# 			output_fname <- sprintf('%shbec.sc.MAST.%s.%s.%s.csv', data_path, ct, tp, condition)

# 			expr = read.csv(expr_fname, row.names = 1)
# 			obs = read.csv(obs_fname, row.names = 1)
# 			var = read.csv(var_fname, row.names = 1)

# 			expr_norm<-log(t(apply(expr,1, function(x) x/sum(x)*10000))+1)

# 			scaRaw <- FromMatrix(t(expr_norm), obs, var)


# 			freq_expressed <- 0.005
# 			sca <- scaRaw
# 			expressed_genes <- freq(sca) > freq_expressed
# 			sca <- sca[expressed_genes,]

# 			cond<-factor(colData(sca)$stim)
# 			colData(sca)$donor <- factor(colData(sca)$donor)
# 			cond<-relevel(cond,condition)
# 			colData(sca)$condition<-cond
# 			zlmCond <- zlm(~condition + donor, sca)

# 			summaryCond <- summary(zlmCond, doLRT='conditioncontrol')

# 			summaryDt <- summaryCond$datatable
# 			fcHurdle <- merge(summaryDt[contrast=='conditioncontrol' & component=='H',.(primerid, `Pr(>Chisq)`)], #hurdle P values
# 								  summaryDt[contrast=='conditioncontrol' & component=='logFC', .(primerid, coef, ci.hi, ci.lo)], by='primerid') #logFC coefficients

# 			fcHurdle[,fdr:=p.adjust(`Pr(>Chisq)`, 'fdr')]

# 			write.csv(fcHurdle, output_fname, row.names = FALSE)
# 			}
							   
# 		}
# 	}
								   
	
for (tp in c('9', '24', '48')) {

	for (condition in c('alpha', 'beta')) {
		print(paste('working on',tp, condition))

		expr_fname <- sprintf('%shbec.single_cell.%s.%s.expr.csv', data_path,  tp, condition)
		obs_fname <- sprintf('%shbec.single_cell.%s.%s.obs.csv', data_path,  tp, condition)
		var_fname <- sprintf('%shbec.single_cell.%s.%s.var.csv', data_path,  tp, condition)
		output_fname <- sprintf('%shbec.sc.MAST.%s.%s.csv', data_path,  tp, condition)

		expr = read.csv(expr_fname, row.names = 1)
		obs = read.csv(obs_fname, row.names = 1)
		var = read.csv(var_fname, row.names = 1)

		expr_norm<-log(t(apply(expr,1, function(x) x/sum(x)*10000))+1)

		scaRaw <- FromMatrix(t(expr_norm), obs, var)


		freq_expressed <- 0.005
		sca <- scaRaw
		expressed_genes <- freq(sca) > freq_expressed
		sca <- sca[expressed_genes,]

		cond<-factor(colData(sca)$stim)
		colData(sca)$donor <- factor(colData(sca)$donor)
		colData(sca)$ct <- factor(colData(sca)$ct)
		cond<-relevel(cond,condition)
		colData(sca)$condition<-cond
		zlmCond <- zlm(~condition + donor + ct, sca)

		summaryCond <- summary(zlmCond, doLRT='conditioncontrol')

		summaryDt <- summaryCond$datatable
		fcHurdle <- merge(summaryDt[contrast=='conditioncontrol' & component=='H',.(primerid, `Pr(>Chisq)`)], #hurdle P values
							  summaryDt[contrast=='conditioncontrol' & component=='logFC', .(primerid, coef, ci.hi, ci.lo)], by='primerid') #logFC coefficients

		fcHurdle[,fdr:=p.adjust(`Pr(>Chisq)`, 'fdr')]

		write.csv(fcHurdle, output_fname, row.names = FALSE)
		}

	}