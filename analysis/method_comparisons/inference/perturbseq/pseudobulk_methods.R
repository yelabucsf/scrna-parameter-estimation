# Pseudobulk methods for cell type comparison in HBEC data

library(edgeR)
library(DESeq2)

data_path <-'/data_volume/memento/method_comparison/tfko140/'

guide_df = read.csv('guides.csv', header=TRUE)
mapping = lapply(with(unique(guide_df), split(guide1_cov, target)), as.list)


for (target in names(mapping)) {

	for (guide in mapping[[target]]) {


	print(paste('working on', target, guide))

		### Read the data
		pseudobulk = read.table(paste(data_path, 'tfko140.pseudobulk.',target,'.',guide,'.csv', sep=''), sep=',', header=1, row.names=1)

		### Create the metadata
		metadata = data.frame(
			name=colnames(pseudobulk), 
			stim=sapply(strsplit(colnames(pseudobulk), '_'), '[', 3),
			ind=sapply(strsplit(colnames(pseudobulk), '_'), '[', 4)
		)

		print(metadata)

		### Run edgeR
		subject <- factor(metadata$ind)
		stim <- factor(metadata$stim)
		design <- model.matrix(~subject+stim)

		edger_preprocess <- function (data, design) {
			y <- DGEList(counts=data)
			keep <- filterByExpr(y,min.count=1)
			y <- y[keep,,keep.lib.sizes=FALSE]
			y <- calcNormFactors(y)
			y <- estimateDisp(y,design)

			return(y)
		}

		edger_get_lrt <- function (y, design) {

			fit <- glmFit(y,design)
			lrt <- glmLRT(fit,coef=2+1)

			return(topTags(lrt, n=Inf))
		}
		edger_get_qlft <- function (y, design) {

			fit <- glmQLFit(y,design)
			qlft <- glmQLFTest(fit,coef=2+1)

			return(topTags(qlft, n=Inf))
		}

		pseudobulk_y <- edger_preprocess(pseudobulk, design)
		pseudobulk_lrt <- edger_get_lrt(pseudobulk_y, design)

		write.csv(pseudobulk_lrt, paste(data_path, 'tfko140.pseudobulk.edger_lrt.',target,'.',guide,'.csv', sep=''))

		dispersion_df = data.frame(gene=rownames(pseudobulk_y), dispersion=pseudobulk_y$tagwise.dispersion)

		write.csv(dispersion_df, paste(data_path, 'tfko140.dispersions.',target,'.',guide,'.csv', sep=''))

		### Run DESeq2

		run_deseq2_wald <- function(data, info) {
			dds <- DESeqDataSetFromMatrix(countData = round(data),
										  colData = info,
										  design= ~ ind + stim)
			levels(dds$ind) <- sub("\\.", "", levels(dds$ind))
			dds <- DESeq(dds)
			resultsNames(dds) # lists the coefficients
			res <- results(dds, name=paste("stim_",'T_vs_', 'F',sep=''))

			return(res)
		}

		run_deseq2_lrt <- function(data, info) {
			dds <- DESeqDataSetFromMatrix(countData = round(data),
										  colData = info,
										  design= ~ ind + stim)
			levels(dds$ind) <- sub("\\.", "", levels(dds$ind))
			dds <- DESeq(dds, test="LRT", reduced=~ind)
			res <- results(dds, name=paste("stim_",'T_vs_', 'F',sep=''))

			return(res)
		}

		pseudobulk_deseq2_wald <- run_deseq2_wald(pseudobulk, metadata)
		write.csv(pseudobulk_deseq2_wald, paste(data_path, 'tfko140.pseudobulk.deseq2_wald.',target,'.',guide,'.csv', sep=''))
		pseudobulk_deseq2_lrt <- run_deseq2_lrt(pseudobulk, metadata)
		write.csv(pseudobulk_deseq2_lrt, paste(data_path, 'tfko140.pseudobulk.deseq2_wald.',target,'.',guide,'.csv', sep=''))
	}
}