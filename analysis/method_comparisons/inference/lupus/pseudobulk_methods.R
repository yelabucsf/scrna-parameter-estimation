# Pseudobulk methods for cell type comparison in lupus data

library(edgeR)
library(DESeq2)

data_path <- '/data_volume/memento/method_comparison/lupus/'

for (numcells in c(100, 150, 200)) {
	
	for (trial in seq(0,49)) {
		print(paste('working on', numcells, trial))

		### Read the data

		bulk = read.table(paste(data_path, 'T4_vs_cM.bulk.', numcells,'.',trial,'.csv', sep=''), sep=',', header=1, row.names=1)
		pseudobulk = read.table(paste(data_path, 'T4_vs_cM.pseudobulk.', numcells,'.',trial,'.csv', sep=''), sep=',', header=1, row.names=1)

		### Create the metadata


		num_inds = 4

		metadata = data.frame(name=colnames(bulk), ct=rep(c('CD4', 'CD14'), num_inds), ind=sapply(strsplit(colnames(bulk), '_'), tail, 1))


		### Run edgeR

		subject <- factor(metadata$ind)
		ct <- factor(metadata$ct)
		design <- model.matrix(~subject+ct)

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
			lrt <- glmLRT(fit,coef=num_inds+1)

			return(topTags(lrt, n=Inf))
		}
		edger_get_qlft <- function (y, design) {

			fit <- glmQLFit(y,design)
			qlft <- glmQLFTest(fit,coef=num_inds+1)

			return(topTags(qlft, n=Inf))
		}

		bulk_y <- edger_preprocess(bulk, design)
		bulk_result_lrt <- edger_get_lrt(bulk_y, design)
		bulk_result_qlft <- edger_get_qlft(bulk_y, design)

		write.csv(bulk_result_lrt, paste(data_path, 'T4_vs_cM.bulk.edger_lrt.', numcells,'.',trial,'.csv', sep=''))
		write.csv(bulk_result_qlft, paste(data_path, 'T4_vs_cM.bulk.edger_qlft.', numcells,'.',trial,'.csv', sep=''))

		pseudobulk_y <- edger_preprocess(pseudobulk, design)
		pseudobulk_lrt <- edger_get_lrt(pseudobulk_y, design)
		write.csv(pseudobulk_lrt, paste(data_path, 'T4_vs_cM.pseudobulk.edger_lrt.', numcells,'.',trial,'.csv', sep=''))

		dispersion_df = data.frame(gene=rownames(pseudobulk_y), dispersion=pseudobulk_y$tagwise.dispersion)

		write.csv(dispersion_df, paste(data_path, 'T4_vs_cM.dispersions.', numcells,'.',trial,'.csv', sep=''))

		### Run DESeq2

		run_deseq2_wald <- function(data, info) {
			dds <- DESeqDataSetFromMatrix(countData = round(data),
										  colData = info,
										  design= ~ ind + ct)
			levels(dds$ind) <- sub("\\.", "", levels(dds$ind))
			dds <- DESeq(dds)
			resultsNames(dds) # lists the coefficients
			res <- results(dds, name="ct_CD4_vs_CD14")

			return(res)
		}

		run_deseq2_lrt <- function(data, info) {
			dds <- DESeqDataSetFromMatrix(countData = round(data),
										  colData = info,
										  design= ~ ind + ct)
			levels(dds$ind) <- sub("\\.", "", levels(dds$ind))
			dds <- DESeq(dds, test="LRT", reduced=~ind)
			res <- results(dds, name="ct_CD4_vs_CD14")

			return(res)
		}

		bulk_deseq2_wald <- run_deseq2_wald(bulk, metadata)
		bulk_deseq2_lrt <- run_deseq2_lrt(bulk, metadata)
		write.csv(bulk_deseq2_lrt, paste(data_path, 'T4_vs_cM.bulk.deseq2_lrt.', numcells, '.', trial,'.csv', sep=''))
		write.csv(bulk_deseq2_wald, paste(data_path, 'T4_vs_cM.bulk.deseq2_wald.', numcells, '.', trial,'.csv', sep=''))

		pseudobulk_deseq2_wald <- run_deseq2_wald(pseudobulk, metadata)
		write.csv(pseudobulk_deseq2_wald, paste(data_path, 'T4_vs_cM.pseudobulk.deseq2_wald.', numcells,'.',trial,'.csv', sep=''))
	}
}
