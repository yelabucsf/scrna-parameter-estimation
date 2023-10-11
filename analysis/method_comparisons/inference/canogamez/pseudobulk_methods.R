# Pseudobulk methods for cell type comparison in lupus data

library(edgeR)
library(DESeq2)

data_path <- '/data_volume/memento/method_comparison/canogamez/'

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
    lrt <- glmLRT(fit,coef=num_inds)

    return(topTags(lrt, n=Inf))
}
edger_get_qlft <- function (y, design) {

    fit <- glmQLFit(y,design)
    qlft <- glmQLFTest(fit,coef=num_inds)

    return(topTags(qlft, n=Inf))
}


for (fname in files) {
    
	
    print(paste('working on', fname))

    ### Read the data

    bulk = read.table(paste(data_path, 'bulk/',fname, '_counts.csv', sep=''), sep=',',header=1, row.names=1)
    bulk_metadata = read.table(paste(data_path, 'bulk/',fname, '_metadata.csv', sep=''), sep=',',header=1, row.names=1)


    ### Run edgeR for bulk

    rep <- factor(bulk_metadata$donor_id)
    treatment <- factor(bulk_metadata$cytokine_condition)
    design <- model.matrix(~rep+treatment)
    num_inds <- dim(design)[2]

    bulk_y <- edger_preprocess(bulk, design)
    bulk_result_lrt <- edger_get_lrt(bulk_y, design)
    bulk_result_qlft <- edger_get_qlft(bulk_y, design)

    write.csv(bulk_result_lrt, paste(data_path, 'bulk_results/', fname, '_edger_lrt.csv', sep=''))
    write.csv(bulk_result_qlft, paste(data_path, 'bulk_results/', fname, '_edger_qlft.csv', sep=''))
    dispersion_df = data.frame(gene=rownames(bulk_y), dispersion=bulk_y$tagwise.dispersion)
    write.csv(dispersion_df, paste(data_path, 'bulk_results/', fname, '_bulk_dispersions.csv', sep=''))
    
    for (trial in seq(1)) {
        
        print(paste('trial', trial))
        pseudobulk = read.table(paste(data_path, 'pseudobulks/', fname,'_', trial,'.csv', sep=''), sep=',', header=1, row.names=1)
        pseudobulk_metadata =  read.table(paste(data_path, 'pseudobulks/', fname,'_meta_', trial,'.csv', sep=''), sep=',', header=1, row.names=1)

        ### Run edgeR for pseudobulk

		rep <- factor(pseudobulk_metadata$donor_id)
		treatment <- factor(pseudobulk_metadata$cytokine_condition)
		design <- model.matrix(~rep+treatment)
		num_inds <- dim(design)[2]
		
        pseudobulk_y <- edger_preprocess(pseudobulk, design)
        pseudobulk_lrt <- edger_get_lrt(pseudobulk_y, design)
        pseudobulk_qlft <- edger_get_qlft(pseudobulk_y, design)

        write.csv(pseudobulk_lrt, paste(data_path, 'sc_results/', fname, '_', trial,'_edger_lrt.csv', sep=''))
        write.csv(pseudobulk_qlft, paste(data_path, 'sc_results/', fname, '_', trial,'_edger_qlft.csv', sep=''))

        dispersion_df = data.frame(gene=rownames(pseudobulk_y), dispersion=pseudobulk_y$tagwise.dispersion)
        write.csv(dispersion_df, paste(data_path, 'sc_results/', fname, '_', trial,'_dispersions.csv', sep=''))

        ### Run DESeq2

        run_deseq2_wald <- function(data, info) {
            print(info)
            dds <- DESeqDataSetFromMatrix(countData = round(data),
                                          colData = info,
                                          design= ~ donor_id + cytokine_condition)
            levels(dds$donor_id) <- sub("\\.", "", levels(dds$donor_id))
            dds <- DESeq(dds)
            resultsNames(dds) # lists the coefficients
            print(paste('cytokine_condition',levels(dds$cytokine_condition)[2],'vs',levels(dds$cytokine_condition)[1], sep='_'))
            res <- results(dds, name=paste('cytokine_condition',levels(dds$cytokine_condition)[2],'vs',levels(dds$cytokine_condition)[1], sep='_'))

            return(res)
        }

        run_deseq2_lrt <- function(data, info) {
            dds <- DESeqDataSetFromMatrix(countData = round(data),
                                          colData = info,
                                          design= ~ donor_id + cytokine_condition)
            levels(dds$donor_id) <- sub("\\.", "", levels(dds$donor_id))
            dds <- DESeq(dds, test="LRT", reduced=~donor_id)
            res <- results(dds, name=paste('cytokine_condition',levels(dds$cytokine_condition)[2],'vs',levels(dds$cytokine_condition)[1], sep='_'))

            return(res)
        }

        bulk_deseq2_wald <- run_deseq2_wald(bulk, bulk_metadata)
        bulk_deseq2_lrt <- run_deseq2_lrt(bulk, bulk_metadata)
        write.csv(bulk_deseq2_lrt, paste(data_path, 'bulk_results/', fname,'_deseq2_lrt.csv', sep=''))
        write.csv(bulk_deseq2_wald, paste(data_path, 'bulk_results/', fname,'_deseq2_wald.csv', sep=''))

        pseudobulk_deseq2_wald <- run_deseq2_wald(pseudobulk, pseudobulk_metadata)
		pseudobulk_deseq2_lrt <- run_deseq2_lrt(pseudobulk, pseudobulk_metadata)
        write.csv(pseudobulk_deseq2_wald, paste(data_path, 'sc_results/', fname,'_',  trial,'_deseq2_wald.csv', sep=''))
		write.csv(pseudobulk_deseq2_lrt, paste(data_path, 'sc_results/', fname,'_',  trial,'_deseq2_lrt.csv', sep=''))
        
    }
	
}
