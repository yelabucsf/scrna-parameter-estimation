# Pseudobulk methods for cell type comparison in lupus data

suppressMessages(library(edgeR))
suppressMessages(library(DESeq2))

data_path <- '/data_volume/memento/hagai/'

files = c(
    'Hagai2018_mouse-lps',
    'Hagai2018_mouse-pic',
    'Hagai2018_pig-lps',
    'Hagai2018_rabbit-lps',
    'Hagai2018_rat-lps',
    'Hagai2018_rat-pic'
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
    lrt <- glmLRT(fit,coef=num_inds+1)

    return(topTags(lrt, n=Inf))
}
edger_get_qlft <- function (y, design) {

    fit <- glmQLFit(y,design)
    qlft <- glmQLFTest(fit,coef=num_inds+1)

    return(topTags(qlft, n=Inf))
}


for (fname in files) {
    
	
    print(paste('working on', fname))

    ### Read the data

    bulk_obj = readRDS(paste(data_path, 'bulk_rnaseq/rds/',fname, '.rds', sep=''))
    bulk = bulk_obj$assay
    bulk_metadata = bulk_obj$meta
    
    num_inds <- dim(bulk)[2]/2

    ### Run edgeR for bulk

    rep <- factor(bulk_metadata$replicate)
    treatment <- factor(bulk_metadata$label)
    design <- model.matrix(~rep+treatment)

    bulk_y <- edger_preprocess(bulk, design)
    bulk_result_lrt <- edger_get_lrt(bulk_y, design)
    bulk_result_qlft <- edger_get_qlft(bulk_y, design)

    write.csv(bulk_result_lrt, paste(data_path, 'bulk_rnaseq/results/', fname, '_edger_lrt.csv', sep=''))
    write.csv(bulk_result_qlft, paste(data_path, 'bulk_rnaseq/results/', fname, '_edger_qlft.csv', sep=''))
    dispersion_df = data.frame(gene=rownames(bulk_y), dispersion=bulk_y$tagwise.dispersion)
    write.csv(dispersion_df, paste(data_path, 'bulk_rnaseq/results/', fname, '_bulk_dispersions.csv', sep=''))
    
    for (trial in c('', '_shuffled')) {
        
        pseudobulk = read.table(paste(data_path, 'sc_rnaseq/pseudobulks/', fname,trial,'.csv', sep=''), sep=',', header=1, row.names=1)

        pseudobulk_metadata = data.frame(
            name=colnames(pseudobulk), 
            label=sapply(strsplit(colnames(pseudobulk), '_'), head, 1), 
            replicate=sapply(strsplit(colnames(pseudobulk), '_'), tail, 1))

        ### Run edgeR for pseudobulk

        rep <- factor(pseudobulk_metadata$replicate)
        treatment <- factor(pseudobulk_metadata$label)
        design <- model.matrix(~rep+treatment)
        pseudobulk_y <- edger_preprocess(pseudobulk, design)
        pseudobulk_lrt <- edger_get_lrt(pseudobulk_y, design)
        pseudobulk_qlft <- edger_get_qlft(pseudobulk_y, design)

        write.csv(pseudobulk_lrt, paste(data_path, 'sc_rnaseq/results/', fname, trial,'_edger_lrt.csv', sep=''))
        write.csv(pseudobulk_qlft, paste(data_path, 'sc_rnaseq/results/', fname, trial,'_edger_qlft.csv', sep=''))

        dispersion_df = data.frame(gene=rownames(pseudobulk_y), dispersion=pseudobulk_y$tagwise.dispersion)
        write.csv(dispersion_df, paste(data_path, 'sc_rnaseq/results/', fname, trial,'_dispersions.csv', sep=''))

        ### Run DESeq2

        run_deseq2_wald <- function(data, info) {
            print(info)
            dds <- DESeqDataSetFromMatrix(countData = round(data),
                                          colData = info,
                                          design= ~ replicate + label)
            levels(dds$replicate) <- sub("\\.", "", levels(dds$replicate))
            dds <- DESeq(dds)
            resultsNames(dds) # lists the coefficients
            print(paste('label',levels(dds$label)[2],'vs',levels(dds$label)[1], sep='_'))
            res <- results(dds, name=paste('label',levels(dds$label)[2],'vs',levels(dds$label)[1], sep='_'))

            return(res)
        }

        run_deseq2_lrt <- function(data, info) {
            dds <- DESeqDataSetFromMatrix(countData = round(data),
                                          colData = info,
                                          design= ~ replicate + label)
            levels(dds$replicate) <- sub("\\.", "", levels(dds$replicate))
            dds <- DESeq(dds, test="LRT", reduced=~replicate)
            res <- results(dds, name=paste('label',levels(dds$label)[2],'vs',levels(dds$label)[1], sep='_'))

            return(res)
        }

        bulk_deseq2_wald <- run_deseq2_wald(bulk, bulk_metadata)
        bulk_deseq2_lrt <- run_deseq2_lrt(bulk, bulk_metadata)
        write.csv(bulk_deseq2_lrt, paste(data_path, 'bulk_rnaseq/results/', fname, trial,'_deseq2_lrt.csv', sep=''))
        write.csv(bulk_deseq2_wald, paste(data_path, 'bulk_rnaseq/results/', fname, trial,'_deseq2_wald.csv', sep=''))

        pseudobulk_deseq2_wald <- run_deseq2_wald(pseudobulk, pseudobulk_metadata)
		pseudobulk_deseq2_lrt <- run_deseq2_lrt(pseudobulk, pseudobulk_metadata)
        write.csv(pseudobulk_deseq2_wald, paste(data_path, 'sc_rnaseq/results/', fname, trial,'_deseq2_wald.csv', sep=''))
		write.csv(pseudobulk_deseq2_lrt, paste(data_path, 'sc_rnaseq/results/', fname, trial,'_deseq2_lrt.csv', sep=''))
        
    }
	
}
