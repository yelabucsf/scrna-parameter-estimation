library(data.table)
library(qvalue)
library(bedr)
options(scipen = 999)

for (method in c('memento', 'mateqtl')){
    for (pop in c('asian', 'eur')){
        for (ct in c('T4', 'cM', 'ncM', 'T8', 'B', 'NK')){

            # args=commandArgs(TRUE)
            if (method == 'memento'){
                res_raw=data.frame(fread(paste('/data_volume/memento/lupus/full_analysis/memento/100kb/', pop, '_',ct, '.csv', sep='')))
				colnames(res_raw)=c('SNP', 'gene','statistic', 'p.value', 'FDR',  'beta')
            }
            if (method == 'mateqtl'){
                res_raw=data.frame(fread(paste('/data_volume/memento/lupus/full_analysis/mateqtl/outputs/', pop, '_',ct, '_all_hg19.csv', sep='')))
				colnames(res_raw)=c('SNP', 'gene', 'beta','statistic', 'p.value', 'FDR')
            }

            res=res_raw
            atac=fread('/data_volume/memento/lupus/atac_enrichment/sorted_simple_atac_lineage_groups3.bed.gz')
			
            atac.groups=names(table(atac$group))

            res$chr=paste('chr', sapply(strsplit(res$SNP, ':'), '[', 1), sep='')
            res$pos=as.numeric(sapply(strsplit(res$SNP, ':'), '[', 2))

            res$id=paste(res$chr, ':', res$pos -1, '-' ,res$pos, sep='')
            res.bed.sorted=bedr.sort.region(unique(paste(res$chr, ':', res$pos -1, '-' ,res$pos, sep='')))

            for(g in atac.groups){
                print(paste(method, pop, ct, g, sep='-'))
                peaks=atac[which(atac$group == g), ]
                peaks=paste(peaks$chr, ':', peaks$start, '-', peaks$stop, sep='')
                overlap.sig.snps=bedr(input=list(a=res.bed.sorted, b=peaks), method='intersect')
                in_peak=as.numeric(res.bed.sorted %in% overlap.sig.snps)
				idx = match(res.bed.sorted, res$id)
                pv=res$p.value[idx]
				snp=res$SNP[idx]
				gene=res$gene[idx]
				beta=res$beta[idx]
				stat=res$statistic[idx]
                write.table(
                    cbind(snp, gene, beta, stat, pv, in_peak),
                    file=paste('/data_volume/memento/lupus/atac_enrichment/peaks/', method, '/',pop, '_',ct, '_', g,'.txt', sep=''), 
                    row.names=F, col.names=T, quote=F, sep='\t')

            }
        }
    }
}