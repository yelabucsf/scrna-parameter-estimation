library(data.table)
library(qvalue)
library(bedr)
options(scipen = 999)
args=commandArgs(TRUE)

#res_raw=data.frame(fread(args[1]))
res_raw=data.frame(fread(args[1]))[-1]
#emp=fread(args[2])
#delta=fread(args[3])
out=args[2]
expr_genes=read.table(args[3],header=F)
print('gene')
print(head(expr_genes))
print(dim(expr_genes))
#print(head(emp))
#emp$FDR=qvalue(emp$Empirical)$qvalues
#thresh=max(emp$Pvalue[which(emp$FDR < 0.1)])

#run if not corrected by celltype
#fdr=qvalue(res_raw$pvalue)
#print(fdr)
#res_raw$FDR=fdr$qvalues

#filter FDR
#res=res_raw[res_raw$FDR <= 0.1,]
#snps gene statistic pvalue FDR beta qvalue



colnames(res_raw)=c("SNP","gene","statistic","p.value","FDR","beta")
#colnames(res_raw)=c("SNP","gene","statistic","p.value","FDR","beta","qvalue")
#res=res_raw
#filter for expressed genes
print(head(res_raw))
print(dim(res_raw))
res_expr=res_raw[res_raw$gene %in% expr_genes$V1,]
print(dim((res_expr)))
res_noexpr=res_raw[!(res_raw$gene %in% expr_genes$V1),]

print(dim(res_expr))
print(head(res_expr))
print(head(res_noexpr))

num_no=length(res_noexpr$gene)
nullp=runif(num_no, min = 0, max = 1)
res_noexpr$p.value=nullp
res=rbind(res_expr,res_noexpr)
print(head(res_expr))
print(dim(res_expr))
print(dim(res))
print(tail(res))
#atac=fread('/ye/yelabstore2/10x.lupus/process.scRNAseq/eqtls/S9_lineage_groups.txt.gz')
atac=fread('simple_sorted_atac_lineage_groups.txt.gz')
#names(atac)=c('chr', 'start', 'end', 'group')
#atac.bed=data.frame(chr=atac$chr, start=atac$start, end=atac$stop, group=atac$group)
atac.groups=names(table(atac$group))
#atac.sorted=bedr.sort.region(atac$chr,atac$start,atac$stop)

res$chr=paste('chr', sapply(strsplit(res$SNP, ':'), '[', 1), sep='')
res$pos=as.numeric(sapply(strsplit(res$SNP, ':'), '[', 2))

#res$chr=paste('chr', sapply(strsplit(res$snp, ':'), '[', 1), sep='')
#res$pos=as.numeric(sapply(strsplit(res$snp, ':'), '[', 2))
res$id=paste(res$chr, ':', res$pos -1, '-' ,res$pos, sep='')
res.bed.sorted=bedr.sort.region(unique(paste(res$chr, ':', res$pos -1, '-' ,res$pos, sep='')))

#res$chr=paste('chr', sapply(strsplit(res$SNP, ':'), '[', 1), sep='')
#res$pos=as.numeric(sapply(strsplit(res$SNP, ':'), '[', 2))
#res$id=paste(res$chr, ':', res$pos -1, '-' ,res$pos, sep='')
#res.bed.sorted=bedr.sort.region(unique(paste(res$chr, ':', res$pos -1, '-' ,res$pos, sep='')))

all.df=NULL
for(g in atac.groups){
    print(g)
    peaks=atac[which(atac$group == g), ]
    peaks=paste(peaks$chr, ':', peaks$start, '-', peaks$stop, sep='')
    overlap.sig.snps=bedr(input=list(a=res.bed.sorted, b=peaks), method='intersect')
    in_peak=as.numeric(res.bed.sorted %in% overlap.sig.snps)
    pvals=res$"p.value"[match(res.bed.sorted, res$id)]
    #pvals=res$"pvalue"[match(res.bed.sorted, res$id)]
    #pvals=res$"p-value"[match(res.bed.sorted, res$id)]
    mw_res=wilcox.test(pvals ~ in_peak, alternative='greater')
    df=data.frame(W=mw_res$statistic, pval=mw_res$p.value, group=g)
    all.df=rbind(all.df, df)   
}

write.table(all.df, file=out, row.names=F, col.names=T, quote=F, sep='\t')


