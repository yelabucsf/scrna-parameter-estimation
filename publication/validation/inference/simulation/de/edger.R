suppressMessages(library(edgeR))

num_groups_per_condition <- 2

data = read.table('/data_volume/memento/simulation/de/pseudobulks.csv', sep=',', header=1, row.names=1)
# groups <- c( rep('ctrl', num_groups_per_condition), rep('stim', num_groups_per_condition))
groups <- rep(c('ctrl', 'stim'), num_groups_per_condition)
replicates <- c('A', 'A', 'B', 'B')

y <- DGEList(counts=data,group=groups)
keep <- filterByExpr(y,min.count=1)
y <- y[keep,,keep.lib.sizes=FALSE]
y <- calcNormFactors(y)
design <- model.matrix(~groups+replicates)
y <- estimateDisp(y,design)

fit <- glmFit(y,design)
lrt <- glmLRT(fit,coef=2)
qfit <- glmQLFit(y,design)
qlft <- glmQLFTest(qfit,coef=2)

write.csv(topTags(lrt, n=Inf), '/data_volume/memento/simulation/de/edger_lrt.csv',)
write.csv(topTags(qlft, n=Inf), '/data_volume/memento/simulation/de/edger_qlft.csv',)

print('edgeR successful')
