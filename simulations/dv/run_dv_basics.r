suppressMessages(library(Seurat))
suppressMessages(library(SeuratDisk))

suppressMessages(library(cowplot))
suppressMessages(library(tidyverse))
suppressMessages(library(dplyr))
suppressMessages(library(BiocParallel))
suppressMessages(library(readr))
suppressMessages(library(Matrix))

suppressMessages(library(BASiCS))


get_chain <- function(seurat, cond) {
    dat <- subset(x=seurat, idents=cond)
    dat.sce <- SingleCellExperiment(
        assays=list(counts = GetAssayData(dat)),
        colData = data.frame(BatchInfo = dat$batch))

    Chain <- BASiCS_MCMC(
      dat.sce,
      N = 10000, Thin = 100, Burn = 1000, WithSpikes = FALSE, SubsetBy = 'cell',
      PrintProgress = FALSE, Regression = TRUE,Threads = getOption("Ncpus", 10),)
    return(Chain)
   }

setwd('/data_volume/memento/simulation/dv/')
Convert('high_expr_anndata_clean.h5ad', dest = "h5seurat", overwrite = TRUE, verbose = FALSE, misc=FALSE)
seurat <- LoadH5Seurat('high_expr_anndata_clean.h5seurat', meta.data=FALSE,misc=FALSE, verbose = FALSE)
df <- read.table('obs.csv', sep=',', header=TRUE)

seurat$condition <- df$condition
seurat$batch<- df$group
Idents(seurat) <- 'condition'

ctrl_chain <- get_chain(seurat, 'ctrl')
saveRDS(ctrl_chain, file = 'ctrl_chain.rds')
stim_chain <- get_chain(seurat, 'stim')
saveRDS(ctrl_chain, file = 'stim_chain.rds')


Test <- BASiCS_TestDE(
  Chain1 = ctrl_chain, Chain2 = stim_chain,
  GroupLabel1 = "ctrl", GroupLabel2 = "stim",
  EpsilonM = 0, EpsilonD =0,
  EpsilonR = 0.0,
  EFDR_M = 1, EFDR_D = 1,EFDR_R=NULL, k=1,ProbThresholdR=0,
  Offset = TRUE, PlotOffset = TRUE, Plot = TRUE
)

dv <- (Test@Results$ResDisp)@Table
write_csv(dv, file = 'dv_basics.csv')
