suppressMessages(library(Seurat))
suppressMessages(library(SeuratDisk))

suppressMessages(library(cowplot))
# suppressMessages(library(tidyverse))
suppressMessages(library(dplyr))
suppressMessages(library(BiocParallel))
suppressMessages(library(readr))
suppressMessages(library(Matrix))

suppressMessages(library(BASiCS))

DATA_PATH <- '/home/ubuntu/Data/'
NUM_TRIALS = 20
CAPTURE_EFFICIENCIES = c(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1)
NUMBER_OF_CELLS = c(10, 20, 30, 40, 50, 100, 200)

get_chain <- function(seurat, cond) {
    dat <- subset(x=seurat, idents=cond)
    dat.sce <- SingleCellExperiment(
        assays=list(counts = GetAssayData(dat)),
        colData = data.frame(BatchInfo = dat$batch))

    Chain <- BASiCS_MCMC(
      dat.sce,
      N = 1000, Thin = 2, Burn = 100, WithSpikes = FALSE, SubsetBy = 'cell',
      PrintProgress = FALSE, Regression = TRUE,Threads = getOption("Ncpus", 5),)
    return(Chain)
   }

setwd(paste(DATA_PATH, 'simulation/variance/', sep=''))

for (num_cell in NUMBER_OF_CELLS){
    
    for (q in CAPTURE_EFFICIENCIES){
        
        for (trial in seq(0, NUM_TRIALS-1)){
            fname = paste(num_cell, q, trial, sep='_')
            Convert(paste(fname, '.h5ad', sep=''), dest = "h5seurat", overwrite = TRUE, verbose = FALSE, misc=FALSE)
            seurat <- LoadH5Seurat(paste(fname, '.h5seurat', sep=''), meta.data=FALSE,misc=FALSE, verbose = FALSE)
            seurat$condition<- 'const'
            seurat$batch <- 'same'
            Idents(seurat) <- 'condition'
            chain <- get_chain(seurat, 'const')
            saveRDS(chain, file = paste(fname, 'chain.rds', sep='_'))       
        }  
    }
}

