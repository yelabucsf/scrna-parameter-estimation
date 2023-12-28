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
CAPTURE_EFFICIENCIES = c(0.05, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1)
NUMBER_OF_CELLS = c(50, 100, 500)

get_chain <- function(seurat, cond) {
    dat <- subset(x=seurat, idents=cond)
    dat.sce <- SingleCellExperiment(
        assays=list(counts = GetAssayData(dat)),
        colData = data.frame(BatchInfo = dat$batch))

    Chain <- BASiCS_MCMC(
      dat.sce,
      N = 5000, Thin = 2, Burn = 1000, WithSpikes = FALSE, SubsetBy = 'cell',
      PrintProgress = FALSE, Regression = TRUE,Threads = getOption("Ncpus", 5),)
    return(Chain)
   }

setwd(paste(DATA_PATH, 'simulation/variance/', sep=''))

for (num_cell in NUMBER_OF_CELLS){
    
    for (q in CAPTURE_EFFICIENCIES){
        
        for (trial in seq(0, NUM_TRIALS-1)){
            
            # Setup and run BASiCS
            fname = paste(num_cell, q, trial, sep='_')
            Convert(paste(fname, '.h5ad', sep=''), dest = "h5seurat", overwrite = TRUE, verbose = FALSE, misc=FALSE)
            seurat <- LoadH5Seurat(paste(fname, '.h5seurat', sep=''), meta.data=FALSE,misc=FALSE, verbose = FALSE)
            seurat$condition<- 'const'
            seurat$batch <- sample( c('A','B'), num_cell, replace=TRUE)
            Idents(seurat) <- 'condition'
            chain <- get_chain(seurat, 'const')
            
            # Extract parameters and save
            mu_values <-  as.data.frame(displayChainBASiCS(chain, Param ='mu' ))
            delta_values <- as.data.frame(displayChainBASiCS(chain, Param ='delta' ))

            mu <- colMeans(mu_values)
            delta <- colMeans(delta_values)

            parameters = as.data.frame(cbind(mu, delta))

            parameters$variance = parameters$mu + parameters$delta*parameters$mu**2

            write.csv(parameters, paste(fname, 'parameters.csv', sep='_'))
        }  
    }
}

