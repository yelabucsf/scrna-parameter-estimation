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

NUM_TRIALS = 100
# NUMBER_OF_CELLS = c(500, 1000, 5000, 8000)

get_chain <- function(seurat, cond) {
    dat <- subset(x=seurat, idents=cond)
    dat.sce <- SingleCellExperiment(
        assays=list(counts = GetAssayData(dat)),
        colData = data.frame(BatchInfo = dat$batch))

    Chain <- BASiCS_MCMC(
      dat.sce,
      N = 5000, Thin = 2, Burn = 1000, WithSpikes = FALSE, SubsetBy = 'cell',
      PrintProgress = FALSE, Regression = TRUE,Threads = getOption("Ncpus", 40),)
    return(Chain)
   }

setwd(paste(DATA_PATH, 'smfish/variance/', sep=''))

for (trial in seq(0, NUM_TRIALS-1)){
    
    if (trial == 0) { NUMBER_OF_CELLS <- c(500, 1000, 5000, 8000) }
    
    else { NUMBER_OF_CELLS <- c(500, 1000) }
    
    for (num_cell in NUMBER_OF_CELLS){
            


        # Setup and run BASiCS
        fname = paste(num_cell, trial, sep='_')
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

