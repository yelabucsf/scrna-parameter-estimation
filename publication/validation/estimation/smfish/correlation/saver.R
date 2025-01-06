library('SAVER')
library(reshape2)

suppressMessages(library(Seurat))
suppressMessages(library(SeuratDisk))

NUM_TRIALS = 20
DATA_PATH <- '/home/ubuntu/Data/'

setwd(paste(DATA_PATH, 'smfish/correlation/', sep=''))


idxs <- c(26973,
 12715,
 13655,
 8976,
 11611,
 8827,
 15880,
 7778,
 12686,
 30953,
 28386,
 9236,
 3718,
 31243,
 30532,
 4360,
 30984,
 15086,
 3158,
 11397,
 11612)+1 #R is one indexed

for (trial in seq(0, NUM_TRIALS-1)){
    
    # if (trial == 0) { NUMBER_OF_CELLS <- c(500, 1000, 5000, 8000) }
    # else { NUMBER_OF_CELLS <- c(500, 1000) }
    if (trial == 0) { NUMBER_OF_CELLS <- c(5000) }
    else { NUMBER_OF_CELLS <- c(5000) }
    
    
    for (num_cell in NUMBER_OF_CELLS){
        
        print(paste(trial, num_cell, sep='-'))
            
        # Setup and run BASiCS
        fname = paste(num_cell, trial, sep='_')
        Convert(paste(fname, '.h5ad', sep=''), dest = "h5seurat", overwrite = TRUE, verbose = FALSE, misc=FALSE)
        seurat <- LoadH5Seurat(paste(fname, '.h5seurat', sep=''), meta.data=FALSE,misc=FALSE, verbose = FALSE)

        matrix = as.matrix(seurat$RNA$counts)
        colnames(matrix) <- colnames(seurat$RNA)
        rownames(matrix) <- rownames(seurat$RNA)

        saver_obj = saver(matrix, ncores=20)
        
#         sf <- colSums(matrix)/mean(colSums(matrix))

#         adj.vec <- rep(0, dim(matrix)[1])
#         for (i in 1:dim(matrix)[1]) {
#           adj.vec[i] <- 
#             sqrt(var(saver_obj$estimate[i, ]*sf, na.rm = TRUE)/
#                    (var(saver_obj$estimate[i, ]*sf, na.rm = TRUE) + 
#                       mean(saver_obj$se[i, ]^2*sf^2, na.rm = TRUE)))
#         }
            
#         scale.factor <- outer(adj.vec, adj.vec)

#         temp.cor <- cor( t(sweep(saver_obj$estimate, 2, sf, "*")))

        # cor1 <- temp.cor*scale.factor
        cor2 <- cor.genes(saver_obj)
            
        # write.csv(cor1[idxs, idxs], paste(fname, '_corr1.csv', sep='') )
        write.csv(cor2[idxs, idxs], paste(fname, '_corr2.csv', sep='') )
    }  
}
