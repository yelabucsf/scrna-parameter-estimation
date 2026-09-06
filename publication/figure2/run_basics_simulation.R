# Figure 2A (variability) - BASiCS arm of the estimator simulation.
#
# Reimplementation of publication/validation/estimation/simulation/variance/run_basics.r.
# Two changes from the original:
#   * counts are read from MatrixMarket rather than converted h5ad, so this needs only
#     BASiCS and SingleCellExperiment -- no Seurat, no SeuratDisk (whose h5ad Convert
#     step is the most fragile part of the original pipeline).
#   * only the (num_cell, q) slice the published panel is drawn at is scored, which is
#     100 MCMC runs rather than the 480 the original loops over.
#
# Inputs come from `panel_a_run_simulations.py variance --dump-basics-inputs`.
# Writes {num_cell}_{q}_{trial}_parameters.csv with mu, delta and variance, matching
# what the original wrote and what panel_a_plot.py reads.

suppressMessages(library(BASiCS))
suppressMessages(library(SingleCellExperiment))
suppressMessages(library(Matrix))

DATA_PATH <- Sys.getenv("MEMENTO_DATA_PATH", "/memento_data/")
WORK_DIR <- paste0(DATA_PATH, "simulation/variance/")

NUM_CELL <- 100
CAPTURE_EFFICIENCIES <- c(0.05, 0.1, 0.2, 0.3, 0.5)
TRIALS <- seq(0, 19)

# Matches the original: N=5000 iterations, thinned by 2, 1000 burn-in, no spike-ins,
# regression on the mean-dispersion trend.
MCMC_N <- 5000
MCMC_THIN <- 2
MCMC_BURN <- 1000

# Usage: Rscript run_basics_simulation.R [capture_efficiency] [threads]
# Passing a single capture efficiency lets the 100 runs be sharded across processes;
# each writes a disjoint set of files, and completed files are skipped on restart.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) CAPTURE_EFFICIENCIES <- as.numeric(args[1])
THREADS <- if (length(args) > 1) as.integer(args[2]) else 4

setwd(WORK_DIR)

for (q in CAPTURE_EFFICIENCIES) {
  for (trial in TRIALS) {
    fname <- paste(NUM_CELL, q, trial, sep = "_")
    outfile <- paste0(fname, "_parameters.csv")
    if (file.exists(outfile)) {
      cat("skip", fname, "\n")
      next
    }

    counts <- as.matrix(readMM(paste0(fname, "_counts.mtx")))
    genes <- read.csv(paste0(fname, "_genes.csv"))$gene_index
    rownames(counts) <- as.character(genes)
    colnames(counts) <- paste0("cell", seq_len(ncol(counts)))
    mode(counts) <- "integer"

    sce <- SingleCellExperiment(
      assays = list(counts = counts),
      colData = data.frame(BatchInfo = sample(c("A", "B"), ncol(counts), replace = TRUE))
    )

    start <- Sys.time()
    chain <- BASiCS_MCMC(
      sce,
      N = MCMC_N, Thin = MCMC_THIN, Burn = MCMC_BURN,
      WithSpikes = FALSE, SubsetBy = "cell", Regression = TRUE,
      PrintProgress = FALSE, Threads = THREADS
    )

    mu <- colMeans(as.data.frame(displayChainBASiCS(chain, Param = "mu")))
    delta <- colMeans(as.data.frame(displayChainBASiCS(chain, Param = "delta")))

    parameters <- as.data.frame(cbind(mu, delta))
    parameters$variance <- parameters$mu + parameters$delta * parameters$mu^2
    write.csv(parameters, outfile)

    cat(fname, "done in", round(difftime(Sys.time(), start, units = "mins"), 1), "min\n")
  }
}
