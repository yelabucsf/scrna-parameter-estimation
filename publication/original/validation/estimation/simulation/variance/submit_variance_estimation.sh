# Make datasets
conda activate mementocxg
python simulate_variance_datasets.py

# Run BASICS
conda activate r-env
Rscript run_basics.r