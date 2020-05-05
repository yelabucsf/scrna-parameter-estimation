#!/bin/bash                        
#                           
#$ -S /bin/bash                  
#$ -o /netapp/home/mincheol/outputs
#$ -e /netapp/home/mincheol/outputs
#$ -cwd                          
#$ -r y                          
#$ -j y                          
#$ -l mem_free=30G             
#$ -l h_rt=3:00:00
##$ -t 1-35

export PYTHONPATH=/netapp/home/mincheol/scrna-parameter-estimation/simplesc/

source activate scvi
python /netapp/home/mincheol/scrna-parameter-estimation/examples/interferon_stimulation/robust_pca.py
source deactivate