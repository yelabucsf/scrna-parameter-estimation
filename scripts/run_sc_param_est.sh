#!/bin/bash                        
#                           
#$ -S /bin/bash                  
#$ -o /netapp/home/mincheol/outputs
#$ -e /netapp/home/mincheol/outputs
#$ -cwd                          
#$ -r y                          
#$ -j y                          
#$ -l mem_free=30G
#$ -l arch=linux-x64             
#$ -l netapp=5G,scratch=5G      
#$ -l h_rt=3:00:00
##$ -t 1-10    

# If you used the -t option above, this same script will be run for each task,
# but with $SGE_TASK_ID set to a different value each time (1-10 in this case).
# The commands below are one way to select a different input (PDB codes in
# this example) for each task.  Note that the bash arrays are indexed from 0,
# while task IDs start at 1, so the first entry in the tasks array variable
# is simply a placeholder

#tasks=(0 1bac 2xyz 3ijk 4abc 5def 6ghi 7jkl 8mno 9pqr 1stu )
#input="${tasks[$SGE_TASK_ID]}"

export PYTHONPATH=/netapp/home/mincheol/scrna-parameter-estimation/simplesc/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/netapp/home/mincheol/anaconda3/lib

# export CUDA_PATH=/ye/yelabstore2/mincheol/cuda-8.0
# export CUDA_HOME=$CUDA_PATH
# export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
# export PATH=$PATH:/ye/yelabstore2/mincheol/cuda-8.0/bin

source activate scvi

python /netapp/home/mincheol/scrna-parameter-estimation/scripts/$1

source deactivate

qstat -j $JOB_ID                                  # This is useful for debugging and usage purposes,
                                                  # e.g. "did my job exceed its memory request?"