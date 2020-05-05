#!/bin/bash                        
#                           
#$ -S /bin/bash                  
#$ -o /wynton/group/ye/mincheol/outputs
#$ -e /wynton/group/ye/mincheol/outputs
#$ -cwd                          
#$ -r y                          
#$ -j y                          
#$ -l mem_free=20G             
#$ -l h_rt=12:00:00

# If you used the -t option above, this same script will be run for each task,
# but with $SGE_TASK_ID set to a different value each time (1-10 in this case).
# The commands below are one way to select a different input (PDB codes in
# this example) for each task.  Note that the bash arrays are indexed from 0,
# while task IDs start at 1, so the first entry in the tasks array variable
# is simply a placeholder

#tasks=(0 1bac 2xyz 3ijk 4abc 5def 6ghi 7jkl 8mno 9pqr 1stu )
#input="${tasks[$SGE_TASK_ID]}"

export PYTHONPATH=/wynton/group/ye/mincheol/Github/scrna-parameter-estimation/simplesc

conda activate single_cell
python $1
conda deactivate