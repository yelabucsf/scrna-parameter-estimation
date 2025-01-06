#!/bin/bash                         #-- what is the language of this shell
#                                  #-- Any line that starts with #$ is an instruction to SGE
#$ -S /bin/bash                     #-- the shell for the job
#$ -o log/                        #-- output directory (fill in)
#$ -e log/                        #-- error directory (fill in)
#$ -cwd                            #-- tell the job that it should start in your working directory
#$ -r y                            #-- tell the system that if a job crashes, it should be restarted
#$ -j y                            #-- tell the system that the STDERR and STDOUT should be joined
#$ -l mem_free=20G                  #-- submits on nodes with enough free memory (required)

##$ -l arch=linux-x64               #-- SGE resources (CPU type)
##$ -l netapp=1G,scratch=1G         #-- SGE resources (home and scratch disks)
#$ -l h_rt=48:00:00                #-- runtime limit (see above; this requests 24 hours)
##$ -t 1-13

source activate ldsc

#qsub -v bed="Brain_DPC_H3K27ac.bed" -v bim="1000G.EUR.QC.22.bim" -v out="Brain_DPC_H3K27ac.annot.gz" submit.make_annot.sh
echo $bed
echo $bim
echo $out

python /wynton/home/ye/ggordon/tools/ldsc/make_annot.py \
		--bed-file $bed \
		--bimfile $bim \
		--annot-file $out 
