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

#qsub -v file="" -v annot="" -v out="" submit.
#cts_name=Cahoy 
#gwas="UKBB_BMI.sumstats.gz"
#ref="1000G_EUR_Phase3_baseline/baseline."
#out="BMI_"${cts_name}
#ldcts=${cts_name}.ldcts
#weights="weights_hm3_no_hla/weights."

echo $file
echo $annot
echo $out

dir="/wynton/group/ye/ggordon/lupus/analyses/ldscore_reg"

#Run the regression
python /wynton/home/ye/ggordon/tools/ldsc/ldsc.py --l2 --bfile ${file} --ld-wind-cm 1 --annot ${annot} --thin-annot --out ${out} --print-snps ${dir}/hm3.snps  

#python /wynton/home/ye/ggordon/tools/ldsc/ldsc.py \
#    --h2-cts ${gwas} \
#    --ref-ld-chr ${ref} \
#    --out ${out} \
#    --ref-ld-chr-cts $ldcts \
#    --w-ld-chr ${weights} 
