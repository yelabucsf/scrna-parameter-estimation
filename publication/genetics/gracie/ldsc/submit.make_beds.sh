#!/bin/bash                         #-- what is the language of this shell
#                                  #-- Any line that starts with #$ is an instruction to SGE
#$ -S /bin/bash                     #-- the shell for the job
#$ -o .                        #-- output directory (fill in)
#$ -e .                        #-- error directory (fill in)
#$ -cwd                            #-- tell the job that it should start in your working directory
#$ -r y                            #-- tell the system that if a job crashes, it should be restarted
#$ -j y                            #-- tell the system that the STDERR and STDOUT should be joined
#$ -l mem_free=20G                  #-- submits on nodes with enough free memory (required)

##$ -l arch=linux-x64               #-- SGE resources (CPU type)
##$ -l netapp=1G,scratch=1G         #-- SGE resources (home and scratch disks)
#$ -l h_rt=48:00:00                #-- runtime limit (see above; this requests 24 hours)
##$ -t 1-13

#cell="b"
#win="100000"

# for c in ${cg_r2[@]}; do echo $c; qsub -v cell=$c -v win="100000" -v out_suf="_expr.bed" -v genelist_dir="/wynton/group/ye/ggordon/lupus/analyses/prep_eqtls/Pseudobulk/" -v genelist_suf=".expressed.genes.txt" ../submit.make_beds.sh ; done

echo $cell
echo $win
echo $out_suf
echo $genelist_dir
echo $genelist_suf


rm ${cell}${out_suf}
while read p ; do grep -w $p /wynton/group/ye/ggordon/ref/genome/gene_locs.txt | awk -v window=$win '{print $2"\t"$3-window"\t"$4+window"\t"$1}' >> ${cell}${out_suf} ; done<${genelist_dir}/${cell}${genelist_suf}


