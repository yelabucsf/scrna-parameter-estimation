#!/bin/bash                         #-- what is the language of this shell
#                                  #-- Any line that starts with #$ is an instruction to SGE
#$ -S /bin/bash                     #-- the shell for the job
#$ -o log/                        #-- output directory (fill in)
#$ -e log/                        #-- error directory (fill in)
#$ -cwd                            #-- tell the job that it should start in your working directory
#$ -r y                            #-- tell the system that if a job crashes, it should be restarted
#$ -j y                            #-- tell the system that the STDERR and STDOUT should be joined
#$ -l mem_free=6G                  #-- submits on nodes with enough free memory (required)
##$ -l arch=linux-x64               #-- SGE resources (CPU type)
##$ -l netapp=1G,scratch=1G         #-- SGE resources (home and scratch disks)
#$ -l h_rt=48:00:00                #-- runtime limit (see above; this requests 24 hours)
##$ -pe smp 4
##$ -t 1-100

#source ~/miniconda3/bin/activate base_py36

for num in $(seq 1 100); do tasks+=($num); done
p="${tasks[$SGE_TASK_ID]}"

##SCRIPT TO RUN ATAC ENRICHMENT FROM MEENA
#dir="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/european_expr_maf01/"
dir="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/european_fixed_expr_maf01/"
dir="/wynton/group/ye/ggordon/lupus/analyses/gregor/SLE/sig_eur_maf0.1_eqtls"
dir="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/cg_european_fixed_expr_maf01/genewise/sig_fdr0.1"
dir="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/cg2_european_fixed_expr_maf01/all_snps/"
dir="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/cg3_european_expr_maf01/all_snps/"
#dir="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/cg2_european_fixed_expr_maf01/all_snps/filt_expr/"
dir="/wynton/group/ye/ggordon/lupus/analyses/fastGxE/test_outs_july_test/mateqtl/"
dir="/wynton/group/ye/ggordon/lupus/analyses/fastGxE/all_cells_final/mateqtl/"
#/wynton/group/ye/ggordon/lupus/analyses/fastGxE/test_outs/mateqtl/b.normalized_and_residualized_expression_heterogeneous.all_pairs.txt

#/wynton/group/ye/ggordon/lupus/analyses/fastGxE/test_outs/TreeQTL/eAssoc_by_gene.b.normalized_and_residualized_expression_heterogeneous.txt
#mateqtl_cm_cis.csv 
cis='1e5'
echo $cis

#dir=/ye/yelabstore3/10x.lupus/eqtls/v5/sle.10pcs/all/
#dir=/ye/yelabstore3/10x.lupus/eqtls/v6.1/sle.10pcs/all/
#dir_res=/ye/yelabstore3/10x.lupus/eqtls/sle.10pcs.subtracted.1mb/all/
#dir_res=/ye/yelabstore3/10x.lupus/eqtls/v5/sle.10pcs/all/
#dir_res=/ye/yelabstore3/10x.lupus/eqtls/sle.10pcs.subtracted/all/

#Rscript intersect.peaks.R ${dir_res}/$ct.all.results.txt ${dir}/$ct.empirical.pval.txt ${dir_res}/$ct.atac.enrichment.txt
#Rscript intersect.peaks.withdelta.MW.R ${dir}/$ct.all.results.txt ${dir}/$ct.empirical.pval.txt ${dir_res}/$ct.all.results.txt ${dir_res}/$ct.atac.intersectdelta.MW.enrichment.txt
#Rscript intersect.peaks.combinect.R ${dir_res} ${dir} ${dir_res}/ct.combined.atac.enrichment.txt

#Rscript intersect.peaks.withdelta.MW.R ${dir}/mateqtl_${ct}_cis.csv ${ct}.atac.MW.enrichment.txt

#Rscript intersect.peaks.withdelta.MW.R ${dir}/eAssoc_by_gene.${ct}.normalized_and_residualized_expression_heterogeneous.txt ${ct}.atac.MW.enrichment.txt

dir="/wynton/group/ye/ggordon/lupus/analyses/fastGxE/euro_all_cells_final/mateqtl/"
Rscript fast_hom.intersect.peaks.withdelta.MW.R ${dir}/AverageTissue.normalized_and_residualized_expression_homogeneous.all_pairs.txt AverageTissue.atac.MW.enrichment.fastgxe_euro.txt 

Rscript fast.intersect.peaks.withdelta.MW.R ${dir}/${ct}.normalized_and_residualized_expression_heterogeneous.all_pairs.txt ${ct}.atac.MW.enrichment.fastgxe_euro.txt /wynton/group/ye/ggordon/lupus/analyses/prep_eqtls/Pseudobulk/european/cg/${ct}.European.expressed.genes.txt

dir="/wynton/group/ye/ggordon/lupus/analyses/fastGxE/asian_all_cells_final/mateqtl/"
Rscript fast_hom.intersect.peaks.withdelta.MW.R ${dir}/AverageTissue.normalized_and_residualized_expression_homogeneous.all_pairs.txt AverageTissue.atac.MW.enrichment.fastgxe_asian.txt

Rscript fast.intersect.peaks.withdelta.MW.R ${dir}/${ct}.normalized_and_residualized_expression_heterogeneous.all_pairs.txt ${ct}.atac.MW.enrichment.fastgxe_asian.txt /wynton/group/ye/ggordon/lupus/analyses/prep_eqtls/Pseudobulk/european/cg/${ct}.European.expressed.genes.txt


#Rscript fast_hom.intersect.peaks.withdelta.MW.R ${dir}/${ct}.normalized_and_residualized_expression_heterogeneous.all_pairs.txt ${ct}.atac.MW.enrichment.fastgxeALL_${p}.txt

#UNCOMMENT FOR ALL OTHERS
#Rscript fast.intersect.peaks.withdelta.MW.R ${dir}/${ct}.normalized_and_residualized_expression_heterogeneous.all_pairs.txt ${ct}.atac.MW.enrichment.fastgxe_${p}.txt /wynton/group/ye/ggordon/lupus/analyses/prep_eqtls/Pseudobulk/european/cg/${ct}_cg.European.expressed.genes.txt






#pdc_cg.European.expressed.genes.txt

#filtered genewise
#Rscript intersect.peaks.withdelta.MW.R ${dir}/${ct}.genewise.sig.eqtl.fdr0.1.txt ${ct}.atac.MW.enrichment.genewise.txt

#Rscript intersect.peaks.withdelta.MW.R ${dir}/${ct}_genewise_ct_eqtls_${cis}_cis_fdr0.1.txt ${ct}.atac.MW.enrichment.genewise.${cis}.txt




