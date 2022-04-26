#!/bin/bash                         #-- what is the language of this shell
#                                  #-- Any line that starts with #$ is an instruction to SGE
#$ -S /bin/bash                     #-- the shell for the job
#$ -o log/                        #-- output directory (fill in)
#$ -e log/                        #-- error directory (fill in)
#$ -cwd                            #-- tell the job that it should start in your working directory
#$ -r y                            #-- tell the system that if a job crashes, it should be restarted
#$ -j y                            #-- tell the system that the STDERR and STDOUT should be joined
#$ -l mem_free=80G                  #-- submits on nodes with enough free memory (required)
##$ -l arch=linux-x64               #-- SGE resources (CPU type)
##$ -l netapp=1G,scratch=1G         #-- SGE resources (home and scratch disks)
#$ -l h_rt=48:00:00                #-- runtime limit (see above; this requests 24 hours)
##$ -pe smp 4
##$ -t 1-100

#source ~/miniconda3/bin/activate base_py36

#for num in $(seq 0 100); do tasks+=($num); done
#p="${tasks[$SGE_TASK_ID]}"

#source ~/miniconda3/bin/activate base_py36

##SCRIPT TO RUN ATAC ENRICHMENT FROM MEENA
#dir="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/european_expr_maf01/"
#dir="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/european_fixed_expr_maf01/"
#dir="/wynton/group/ye/ggordon/lupus/analyses/gregor/SLE/sig_eur_maf0.1_eqtls"
#dir="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/cg_european_fixed_expr_maf01/genewise/sig_fdr0.1"
#dir="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/cg2_european_fixed_expr_maf01/all_snps/"
#dir="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/cg3_european_expr_maf01/all_snps/"
#dir="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/asian_euro_meta/1e5/"
#dir="/wynton/group/ye/ggordon/lupus/analyses/fastGxE/meta_fastGxE/" 
#dir="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/cg2_european_fixed_expr_maf01/all_snps/filt_expr/"
#dir="/wynton/group/ye/ggordon/lupus/analyses/fastGxE/test_outs/TreeQTL/"


dir="/wynton/group/ye/ggordon/lupus/analyses/fastGxE/meta_fastGxE/" 
dir_nopdc="/wynton/group/ye/ggordon/lupus/analyses/fastGxE/meta_fastGxE_nopdc/"
dir2="/wynton/group/ye/ggordon/lupus/analyses/mat_eqtl/asian_euro_meta/"
#/wynton/group/ye/ggordon/lupus/analyses/fastGxE/test_outs/TreeQTL/eAssoc_by_gene.b.normalized_and_residualized_expression_heterogeneous.txt
dir="/wynton/group/ye/ggordon/lupus/data/mateqtl_format_data/meta_fastGxE_all/"
dir2="/wynton/group/ye/ggordon/lupus/data/mateqtl_format_data/meta_van_all/"
#mateqtl_cm_cis.csv 

#cis='1e6'
cis='1e5'
echo $cis
echo $dir
echo $dir2
echo $ct

##run average
#Rscript meta_nofilt.intersect.peaks.withdelta.MW.R ${dir}/AverageTissue_metasoft_out.txt AverageTissue.atac.MW.enrichment.fastgxe_meta_het_nof.txt & 
#
##run het
##Rscript meta_nofilt.intersect.peaks.withdelta.MW.R ${dir}/${ct}_metasoft_het_out.txt ${ct}.atac.MW.enrichment.fastgxe_meta_het_nof.txt & 
#Rscript meta_nofilt.intersect.peaks.withdelta.MW.R ${dir}/${ct}_metasoft_out.txt ${ct}.atac.MW.enrichment.fastgxe_meta_het_nof.txt &
#
##run vanilla from fast pipe
Rscript meta_nofilt.intersect.peaks.withdelta.MW.R ${dir2}/${ct}_metasoft_out.txt ${ct}.atac.MW.enrichment.fastgxe_meta_van_${cis}_new.txt &
Rscript meta_nofilt.intersect.peaks.withdelta.MW.R ${dir2}/pbmc_metasoft_out.txt pbmc.atac.MW.enrichment.fastgxe_meta_van_${cis}_new.txt &
#
#wait
#
#echo 'done'

##run average
Rscript meta.intersect.peaks.withdelta.MW.R ${dir}/AverageTissue_metasoft_out.txt AverageTissue.atac.MW.enrichment.fastgxe_meta_het_${cis}_new.txt /wynton/group/ye/ggordon/lupus/analyses/prep_eqtls/Pseudobulk/european/cg/pbmc_cg.European.expressed.genes.txt &
#
#run het
Rscript meta.intersect.peaks.withdelta.MW.R ${dir}/${ct}_metasoft_out.txt ${ct}.atac.MW.enrichment.fastgxe_meta_het_${cis}_new.txt /wynton/group/ye/ggordon/lupus/analyses/prep_eqtls/Pseudobulk/european/cg/${ct}_cg.European.expressed.genes.txt 
#
##no pdc 
#
##run average
#Rscript meta.intersect.peaks.withdelta.MW.R ${dir_nopdc}/${cis}/AverageTissue_metasoft_out.txt AverageTissue.atac.MW.enrichment.fastgxe_meta_het_nopdc_${cis}.txt /wynton/group/ye/ggordon/lupus/analyses/prep_eqtls/Pseudobulk/european/cg/pbmc_cg.European.expressed.genes.txt &
#
##run het
#Rscript meta.intersect.peaks.withdelta.MW.R ${dir_nopdc}/${cis}/${ct}_metasoft_out.txt ${ct}.atac.MW.enrichment.fastgxe_meta_het_nopdc_${cis}.txt /wynton/group/ye/ggordon/lupus/analyses/prep_eqtls/Pseudobulk/european/cg/${ct}_cg.European.expressed.genes.txt &
#
###run vanilla from fast pipe
#Rscript meta.intersect.peaks.withdelta.MW.R ${dir2}/${ct}_metasoft_out.txt ${ct}.atac.MW.enrichment.fastgxe_meta_${cis}_new.txt /wynton/group/ye/ggordon/lupus/analyses/prep_eqtls/Pseudobulk/european/cg/${ct}_cg.European.expressed.genes.txt &

wait

echo 'done'





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

#Rscript meta.intersect.peaks.withdelta.MW.R ${dir}/${ct}_asian_euro_metasoft_out.txt ${ct}.atac.MW.enrichment.${cis}_meta_simple2_${p}.txt /wynton/group/ye/ggordon/lupus/analyses/prep_eqtls/Pseudobulk/european/cg/${ct}.European.expressed.genes.txt

#Rscript meta.intersect.peaks.withdelta.MW.R ${dir}/${ct}_asian_euro_metasoft_out.txt ${ct}.atac.MW.enrichment.${cis}_meta.txt /wynton/group/ye/ggordon/lupus/analyses/prep_eqtls/Pseudobulk/european/cg/${ct}.European.expressed.genes.txt

#pdc_cg.European.expressed.genes.txt

#filtered genewise
#Rscript intersect.peaks.withdelta.MW.R ${dir}/${ct}.genewise.sig.eqtl.fdr0.1.txt ${ct}.atac.MW.enrichment.genewise.txt

#Rscript intersect.peaks.withdelta.MW.R ${dir}/${ct}_genewise_ct_eqtls_${cis}_cis_fdr0.1.txt ${ct}.atac.MW.enrichment.genewise.${cis}.txt


