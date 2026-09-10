export cgs=('T8' 'T4' 'NK' 'ncM' 'cM' 'B')

# # all expressed genes (eur)
# export pop="eur"
# export cgs2=('pbmc')
# for c in ${cgs2[@]}; do echo $c; for chr in {1..22}; do echo $chr; python ~/Github/ldsc/make_annot.py --bed-file "/data_volume/memento/lupus/ldsc/bedfiles/pbmc.bed" --bimfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}.bim" --annot-file "/data_volume/memento/lupus/ldsc/annot_files/expressed/${pop}_pbmc_${chr}.annot.gz"; done ; done 

# # all expressed genes (asian)
# export pop="asian"
# export cgs2=('pbmc')
# for c in ${cgs2[@]}; do echo $c; for chr in {1..22}; do echo $chr; python ~/Github/ldsc/make_annot.py --bed-file "/data_volume/memento/lupus/ldsc/bedfiles/pbmc.bed" --bimfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}.bim" --annot-file "/data_volume/memento/lupus/ldsc/annot_files/expressed/${pop}_pbmc_${chr}.annot.gz"; done ; done 

# # European memento
# export pop="eur"
# for c in ${cgs[@]}; do echo $c; for chr in {1..22}; do echo $chr; python ~/Github/ldsc/make_annot.py --bed-file "/data_volume/memento/lupus/ldsc/bedfiles/memento/${pop}_${c}.bed" --bimfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}.bim" --annot-file "/data_volume/memento/lupus/ldsc/annot_files/memento/${pop}_${c}_${chr}.annot.gz"; done ; done 

# asian memento
export pop="asian"
for c in ${cgs[@]}; do echo $c; for chr in {1..22}; do echo $chr; python ~/Github/ldsc/make_annot.py --bed-file "/data_volume/memento/lupus/ldsc/bedfiles/memento/${pop}_${c}.bed" --bimfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}.bim" --annot-file "/data_volume/memento/lupus/ldsc/annot_files/memento/${pop}_${c}_${chr}.annot.gz"; done ; done 

# # combined memento (use eur reference)
# export pop="both"
# for c in ${cgs[@]}; do echo $c; for chr in {1..22}; do echo $chr; python ~/Github/ldsc/make_annot.py --bed-file "/data_volume/memento/lupus/ldsc/bedfiles/memento/${pop}_${c}.bed" --bimfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}.bim" --annot-file "/data_volume/memento/lupus/ldsc/annot_files/memento/${pop}_${c}_${chr}.annot.gz"; done ; done 

# # European mateqtl
# export pop="eur"
# for c in ${cgs[@]}; do echo $c; for chr in {1..22}; do echo $chr; python ~/Github/ldsc/make_annot.py --bed-file "/data_volume/memento/lupus/ldsc/bedfiles/mateqtl/${pop}_${c}.bed" --bimfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}.bim" --annot-file "/data_volume/memento/lupus/ldsc/annot_files/mateqtl/${pop}_${c}_${chr}.annot.gz"; done ; done 

# asian matqetl
export pop="asian"
for c in ${cgs[@]}; do echo $c; for chr in {1..22}; do echo $chr; python ~/Github/ldsc/make_annot.py --bed-file "/data_volume/memento/lupus/ldsc/bedfiles/mateqtl/${pop}_${c}.bed" --bimfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}.bim" --annot-file "/data_volume/memento/lupus/ldsc/annot_files/mateqtl/${pop}_${c}_${chr}.annot.gz"; done ; done 

# # combined mateqtl (use eur reference)
# export pop="both"
# for c in ${cgs[@]}; do echo $c; for chr in {1..22}; do echo $chr; python ~/Github/ldsc/make_annot.py --bed-file "/data_volume/memento/lupus/ldsc/bedfiles/mateqtl/${pop}_${c}.bed" --bimfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}.bim" --annot-file "/data_volume/memento/lupus/ldsc/annot_files/mateqtl/${pop}_${c}_${chr}.annot.gz"; done ; done 