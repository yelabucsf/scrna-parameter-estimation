export cgs=('T8' 'T4' 'NK' 'ncM' 'cM' 'B')

export pop="asian"
for c in ${cgs[@]}; do echo $c; for chr in {1..22}; do echo $chr; python ~/Github/ldsc/make_annot.py --bed-file "/data_volume/memento/lupus/ldsc/bedfiles/mateqtl/${pop}_${c}.bed" --bimfile "/data_volume/memento/lupus/ldsc/resources/1000G_Phase3_EAS_plinkfiles/1000G.EAS.QC.${chr}.bim" --annot-file "/data_volume/memento/lupus/ldsc/annot_files/mateqtl/${pop}_${c}_${chr}.annot.gz"; done ; done 