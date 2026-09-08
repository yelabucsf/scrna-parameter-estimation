# export cgs=('T8' 'T4' 'NK' 'ncM' 'cM' 'B')
# export pop='eur'
# export method='memento'
# for c in ${cgs[@]}; do for chr in {1..22}; do echo $chr; python ~/Github/ldsc/ldsc.py --l2 --bfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}" --ld-wind-cm 1 --annot "/data_volume/memento/lupus/ldsc/annot_files/${method}/${pop}_${c}_${chr}.annot.gz" --thin-annot --out "/data_volume/memento/lupus/ldsc/ldscores/${method}/${pop}_${c}_${chr}" --print-snps "/data_volume/memento/lupus/ldsc/ldscores/hm3.snps"  ; done; done

# export cgs=('T8' 'T4' 'NK' 'ncM' 'cM' 'B')
# export pop='eur'
# export method='mateqtl'
# for c in ${cgs[@]}; do for chr in {1..22}; do echo $chr; python ~/Github/ldsc/ldsc.py --l2 --bfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}" --ld-wind-cm 1 --annot "/data_volume/memento/lupus/ldsc/annot_files/${method}/${pop}_${c}_${chr}.annot.gz" --thin-annot --out "/data_volume/memento/lupus/ldsc/ldscores/${method}/${pop}_${c}_${chr}" --print-snps "/data_volume/memento/lupus/ldsc/ldscores/hm3.snps"  ; done; done

export cgs=('T8' 'T4' 'NK' 'ncM' 'cM' 'B')
export pop='asian'
export method='memento'
echo $method
for c in ${cgs[@]}; do for chr in {1..22}; do echo $chr; python ~/Github/ldsc/ldsc.py --l2 --bfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}" --ld-wind-cm 1 --annot "/data_volume/memento/lupus/ldsc/annot_files/${method}/${pop}_${c}_${chr}.annot.gz" --thin-annot --out "/data_volume/memento/lupus/ldsc/ldscores/${method}/${pop}_${c}_${chr}" --print-snps "/data_volume/memento/lupus/ldsc/ldscores/hm3.snps"  ; done; done

export cgs=('T8' 'T4' 'NK' 'ncM' 'cM' 'B')
export pop='asian'
export method='mateqtl'
echo $method
for c in ${cgs[@]}; do for chr in {1..22}; do echo $chr; python ~/Github/ldsc/ldsc.py --l2 --bfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}" --ld-wind-cm 1 --annot "/data_volume/memento/lupus/ldsc/annot_files/${method}/${pop}_${c}_${chr}.annot.gz" --thin-annot --out "/data_volume/memento/lupus/ldsc/ldscores/${method}/${pop}_${c}_${chr}" --print-snps "/data_volume/memento/lupus/ldsc/ldscores/hm3.snps"  ; done; done

# export cgs=('T8' 'T4' 'NK' 'ncM' 'cM' 'B')
# export pop='both'
# export method='memento'
# for c in ${cgs[@]}; do for chr in {1..22}; do echo $chr; python ~/Github/ldsc/ldsc.py --l2 --bfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}" --ld-wind-cm 1 --annot "/data_volume/memento/lupus/ldsc/annot_files/${method}/${pop}_${c}_${chr}.annot.gz" --thin-annot --out "/data_volume/memento/lupus/ldsc/ldscores/${method}/${pop}_${c}_${chr}" --print-snps "/data_volume/memento/lupus/ldsc/ldscores/hm3.snps"  ; done; done

# export cgs=('T8' 'T4' 'NK' 'ncM' 'cM' 'B')
# export pop='both'
# export method='mateqtl'
# for c in ${cgs[@]}; do for chr in {1..22}; do echo $chr; python ~/Github/ldsc/ldsc.py --l2 --bfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}" --ld-wind-cm 1 --annot "/data_volume/memento/lupus/ldsc/annot_files/${method}/${pop}_${c}_${chr}.annot.gz" --thin-annot --out "/data_volume/memento/lupus/ldsc/ldscores/${method}/${pop}_${c}_${chr}" --print-snps "/data_volume/memento/lupus/ldsc/ldscores/hm3.snps"  ; done; done

# export cgs=('pbmc')
# export pop='eur'
# for c in ${cgs[@]}; do for chr in {1..22}; do echo $chr; python ~/Github/ldsc/ldsc.py --l2 --bfile "/data_volume/memento/lupus/ldsc/resources/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}" --ld-wind-cm 1 --annot "/data_volume/memento/lupus/ldsc/annot_files/expressed/${pop}_pbmc_${chr}.annot.gz" --thin-annot --out "/data_volume/memento/lupus/ldsc/ldscores/expressed/${pop}_${c}_${chr}" --print-snps "/data_volume/memento/lupus/ldsc/ldscores/hm3.snps"  ; done; done

# export cgs=('pbmc')
# export pop='asian'
# for c in ${cgs[@]}; do for chr in {1..22}; do echo $chr; python ~/Github/ldsc/ldsc.py --l2 --bfile "/data_volume/memento/lupus/ldsc/resources/1000G_Phase3_EAS_plinkfiles/1000G.EAS.QC.${chr}" --ld-wind-cm 1 --annot "/data_volume/memento/lupus/ldsc/annot_files/expressed/${pop}_pbmc_${chr}.annot.gz" --thin-annot --out "/data_volume/memento/lupus/ldsc/ldscores/expressed/${pop}_${c}_${chr}" --print-snps "/data_volume/memento/lupus/ldsc/ldscores/hm3.snps"  ; done; done