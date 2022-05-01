traits=( PASS_AgeFirstBirth PASS_Anorexia PASS_Autism PASS_BMI1 PASS_Coronary_Artery_Disease PASS_Crohns_Disease PASS_Height1 PASS_Rheumatoid_Arthritis PASS_Schizophrenia PASS_Type_2_Diabetes PASS_Ulcerative_Colitis UKB_460K.blood_EOSINOPHIL_COUNT UKB_460K.blood_PLATELET_COUNT UKB_460K.blood_RED_COUNT UKB_460K.blood_WHITE_COUNT UKB_460K.cov_SMOKING_STATUS UKB_460K.disease_ALLERGY_ECZEMA_DIAGNOSED bentham.lupus )

weights="/data_volume/memento/lupus/ldsc/resources/weights_hm3_no_hla/weights."

# cohorts=( memento_asian mateqtl_asian )
# ref="/data_volume/memento/lupus/ldsc/resources/1kg_eas/baselineLD."


# for cohort in ${cohorts[@]}; do 
# 	for t in ${traits[@]}; do 
# 		echo $t ;		
# 		~/Github/ldsc/ldsc.py \
# 			--h2-cts "/data_volume/memento/lupus/ldsc/resources/gwas_data/${t}.sumstats.gz" \
# 			--ref-ld-chr ${ref}\
# 			--out "/data_volume/memento/lupus/ldsc/results/${cohort}_${t}" \
# 			--ref-ld-chr-cts "${cohort}.ldcts" \
# 			--w-ld-chr ${weights} ;
# 	done; 
# done

# cohorts=( memento_eur mateqtl_eur )
# ref="/data_volume/memento/lupus/ldsc/resources/1kg_eur/"


# for cohort in ${cohorts[@]}; do 
# 	for t in ${traits[@]}; do 
# 		echo $t ;		
# 		~/Github/ldsc/ldsc.py \
# 			--h2-cts "/data_volume/memento/lupus/ldsc/resources/gwas_data/${t}.sumstats.gz" \
# 			--ref-ld-chr ${ref}\
# 			--out "/data_volume/memento/lupus/ldsc/results/${cohort}_${t}" \
# 			--ref-ld-chr-cts "${cohort}.ldcts" \
# 			--w-ld-chr ${weights} ;
# 	done; 
# done

cohorts=( memento_both mateqtl_both )
ref="/data_volume/memento/lupus/ldsc/resources/1kg_eur/"


for cohort in ${cohorts[@]}; do 
	for t in ${traits[@]}; do 
		echo $t ;		
		~/Github/ldsc/ldsc.py \
			--h2-cts "/data_volume/memento/lupus/ldsc/resources/gwas_data/${t}.sumstats.gz" \
			--ref-ld-chr ${ref}\
			--out "/data_volume/memento/lupus/ldsc/results/${cohort}_${t}" \
			--ref-ld-chr-cts "${cohort}.ldcts" \
			--w-ld-chr ${weights} ;
	done; 
done


# 		~/Github/ldsc/ldsc.py \
# 			--h2-cts "/data_volume/memento/lupus/ldsc/resources/gwas_data/PASS_Ulcerative_Colitis.sumstats.gz" \
# 			--ref-ld-chr "/data_volume/memento/lupus/ldsc/resources/1kg_eur/"\
# 			--out "/data_volume/memento/lupus/ldsc/results/memento_eur_PASS_Ulcerative_Colitis" \
# 			--ref-ld-chr-cts "${cohort}.ldcts" \
# 			--w-ld-chr ${weights} ;