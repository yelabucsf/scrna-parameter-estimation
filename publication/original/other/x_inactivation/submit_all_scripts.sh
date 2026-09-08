#!/bin/bash

# For CT differences
# for i in $(seq 0 11)
# do
# 	echo $i
# 	qsub submit_x_inactivation_statistics.sh ct $i
# done

# For CT differences in individuals
# for i in $(seq 0 168)

# do
# 	echo $i
# 	qsub submit_x_inactivation_statistics.sh ct_ind $i
# done

# For sex differences
for i in $(seq 0 11)
do
	echo $i
	qsub submit_x_inactivation_statistics.sh sex $i
done