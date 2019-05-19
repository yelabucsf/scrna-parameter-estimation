"""
	Preprocess and save the lupus data.

	1) Filter genes and cells
	2) Attach gender of each patient
"""


import scanpy.api as sc
import pandas as pd

data_path = '/ye/yelabstore3/mincheol/parameter_estimation/x_inactivation_data/'

if __name__ == '__main__':

