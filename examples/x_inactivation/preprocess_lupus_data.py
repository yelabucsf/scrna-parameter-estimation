"""
	Preprocess and save the lupus data.

	1) Filter genes and cells
	2) Attach gender of each patient
"""


import scanpy.api as sc
import pandas as pd

