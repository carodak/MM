"""Gr<aph drawing test Script."""
from minerminor import mm_draw as mmd

data = mmd.csv_to_dic("resultats/base_planar_k5k33_16:14:01.892811.txt")

mmd.create_fmeasure_curve(data)
