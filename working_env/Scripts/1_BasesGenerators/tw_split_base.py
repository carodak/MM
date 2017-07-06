"""Script split TW."""
import minerminor.mm_utils as mmu
import os
import argparse

parser = argparse.ArgumentParser(prog="Split Tw base")

parser.add_argument("-p", "--path_dir", default="base_test",
                    help="Path of base dir")
args = parser.parse_args()

learning_base = []
path_base = args.path_dir
list_labels_dir = os.listdir(path_base)
for dir_name in list_labels_dir:
    base = mmu.tw_split_base(path_base+"/"+dir_name)
    mmu.store_base(base, path_base+"_transf/"+dir_name)

# for base in learning_base:
# print(learning_base)
