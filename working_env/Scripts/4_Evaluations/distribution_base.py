import minerminor.mm_utils as mmu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx


base_path = "bases/base_TW1/learning-base-rdm_18_[0, 1]_1000"

learning_base = mmu.load_base(base_path)

arr_vec = []

for classes in learning_base:
    arr_vec.append([])
    for graph in classes:
        arr_vec[-1].append(graph.number_of_edges())


h1 = plt.hist(arr_vec[0], 100, normed=False, facecolor="blue", alpha=0.75)
h2 = plt.hist(arr_vec[1], 100, normed=False, facecolor="green", alpha=0.75)

plt.xlabel('Arêtes')
plt.ylabel('Quantité de graphes (%)')
plt.grid(True)


blue_patch = mpatches.Patch(color='blue', label='P')
green_patch = mpatches.Patch(color='green', label='!P')
plt.legend(handles=[blue_patch, green_patch])

print(arr_vec[0], arr_vec[1])
plt.axis([0, 160, 0, 1000])
plt.show()
