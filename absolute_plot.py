import sys
import os
import dill
import matplotlib.pyplot as plt
from main import *


if len(sys.argv) > 1:
	filepath = sys.argv[1]
else:
	mypath = os.path.join(os.getcwd(), "experiments")
	onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
	onlyfiles.sort()
	filepath = os.path.join(mypath, onlyfiles[-1])

population = []
with open(filepath, "rb") as file:
	population = dill.load(file)

ranks = {}
energies = {}
for rtype in ["exact", "tabu", "siman", "dwave"]:
	ranks[rtype] = {}
	energies[rtype] = []
	for i, qubo in enumerate(sorted(population, key=lambda qubo: qubo.compute_metric(rtype))):
		ranks[rtype][qubo] = i
		energies[rtype].append(qubo.compute_metric(rtype))

# print(ranks)
plt.boxplot([energies["tabu"], energies["siman"], energies["dwave"]])
plt.xticks([1,2,3], ['$H_{{tabu}}^{{min}}$', '$H_{{sa}}^{{min}}$', '$H_{{adv}}^{{min}}$'])

plt.show()
