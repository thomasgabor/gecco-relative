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
for rtype in ["tabu", "dwave"]:
	ranks[rtype] = {}
	energies[rtype] = []
	for i, qubo in enumerate(sorted(population, key=lambda qubo: qubo.compute_metric(rtype))):
		ranks[rtype][qubo] = i
		energies[rtype].append(qubo.compute_metric(rtype))

# print(ranks)
plt.boxplot([energies["tabu"], energies["dwave"]])

for qubo in population:
	color = "grey"
	m1 = qubo.compute_metric("tabu")
	m2 = qubo.compute_metric("dwave")
	# m1 = ranks["siman"][qubo]
	# m2 = ranks["dwave"][qubo]
	if m1 > m2:
		color = "red"
	if m1 < m2:
		color = "blue"
	plt.plot([1, 2], [m1, m2],
		linestyle="solid",
		marker = 'o',
		alpha = 0.5,
		color=color
	)

plt.xticks([1,2], ['$H_{{tabu}}^{{min}}$', '$H_{{adv}}^{{min}}$'])


plt.show()
