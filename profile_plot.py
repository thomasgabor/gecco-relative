import sys
import os
import dill
import matplotlib.pyplot as plt
import math
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

def compute_total(qubo):
	total = 0
	for cell in qubo:
		value = qubo[cell]
		total += abs(value)
	return total


total = {}
for i, qubo in enumerate(population):
		total[qubo] = (compute_total(qubo.qubo))

xs = [i for i in range(len(population))]

for qubo in sorted(population, key=lambda qubo: compute_total(qubo.qubo)):
	color = "grey"
	m1 = qubo.compute_metric("tabu")
	m2 = qubo.compute_metric("siman")
	m3 = qubo.compute_metric("dwave")
	y = qubo.compute_metric("exact")
	# m1 = ranks["siman"][qubo]
	# m2 = ranks["dwave"][qubo]
	y = m3 - m1
	plt.plot([total[qubo]], [y],
		marker = 'o',
		alpha = 0.5,
		color="blue"
	)

plt.xlabel('$\sum_{i,j} |Q_{ij}|$')
plt.ylabel('$H^{++}_{sa, adv}$')

plt.show()
