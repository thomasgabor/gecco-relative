import os
import random
from dimod import BinaryQuadraticModel,ExactSolver,SimulatedAnnealingSampler
from tabu import TabuSampler
from dimod.utilities import qubo_energy
from dwave.system import DWaveSampler,EmbeddingComposite
from dwave_qbsolv import QBSolv
import networkx as nx
import dill
import time
from tqdm import tqdm


random.seed(42)

qubo_size = 250
pop_size = 100

qubits_offset = 0

endpoint = "https://cloud.dwavesys.com/sapi"
token = "YOUR TOKEN HERE"
solver = "Advantage_system4.1"

sampler = DWaveSampler(endpoint=endpoint, token=token, solver=solver)
# print(sampler.edgelist)
substrate_graph = nx.Graph(sampler.edgelist)

def generate_random_qubo_pegasus():
	qubo = {}
	count = 0
	for j in range(qubits_offset, qubo_size+qubits_offset):
		for k in range(j, qubo_size+qubits_offset):
			# print(str(j) +"," + str(k))
			if substrate_graph.has_edge(j, k) or j == k:
				count += 1
				qubo[(j,k)] = random.randint(-1000, +1000)
	# print(count)
	return qubo

def generate_random_qubo_any(add_filled_in=0.01):
	qubo = {}
	count = 0
	for j in range(qubits_offset, qubo_size+qubits_offset):
		qubo[(j,random.randint(j,qubo_size+qubits_offset))] = random.randint(-1000, +1000)
		for k in range(j, qubo_size+qubits_offset):
			if random.random() < add_filled_in:
				qubo[(j,k)] = random.randint(-1000, +1000)
				count += 1
	# print(count)
	return qubo


def solve_qubo_dwave(qubo_dict, **parameters):
    bqm = BinaryQuadraticModel.from_qubo(qubo_dict)
    e_sampler = EmbeddingComposite(sampler)
    return e_sampler.sample(bqm, return_embedding=True, **parameters).aggregate().record

def solve_qubo_qbsolv(qubo_dict, **parameters):
	results = QBSolv().sample_qubo(qubo.qubo, num_reads=100)
	record = []
	for sample, energy in results.data(['sample', 'energy']):
		record.append([sample, energy])
	return record

def solve_qubo_exact(qubo_dict, **parameters):
	bqm = BinaryQuadraticModel.from_qubo(qubo_dict)
	e_sampler = ExactSolver()
	return e_sampler.sample(bqm, **parameters).aggregate().record

def solve_qubo_tabu(qubo_dict, **parameters):
	bqm = BinaryQuadraticModel.from_qubo(qubo_dict)
	e_sampler = TabuSampler()
	response = e_sampler.sample(bqm, **parameters).aggregate().record
	l = []
	for entry in response:
		new_entry = []
		d = {}
		for var,val in enumerate(entry[0]):
			d[var+qubits_offset] = val
		new_entry.append(d)
		new_entry.append(entry[1])
		l.append(new_entry)
	return l

def solve_qubo_siman(qubo_dict, **parameters):
	bqm = BinaryQuadraticModel.from_qubo(qubo_dict)
	e_sampler = SimulatedAnnealingSampler()
	return e_sampler.sample(bqm, **parameters).aggregate().record

def compute_energy(entry, qubo):
	return qubo_energy(entry, qubo)

def get_best_response(record, qubo):
	best_energy = None
	best_response = None
	for entry in record:
		entry_energy = compute_energy(entry[0], qubo)
		if not best_energy or entry_energy < best_energy:
			best_energy = entry_energy
			best_response = entry[0]
	return best_response


class QUBO:
	def __init__(self, qubo={}) -> None:
		self.qubo = qubo
		self.responses = {}
		pass
	def save_response(self, response_type, response):
		self.responses[response_type] = response
	def get_best_response(self, response_type):
		best_response = get_best_response(self.responses[response_type], self.qubo)
		return best_response
	def get_best_energy(self, response_type):
		best_response = self.get_best_response(response_type)
		return compute_energy(best_response, self.qubo)
	def compute_metric(self, response_type, ground_response_type="exact"):
		return self.get_best_energy(ground_response_type) - self.get_best_energy(response_type)

if __name__ == "__main__":
	population = []

	print(">> generate population")
	for i in tqdm(range(pop_size)):
		qubo = QUBO(generate_random_qubo_pegasus())
		population.append(qubo)

	print(">> evaluate population")
	for qubo in tqdm(population):
		qubo.save_response("exact", solve_qubo_qbsolv(qubo.qubo))
		qubo.save_response("tabu", solve_qubo_tabu(qubo.qubo, num_reads=10))
		qubo.save_response("dwave", solve_qubo_dwave(qubo.qubo, num_reads=10))
		qubo.save_response("siman", solve_qubo_siman(qubo.qubo, num_reads=10))

	for qubo in population:
		print(str(qubo.compute_metric("siman")) + " <=> " + str(qubo.compute_metric("dwave")) + " <=> " + str(qubo.compute_metric("tabu")))

	with open("experiments/ex" + str(time.time()) + ".dill", "wb") as file:
		dill.dump(population, file)


