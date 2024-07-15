async function main(){
	let pyodide = await loadPyodide();
	await pyodide.loadPackage("micropip");
	const micropip = pyodide.pyimport("micropip");
	await micropip.install(['numpy', 'pandas', 'igraph', 'networkx', 'scipy', 'scikit-learn', 'seaborn', 'matplotlib', 'statsmodels'])

	pyodide.runPython(`
	import igraph as ig
	import numpy as np
	import pandas as pd
	import networkx as nx
	from scipy.optimize import minimize
	import random
	from collections import Counter
	from itertools import combinations
	from scipy.stats import multivariate_normal
	from sklearn.mixture import GaussianMixture
	from sklearn.cluster import KMeans
	from sklearn.metrics import silhouette_score
	import seaborn as sns
	import matplotlib.pyplot as plt
	from sklearn.decomposition import PCA
	from scipy.cluster.hierarchy import dendrogram, linkage
	from sklearn.cluster import DBSCAN
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.metrics import mean_squared_error
	from sklearn.linear_model import Lasso
	from sklearn.svm import SVC
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import silhouette_score
	from scipy.spatial.distance import pdist
	from scipy.cluster.hierarchy import cophenet
	from sklearn.preprocessing import StandardScaler
	from sklearn.model_selection import GridSearchCV
	from sklearn.pipeline import Pipeline
	from sklearn.metrics import roc_curve, auc
	from scipy.signal import find_peaks
	from scipy.stats import norm
	import statsmodels.api as sm
	import seaborn as sns
	import warnings
	import math
	import matplotlib.colors as mcolors
	from scipy.stats import zscore
	warnings.filterwarnings("ignore")
	`);
	
	pyodide.runPython(`
	edges = [
		('short-term counterparty funds', 'funding'),
		('credit risk', 'short-term counterparty funds'),
		('deposits', 'funding'),
		('regional economic conditions', 'deposits'),
		('regional economic conditions', 'asset prices'),
		('asset prices', 'regional economic conditions'),
		('asset prices', 'external capital sources'),
		('asset prices', 'earning assets'),
		('asset prices', 'concentrations'),
		('asset prices', 'competitive pressure'),
		('credit risk', 'asset prices'),
		('earning assets', 'asset prices'),
		('external capital sources', 'capital'),
		('credit risk', 'external capital sources'),
		('capital', 'earning assets'),
		('earning assets', 'capital'),
		('earning assets', 'concentrations'),
		('liquidity', 'earning assets'),
		('funding', 'liquidity'),
		('capital', 'regulatory burden/expectation'),
		('expenses', 'capital'),
		('liquidity', 'capital'),
		('losses', 'capital'),
		('spreads', 'capital'),
		('credit risk', 'spreads'),
		('credit risk', 'losses'),
		('concentrations', 'credit risk'),
		('ERM', 'credit risk'),
		('competitive pressure', 'credit risk'),
		('regulatory burden/expectation', 'expenses'),
		('liquidity', 'regulatory burden/expectation'),
		('reputational/legal risk', 'liquidity'),
		('spreads', 'liquidity'),
		('losses', 'spreads'),
		('losses', 'reputational/legal risk'),
		('losses', 'human talent'),
		('human talent', 'losses'),
		('reputational/legal risk', 'losses'),
		('operational risk', 'losses'),
		('human talent', 'operational risk'),
		('competitive pressure', 'human talent'),
		('competitive pressure', 'strategic risk'),
		('competitive pressure', 'technology'),
		('ERM', 'reputational/legal risk'),
		('strategic risk', 'reputational/legal risk'),
		('ERM', 'expenses'),
		('ERM', 'operational risk'),
		('ERM', 'strategic risk'),
		('technology', 'ERM'),
		('technology', 'operational risk'),
		('technology', 'expenses'),
		('technology', 'efficiency'),
		('efficiency', 'expenses')
	]
	
	edge_polarities = [
		'+', '-', '+', '+', '+', '+', '+', '+', '+', '+', '-', '-', '+', '-', '+', '+', '+', '+', '+', '-', '-', '-', '-', '-', '+', '+', '+', '-', '+', '+', '-', '-', '-', '+', '+', '+', '+', '+', '+', '-', '-', '+', '+', '-', '+', '+', '-', '-', '+', '+', '+', '+', '-'
	]
	`);
	
	// pyodide.runPython(`
	// # Create a graph from the edges
	// g = ig.Graph.TupleList(edges, directed=True)
	
	// # Assign the edge polarity to the graph
	// g.es["polarity"] = edge_polarities  # Ensure the attribute name matches
	
	// # Visualize the graph
	// layout = g.layout("fr")  # Fruchterman-Reingold layout
	
	// # Plot the graph
	// fig, ax = plt.subplots(figsize=(12, 12))  # Increase the figure size for better visibility
	// ig.plot(
	// 	g,
	// 	layout=layout,
	// 	vertex_label=g.vs["name"], 
	// 	vertex_size=30,  # Increase vertex size
	// 	vertex_label_size=10,  # Increase label font size
	// 	edge_color=['green' if polarity == '+' else 'red' for polarity in g.es["polarity"]],
	// 	target=ax,  # Plot directly on the matplotlib axis
	// 	bbox=(1000, 1000),  # Increase the bounding box size
	// 	margin=100  # Add margin to ensure text fits within the image
	// )
	// plt.show()
	// `);
	
	pyodide.runPython(`
	# Create a directed graph
	G = nx.DiGraph()
	
	for (node1, node2), polarity in zip(edges, edge_polarities):
		G.add_edge(node1, node2, polarity=polarity)
	
	# Find all simple cycles in the graph
	cycles = list(nx.simple_cycles(G))
	
	# Print the total number of cycles found
	print(f"Total cycles found: {len(cycles)}\\n")
	
	# Print each cycle
	for idx, cycle in enumerate(cycles, start=1):
		print(f"Cycle {idx}: {cycle}")
	`);
	
	pyodide.runPython(`
	# Function to determine the polarity of a cycle
	def cycle_polarity(cycle):
		polarity = '+'
		for i in range(len(cycle)):
			edge = (cycle[i], cycle[(i + 1) % len(cycle)])
			if G.edges[edge]['polarity'] == '-':
				polarity = '-' if polarity == '+' else '+'
		return polarity
	
	# Determine the polarity of each cycle and count them for each node
	node_cycles = {node: {'positive': 0, 'negative': 0} for node in G.nodes}
	for cycle in cycles:
		pol = cycle_polarity(cycle)
		for node in cycle:
			if pol == '+':
				node_cycles[node]['positive'] += 1
			else:
				node_cycles[node]['negative'] += 1
	
	# Creating a tabular representation
	max_node_length = max([len(node) for node in G.nodes]) + 2  # for padding
	
	header = f"{'Node'.ljust(max_node_length)} {'Positive Cycles'.rjust(15)} {'Negative Cycles'.rjust(15)} {'Total Cycles'.rjust(15)}"
	divider = '-' * len(header)
	
	table_rows = [header, divider]
	
	for node, counts in node_cycles.items():
		total = counts['positive'] + counts['negative']
		row = f"{node.ljust(max_node_length)} {str(counts['positive']).rjust(15)} {str(counts['negative']).rjust(15)} {str(total).rjust(15)}"
		table_rows.append(row)
	
	# Print the table
	table = "\\n".join(table_rows)
	print(table)
	`);
}		
main();