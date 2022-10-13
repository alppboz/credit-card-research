import itertools
from itertools import combinations
import networkx as nx
file = open("c_m_unique.txt")
data = [line.strip() for line in file.readlines()]
merchant_data = set(line.strip() for line in open('merchant_id.txt'))
merchants = set([])
c_map = {}

for line in data:
    line = line.split(',')
    c = line[0]
    for i in range(1, len(line)):
        m = line[i].strip()
        if m in merchant_data:
            merchants.add(m)
            if m not in c_map:
                c_map[m] = set([])
            c_map[m].add(c)

file.close()

G = nx.Graph()

for m1, m2 in combinations(merchants, 2):
    c1 = c_map[m1]
    c2 = c_map[m2]
    common = c1.intersection(c2)
    if common:
        G.add_edge(m1, m2, weight=len(common))

		
degrees = G.degree()
c_e = nx.eigenvector_centrality(G, max_iter=100, tol=1e-06, nstart=None, weight='weight')
c_c = nx.closeness_centrality(G, u=None, distance='weight')
c_b = nx.betweenness_centrality(G, k=None, normalized=True, weight='weight', endpoints=False, seed=None) 


with open("results.txt", 'w') as results:
    results.write('Merchant_id' + ',Degree' + ',Betweenness_Centrality' + ',Closeness_Centrality' + ',Eigenvector_Centrality' + 'Adjacent_Nodes_Count'+ ',Adjacent_Nodes\n')
    for m in G.nodes():
        results.write(m + ',' + str(degrees[m]) + ',' + str(c_b[m]) + ',' + str(c_c[m]) + ',' + str(c_e[m]) + ',' + str(len([n for n in G[m]])) +',' + str([n for n in G[m]]) + '\n')
results.close()