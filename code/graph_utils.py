import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('nome_file.csv')
G = nx.Graph()
G.add_nodes_from(df['nodo'].unique())

nx.draw(G, with_labels=True)
plt.show()
