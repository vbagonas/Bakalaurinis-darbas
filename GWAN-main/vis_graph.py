import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# Load the pkl
with open('./GWAN-main/GWAN_master/datasets/HypoidGearRadius.pkl', 'rb') as f:
    graphs = pickle.load(f)

print(f'Total graphs: {len(graphs)}')
print(f'First graph: {graphs[0]}')
print(f'  Nodes: {graphs[0].num_nodes}')
print(f'  Edges: {graphs[0].num_edges}')
print(f'  Label: {graphs[0].y.item()}')

CLASS_NAMES = ['N', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6']  # adjust to your labels

fig, axes = plt.subplots(1, 7, figsize=(21, 4))

for cls in range(7):
    # Find first graph of this class
    graph = next(g for g in graphs if g.y.item() == cls)
    
    # Convert to networkx
    G = to_networkx(graph, to_undirected=True)
    
    ax = axes[cls]
    nx.draw_circular(G, ax=ax, 
                     with_labels=True,
                     node_color='steelblue',
                     node_size=500,
                     edge_color='gray',
                     font_color='white',
                     font_size=10)
    ax.set_title(f'Class {cls} ({CLASS_NAMES[cls]})\n'
                 f'{graph.num_edges} edges')

plt.suptitle('One graph per fault class', fontsize=14)
plt.tight_layout()
plt.savefig('graphs_per_class.png', dpi=150)
plt.show()