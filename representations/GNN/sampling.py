import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import subgraph
import os
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import json
import scipy.sparse as sp
from gensim.models import KeyedVectors

class GraphSampler:
    def __init__(
        self,
        num_samples=100,     # Number of subgraphs to sample
        cache_dir='samples'   # Directory to store sampled subgraphs
    ):
        self.num_samples = num_samples
        self.cache_dir = cache_dir
        
        # Create cache directory if it does not exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Statistics collection
        self.node_counts = []   # Node counts per subgraph
        self.edge_counts = []   # Edge counts per subgraph
        
    def sample_subgraph(self, data):
        """
        Sample a single subgraph using random sampling.
        
        This method randomly selects 1% of the total nodes from the graph
        and generates the induced subgraph containing all edges among the selected nodes.
        """
        num_nodes_total = data.x.size(0)
        
        # Randomly select 1% of nodes as the subgraph (minimum 1 node)
        num_sample = max(1, num_nodes_total // 100)
        perm = torch.randperm(num_nodes_total)[:num_sample]
        
        # Extract the induced subgraph for the selected nodes
        edge_index, _ = subgraph(perm, data.edge_index, relabel_nodes=True)
        
        # Record statistics (node and edge counts)
        self.node_counts.append(len(perm))
        self.edge_counts.append(edge_index.size(1))
        
        # Construct the subgraph Data object
        subgraph_data = Data(
            x=data.x[perm],
            edge_index=edge_index,
            original_indices=perm  # Save original node indices
        )
        
        return subgraph_data

    def save_histograms(self):
        """Plot and save histograms of node and edge distributions."""
        # Plot histogram for node counts
        plt.figure(figsize=(8, 6))
        plt.hist(self.node_counts, bins=50, color='skyblue', edgecolor='black')
        plt.title('Distribution of Node Counts')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Frequency')
        node_histogram_path = os.path.join(self.cache_dir, 'node_distribution.png')
        plt.savefig(node_histogram_path)
        plt.close()
        
        # Plot histogram for edge counts
        plt.figure(figsize=(8, 6))
        plt.hist(self.edge_counts, bins=50, color='salmon', edgecolor='black')
        plt.title('Distribution of Edge Counts')
        plt.xlabel('Number of Edges')
        plt.ylabel('Frequency')
        edge_histogram_path = os.path.join(self.cache_dir, 'edge_distribution.png')
        plt.savefig(edge_histogram_path)
        plt.close()
    
    def sample_and_save(self, data):
        """
        Sample subgraphs from the original graph and save them,
        while collecting statistics and saving histograms.
        """
        for i in tqdm(range(self.num_samples)):
            subgraph_data = self.sample_subgraph(data)
            save_path = os.path.join(self.cache_dir, f'subgraph_{i}.pt')
            torch.save(subgraph_data, save_path)
        
        # Save histograms
        self.save_histograms()


def load_large_graph():
    """
    Load the large graph using pre-stored node vectors from gensim.models.KeyedVectors.
    
    Steps:
    1. Load the vid_to_index mapping which maps a vertex ID (VID) to its index in the citation matrix.
    2. Load the citation matrix (sparse format).
    3. Load the pre-trained node vectors from gensim.
    4. Filter nodes that do not have a representation in the keyed vectors.
    5. Adjust the citation matrix and node features accordingly.
    """
    # Load the vertex id to index mapping
    mapping_path = os.path.join('data', 'direct_citation', 'vid_to_index.json')
    with open(mapping_path, 'r') as f:
        vid_to_index = json.load(f)
    print("VID to index mapping loaded successfully!")
    
    # Load KeyedVectors (assumed to be stored at the following path)
    vectors_path = os.path.join('data', 'Word2Vec', 'vectors.kv')
    kv_model = KeyedVectors.load(vectors_path, mmap='r')
    print("Gensim KeyedVectors loaded successfully!")
    
    # Identify valid nodes: those whose vertex id exists in the KeyedVectors
    valid_vids = []
    valid_old_indices = []  # original indices in the citation matrix
    for vid, idx in vid_to_index.items():
        # Only use nodes that have a vector in the keyed vectors
        if int(float(vid)) in kv_model:
            valid_vids.append(int(float(vid)))
            valid_old_indices.append(idx)
    
    valid_old_indices = np.array(sorted(valid_old_indices))
    print(f"Filtered valid nodes: {len(valid_old_indices)} out of {len(vid_to_index)}")
    
    # Load the sparse citation matrix
    loaded_data = np.load(os.path.join('data', 'direct_citation', 'citation_matrix.npz'))
    num_nodes_total = len(vid_to_index)
    citation_matrix = sp.coo_matrix(
        (loaded_data['data'], (loaded_data['row'], loaded_data['col'])),
        shape=(num_nodes_total, num_nodes_total)
    )
    print("Citation matrix loaded successfully!")
    
    # Filter the citation matrix to only include rows and columns corresponding to valid nodes
    citation_matrix = citation_matrix.tocsr()[valid_old_indices, :][:, valid_old_indices]
    citation_matrix = citation_matrix.tocoo()
    
    # Build edge_index from the filtered citation matrix
    edge_index = torch.tensor(np.array([citation_matrix.row, citation_matrix.col]), dtype=torch.long)
    edge_attr = torch.tensor(citation_matrix.data, dtype=torch.float)
    
    # Build node features from keyed vectors for valid nodes.
    # Create a new sorted mapping: For each valid old index, find the corresponding vid,
    # then get the vector from the KeyedVectors.
    # We need an inverse mapping from original index to vertex id.
    index_to_vid = {int(idx): int(float(vid)) for vid, idx in vid_to_index.items()}
    node_features = []
    for idx in valid_old_indices:
        vid = index_to_vid[int(idx)]
        vec = kv_model[vid]
        node_features.append(vec)
    
    node_features = torch.tensor(np.vstack(node_features), dtype=torch.float)
    
    # Create a PyG Data object with the filtered citation matrix and corresponding features.
    data = Data(edge_index=edge_index, edge_attr=edge_attr, x=node_features)
    
    return data

def main():
    print("Loading large graph...")
    data = load_large_graph()
    print("Large graph loaded successfully!")
    
    sampler = GraphSampler(
        num_samples=10000,
        cache_dir='samples'
    )
    
    sampler.sample_and_save(data)

if __name__ == '__main__':
    main()