import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
from torch_geometric.nn import GATConv

# 设置矩阵乘法精度为 'high' 以平衡性能和精度
torch.set_float32_matmul_precision('high')

class WeightedBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pos_score, neg_score, edge_weight):
        """
        加权的二分类交叉熵损失
        
        Args:
            pos_score: 正样本的预测分数
            neg_score: 负样本的预测分数
            edge_weight: 正样本边的权重 (已归一化到[0,1])
        """
        # 合并分数和标签
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ])
        
        # 构造样本权重:
        # - 对于正样本：直接使用边权重
        # - 对于负样本：使用边权重的平均值
        avg_weight = edge_weight.mean()
        sample_weights = torch.cat([
            edge_weight,
            avg_weight * torch.ones_like(neg_score)
        ])
        
        # 计算加权BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(
            scores, 
            labels,
            weight=sample_weights,
            reduction='sum'
        ) / sample_weights.sum()
        
        return bce_loss

class WeightedGAT(L.LightningModule):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.6):
        super().__init__()
        # 边权重编码器
        self.weight_encoder = nn.Linear(1, heads)
        
        # GAT层
        self.gat_in = GATConv(
            in_dim, 
            hidden_dim // heads,
            heads=heads,
            add_self_loops=False
        )
        self.gat_out = GATConv(
            in_dim, 
            hidden_dim // heads,
            heads=heads,
            add_self_loops=False
        )
        self.gat2 = GATConv(
            hidden_dim,
            out_dim,
            heads=1,
            add_self_loops=False
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            )
        self.loss_fn = WeightedBCELoss()
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        # 处理边权重
        edge_weight = self.weight_encoder(edge_weight.unsqueeze(-1))
        
        # GAT层
        x1 = F.elu(self.gat_in(x, edge_index, edge_weight))
        x2 = F.elu(self.gat_out(x, edge_index.flip([0]), edge_weight))

        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        # 合并双向信息
        x_combined = torch.cat([x1, x2], dim=-1)
        x = self.combine_mlp(x_combined)
        return self.gat2(x, edge_index)

    
def get_node_embeddings(model, graph_data):
    """
    使用CPU获取图中所有节点的embedding
    
    参数:
        model: 训练好的DirectedGAT模型
        graph_data: PyG的Data对象，包含图的信息
        
    返回:
        node_embeddings: 张量，形状为 [节点数量, embedding维度]
    """
    # 将模型移动到CPU并设置为评估模式
    model = model.cpu()
    model.eval()
    
    # 确保数据在CPU上
    graph_data = graph_data.cpu()
    
    # 使用torch.no_grad()避免计算梯度
    with torch.no_grad():
        # 获取节点embedding
        node_embeddings = model(graph_data)
        
    return node_embeddings

import os
import json
from gensim.models import KeyedVectors
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data

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
    mapping_path = 'data/direct_citation/vid_to_index.json'
    with open(mapping_path, 'r') as f:
        vid_to_index = json.load(f)
    print("VID to index mapping loaded successfully!")
    
    # Load KeyedVectors (assumed to be stored at the following path)

    vectors_path = 'data/Word2Vec/vectors.kv'
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
    loaded_data = np.load('data/direct_citation/citation_matrix.npz')
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

# 使用示例：
# 1. 加载一个图
test_graph = load_large_graph()

# 2. 加载训练好的模型
checkpoint_path = 'data/GNN/gnn-epoch=09-val_loss=0.38-val_ap=0.924.ckpt'
# 使用保存的最佳模型

trained_model = WeightedGAT.load_from_checkpoint(
    checkpoint_path,
    in_dim=100,          # 输入特征维度
    hidden_dim=256,      # 隐藏层维度
    out_dim=128,   # 输出嵌入维度
)

# 3. 获取节点embedding
embeddings = get_node_embeddings(trained_model, test_graph)

# 4. 保存embedding（可选）
torch.save(embeddings, 'GNN_embeddings.pt')

# 5. 查看embedding的形状
print(f"Node embeddings shape: {embeddings.shape}")  # 应该是 [节点数量, 128]