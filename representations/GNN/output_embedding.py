import torch
# 设置矩阵乘法精度为 'high' 以平衡性能和精度
torch.set_float32_matmul_precision('high')
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import random
import lightning as L

class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, margin=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(self, pos_score, neg_score, edge_weight):
        # 存在性损失
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ])
        exist_loss = self.bce(scores, labels)
        
        # 权重预测损失
        weight_pred = torch.sigmoid(pos_score)
        weight_loss = F.mse_loss(weight_pred, edge_weight)
        
        # 排序损失
        rank_loss = torch.mean(
            torch.relu(
                self.margin - pos_score.unsqueeze(1) + neg_score.unsqueeze(0)
            ) * edge_weight.unsqueeze(1)
        )
        
        return exist_loss + self.alpha * weight_loss + self.beta * rank_loss
    
class DirectedGAT(L.LightningModule):
    def __init__(self, in_dim, out_dim, heads=8, dropout=0.2):
        super().__init__()
        # 添加dropout和层归一化
        self.dropout = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(out_dim * heads)
        self.norm2 = torch.nn.LayerNorm(out_dim)
        
        # GAT层保持不变
        self.in_conv1 = GATConv(in_dim, out_dim, heads=heads, edge_dim=1)
        self.out_conv1 = GATConv(in_dim, out_dim, heads=heads, edge_dim=1)
        self.in_conv2 = GATConv(out_dim * heads, out_dim, heads=1, concat=False, edge_dim=1)
        self.out_conv2 = GATConv(out_dim * heads, out_dim, heads=1, concat=False, edge_dim=1)
        
        # 改进特征合并层
        self.merge = torch.nn.Sequential(
            torch.nn.Linear(out_dim * 2, out_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(out_dim, out_dim)
        )
        self.loss_fn = CombinedLoss()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # 处理入边
        x_in = F.relu(self.in_conv1(x, edge_index, edge_weight))
        x_in = self.in_conv2(x_in, edge_index, edge_weight)
        
        # 处理出边（翻转边的方向）
        edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        x_out = F.relu(self.out_conv1(x, edge_index_rev, edge_weight))
        x_out = self.out_conv2(x_out, edge_index_rev, edge_weight)
        
        # 合并入边和出边的特征
        x = self.merge(torch.cat([x_in, x_out], dim=-1))
        return x

    def get_neg_edges(self, edge_set, num_nodes, num_samples, device):
        neg_edges = []
        
        # 基于正边的权重分布进行负采样
        while len(neg_edges) < num_samples:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u != v and (u, v) not in edge_set:  # 只检查单向
                neg_edges.append((u, v))
                
        return torch.tensor(neg_edges, dtype=torch.long).t().contiguous().to(device)
    
    def compute_auc(self, pos_score, neg_score):
        """计算AUC分数"""
        scores = torch.cat([pos_score, neg_score]).cpu()
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ]).cpu()
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, scores.detach())

    def training_step(self, batch):
        out = self(batch)
        pos_edges = batch.edge_index
        edge_weights = batch.edge_attr
        num_pos = pos_edges.size(1)
        
        # 构建方向感知的边集合
        edge_set = set(zip(pos_edges[0].cpu().numpy(), pos_edges[1].cpu().numpy()))
        neg_edges = self.get_neg_edges(edge_set, batch.x.size(0), num_pos, batch.x.device)

        # 计算方向感知的分数
        pos_score = (out[pos_edges[0]] * out[pos_edges[1]]).sum(dim=1)
        neg_score = (out[neg_edges[0]] * out[neg_edges[1]]).sum(dim=1)

        # 计算组合损失
        loss = self.loss_fn(pos_score, neg_score, edge_weights)
        
        # 记录额外的指标
        with torch.no_grad():
            auc = self.compute_auc(pos_score, neg_score)
            self.log('train_auc', auc, prog_bar=True)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        out = self(batch)
        pos_edges = batch.edge_index
        edge_weights = batch.edge_attr
        num_pos = pos_edges.size(1)
        
        edge_set = set(zip(pos_edges[0].cpu().numpy(), pos_edges[1].cpu().numpy()))
        neg_edges = self.get_neg_edges(edge_set, batch.x.size(0), num_pos, batch.x.device)
        
        pos_score = (out[pos_edges[0]] * out[pos_edges[1]]).sum(dim=1)
        neg_score = (out[neg_edges[0]] * out[neg_edges[1]]).sum(dim=1)
        
        loss = self.loss_fn(pos_score, neg_score, edge_weights)
        auc = self.compute_auc(pos_score, neg_score)
        
        self.log('val_loss', loss)
        self.log('val_auc', auc)
        return {'val_loss': loss, 'val_auc': auc}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.01)  # 降低学习率
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 
            mode='min',
            factor=0.5,
            patience=5,  # 减少patience以更快响应
            min_lr=1e-5
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': sched,
                'monitor': 'train_loss'
            }
        }
    
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
test_graph = load_large_graph()  # 替换成你的图文件路径

# 2. 加载训练好的模型
checkpoint_path = 'data/GNN/directed_gat-epoch=18-val_loss=1.26-val_auc=0.935.ckpt'
# 使用保存的最佳模型

trained_model = DirectedGAT.load_from_checkpoint(
    checkpoint_path,
    in_dim=100,
    out_dim=128,
    heads=4,
    dropout=0.1
)

# 3. 获取节点embedding
embeddings = get_node_embeddings(trained_model, test_graph)

# 4. 保存embedding（可选）
torch.save(embeddings, 'GNN_embeddings.pt')

# 5. 查看embedding的形状
print(f"Node embeddings shape: {embeddings.shape}")  # 应该是 [节点数量, 128]