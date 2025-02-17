import os
import glob
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import average_precision_score
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

# 设置矩阵乘法精度为 'high' 以平衡性能和精度
torch.set_float32_matmul_precision('high')

class SubgraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = root
        self.file_list = glob.glob(os.path.join(root, "subgraph_*.pt"))
        self.file_list.sort()
        
        # 添加边权重归一化
        self.normalize_weights = True
    
    def len(self):  # PyG的Dataset基类要求的方法
        return len(self.file_list)
    
    def __len__(self):  # Python内置len()函数调用的方法
        return len(self.file_list)
    
    def get(self, idx):
        data = torch.load(self.file_list[idx], weights_only=False)
        if self.normalize_weights and hasattr(data, 'edge_attr'):
            # 对边权重进行最小-最大归一化到[0,1]范围
            min_weight = data.edge_attr.min()
            max_weight = data.edge_attr.max()
            data.edge_attr = (data.edge_attr - min_weight) / (max_weight - min_weight)
        return data

class WeightedBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pos_score, neg_score, edge_weight):
        """
        Weighted binary cross entropy loss
        
        Args:
            pos_score: Prediction scores for positive samples
            neg_score: Prediction scores for negative samples
            edge_weight: Weights of positive edges (normalized to [0,1])
        """
        # Combine scores and labels
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ])
        
        # Construct sample weights:
        # - For positive samples: use edge weights directly
        # - For negative samples: use mean of edge weights
        avg_weight = edge_weight.mean()
        sample_weights = torch.cat([
            edge_weight,
            avg_weight * torch.ones_like(neg_score)
        ])
        
        # Calculate weighted BCE loss
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

    def get_neg_edges(self, edge_set, num_nodes, num_samples, device):
        neg_edges = []
        while len(neg_edges) < num_samples:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u != v and (u, v) not in edge_set:
                neg_edges.append((u, v))
        return torch.tensor(neg_edges, dtype=torch.long).t().contiguous().to(device)

    def compute_ap(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).cpu()
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ]).cpu()
        return average_precision_score(labels, scores.detach())

    def training_step(self, batch):
        out = self(batch)
        pos_edges = batch.edge_index
        edge_weights = batch.edge_attr
        num_pos = pos_edges.size(1)
        
        edge_set = set(zip(pos_edges[0].cpu().numpy(), pos_edges[1].cpu().numpy()))
        neg_edges = self.get_neg_edges(edge_set, batch.x.size(0), num_pos, batch.x.device)

        pos_score = (out[pos_edges[0]] * out[pos_edges[1]]).sum(dim=1)
        neg_score = (out[neg_edges[0]] * out[neg_edges[1]]).sum(dim=1)

        loss = self.loss_fn(pos_score, neg_score, edge_weights)
        
        with torch.no_grad():
            ap = self.compute_ap(pos_score, neg_score)
            self.log('train_ap', ap, prog_bar=True, batch_size=batch.num_nodes)
        
        self.log('train_loss', loss, prog_bar=True, batch_size=batch.num_nodes)
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
        ap = self.compute_ap(pos_score, neg_score)
        
        self.log('val_loss', loss, batch_size=batch.num_nodes)
        self.log('val_ap', ap, batch_size=batch.num_nodes)
        return {'val_loss': loss, 'val_ap': ap}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.001)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-5
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': sched,
                'monitor': 'train_loss'
            }
        }

# 训练配置
data_dir = "/home/lyuzhuoqi/projects/clustering/data/2010s/GNN/samples"
ckpt_dir = '/home/lyuzhuoqi/projects/clustering/representations/GNN/checkpoints'

# 数据加载
dataset = SubgraphDataset(root=data_dir)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=64,
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

# 检查点回调
ckpt_callback = ModelCheckpoint(
    monitor='train_loss',
    dirpath=ckpt_dir,
    filename='gat-{epoch:02d}-{train_loss:.2f}',
    save_top_k=3,
    mode='min'
)

# 训练器设置
trainer = L.Trainer(
    accelerator='gpu',
    devices=torch.cuda.device_count(),
    max_epochs=10,
    callbacks=[
        ModelCheckpoint(
            monitor='val_loss',  # 改为监控验证损失
            dirpath=ckpt_dir,
            filename='gnn-{epoch:02d}-{val_loss:.2f}-{val_ap:.3f}',
            save_top_k=3,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )
    ],
    precision='16-mixed'
)

# 模型训练
model = WeightedGAT(
    in_dim=100,          # 输入特征维度
    hidden_dim=256,      # 隐藏层维度
    out_dim=128,   # 输出嵌入维度
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)