import os
import glob
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
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
            # 对边权重进行归一化
            data.edge_attr = (data.edge_attr - data.edge_attr.mean()) / data.edge_attr.std()
        return data

class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, margin=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(self, pos_score, neg_score, edge_weight):
        # existence loss
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ])
        exist_loss = self.bce(scores, labels)
        
        # weight prediction loss
        weight_pred = torch.sigmoid(pos_score)
        weight_loss = F.mse_loss(weight_pred, edge_weight)
        
        # ranking loss
        rank_loss = torch.mean(
            torch.relu(
                self.margin - pos_score.unsqueeze(1) + neg_score.unsqueeze(0)
            ) * edge_weight.unsqueeze(1)
        )
        
        return exist_loss + self.alpha * weight_loss + self.beta * rank_loss

class SimpleDirectedGNN(L.LightningModule):
    def __init__(self, in_dim, hidden_dim, embedding_dim, dropout=0.1):
        super().__init__()
        
        # 基础层
        self.encoder = nn.Linear(in_dim, hidden_dim)
        
        # 有向图卷积层
        self.conv = DirectedConv(hidden_dim, hidden_dim)
        
        # 输出层
        self.output = nn.Linear(hidden_dim, embedding_dim)
        
        self.dropout = dropout
        self.loss_fn = CombinedLoss()
        
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # 初始特征编码
        x = self.encoder(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 处理入边和出边
        x_in = self.conv(x, edge_index, edge_weight)
        x_out = self.conv(x, edge_index.flip([0]), edge_weight)
        
        # 合并并生成最终嵌入
        x = x_in + x_out
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.output(x)

    def get_neg_edges(self, edge_set, num_nodes, num_samples, device):
        neg_edges = []
        while len(neg_edges) < num_samples:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u != v and (u, v) not in edge_set:
                neg_edges.append((u, v))
        return torch.tensor(neg_edges, dtype=torch.long).t().contiguous().to(device)

    def compute_auc(self, pos_score, neg_score):
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
        
        edge_set = set(zip(pos_edges[0].cpu().numpy(), pos_edges[1].cpu().numpy()))
        neg_edges = self.get_neg_edges(edge_set, batch.x.size(0), num_pos, batch.x.device)

        pos_score = (out[pos_edges[0]] * out[pos_edges[1]]).sum(dim=1)
        neg_score = (out[neg_edges[0]] * out[neg_edges[1]]).sum(dim=1)

        loss = self.loss_fn(pos_score, neg_score, edge_weights)
        
        with torch.no_grad():
            auc = self.compute_auc(pos_score, neg_score)
            self.log('train_auc', auc, prog_bar=True, batch_size=batch.num_nodes)
        
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
        auc = self.compute_auc(pos_score, neg_score)
        
        self.log('val_loss', loss, batch_size=batch.num_nodes)
        self.log('val_auc', auc, batch_size=batch.num_nodes)
        return {'val_loss': loss, 'val_auc': auc}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.01)
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

class DirectedConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        
        self.linear = nn.Linear(in_channels, out_channels)
        self.edge_encoder = nn.Linear(1, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        x = self.linear(x)
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        
        edge_embedding = self.edge_encoder(edge_weight.unsqueeze(-1))
        
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)
    
    def message(self, x_j, edge_attr):
        return x_j * edge_attr

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
    batch_size=8, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=8,
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
    max_epochs=5,
    callbacks=[
        ModelCheckpoint(
            monitor='val_loss',  # 改为监控验证损失
            dirpath=ckpt_dir,
            filename='gnn-{epoch:02d}-{val_loss:.2f}-{val_auc:.3f}',
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
model = SimpleDirectedGNN(
    in_dim=100,          # 输入特征维度
    hidden_dim=256,      # 隐藏层维度
    embedding_dim=128,   # 输出嵌入维度
    dropout=0.1
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)