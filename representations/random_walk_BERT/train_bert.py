from itertools import islice
import pickle
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaForMaskedLM, RobertaConfig
from torch.optim import AdamW
import pickle

# 定义数据集
class CitationTrailDataset(Dataset):
    def __init__(self, citation_trails, pid_to_idx, max_length=10, pad_value=0):
        self.citation_trails = citation_trails
        self.pid_to_idx = pid_to_idx
        self.max_length = max_length
        self.pad_value = pad_value

    def __len__(self):
        return len(self.citation_trails)

    def __getitem__(self, idx):
        trail = self.citation_trails[idx]
        trail = [self.pid_to_idx.get(pid, self.pad_value) for pid in trail] # 未知 PID (出现次数小于min_count) 用 pad_value 替换
        if len(trail) > self.max_length:
            trail = trail[:self.max_length]
        trail += [self.pad_value] * (self.max_length - len(trail))
        return torch.tensor(trail, dtype=torch.long)

# PyTorch Lightning 模型
class CitationTrailModel(pl.LightningModule):
    def __init__(self, pid_to_idx, max_length=10, learning_rate=5e-5, batchsize=2048):
        super().__init__()
        # 配置模型
        num_pids = len(pid_to_idx)
        self.pad_value = pid_to_idx['<pad>']
        
        # 初始化模型
        config = RobertaConfig(
            vocab_size=num_pids,
            pad_token_id=self.pad_value,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=514,
        )
        self.model = RobertaForMaskedLM(config=config)
        
        # 初始化优化器的学习率
        self.learning_rate = learning_rate

        # 训练用的数据集
        self.pid_to_idx = pid_to_idx
        self.max_length = max_length
        self.batchsize = batchsize

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch):
        # 提取输入
        inputs = batch
        attention_mask = (inputs != self.pad_value).long()
        
        # 准备标签
        labels = inputs.clone()
        rand = torch.rand(inputs.shape, device='cuda')
        mask_arr = (rand < 0.15) * (labels != self.pad_value)

        # 如果没有被掩盖的 token，手动掩盖一个
        if mask_arr.sum() == 0:
            non_pad_indices = (labels != self.pad_value).nonzero(as_tuple=False)
            if non_pad_indices.numel() > 0:
                idx = non_pad_indices[torch.randint(0, non_pad_indices.size(0), (1,)).item()]
                mask_arr[idx[0], idx[1]] = True

        # 更新 labels
        labels[~mask_arr] = -100

        # 替换被掩盖的 token 为 <mask>
        inputs[mask_arr] = self.pid_to_idx['<mask>']

        # 前向传播
        outputs = self.model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        with open("citation_trails.pkl", "rb") as f:
            citation_trail = pickle.load(f)
        dataset = CitationTrailDataset(citation_trail, self.pid_to_idx, max_length=self.max_length, pad_value=self.pid_to_idx['<pad>'])
        dataloader = DataLoader(dataset, batch_size=self.batchsize, shuffle=True)
        return dataloader

    def on_epoch_end(self):
        # 可以在每个 epoch 结束时打印一些信息
        pass

def main(resume_ckpt=None, lr=5e-5, batchsize=2048, min_count=50): 
    # 读取 PID 到索引的映射
    print("Loading PID to index mapping...")
    with open(f"pid_to_idx_min_count_{min_count}.pkl", "rb") as f:
        pid_to_idx = pickle.load(f)
    print(f"PID {len(pid_to_idx)} mapped to model input IDs:", dict(islice(pid_to_idx.items(), 10)))  # 仅打印前10项

    # 初始化模型
    model = CitationTrailModel(pid_to_idx, learning_rate=lr, batchsize=batchsize)

    # 使用 PyTorch Lightning 的 Trainer 来训练模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./checkpoints/min_count_{min_count}',
        filename='checkpoint-{epoch}',
        save_top_k=-1,  # 保存所有 epoch 的 checkpoint
        every_n_epochs=1,
    )
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator='gpu',
        callbacks=[checkpoint_callback],
        )
    if resume_ckpt is not None:
        resume_ckpt = f"./checkpoints/checkpoint-{resume_ckpt}"
    # 开始训练
    trainer.fit(model, ckpt_path=resume_ckpt)

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Train a BERT model on citation trails")
    parser.add_argument('--resume_ckpt', type=int, default=None, help='Epoch number to resume from')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--batchsize', type=int, default=2048, help='Batch size for training')
    parser.add_argument('--min_count', type=int, default=50, help='Minimum count of a PID to be included in the model')
    args = parser.parse_args()

    # 调用 main 函数并传递参数
    main(args.resume_ckpt, args.lr, args.batchsize, args.min_count)
