from transformers import AutoModel, AutoTokenizer
import json
import argparse
from tqdm.auto import tqdm
import pathlib
import torch

class StreamingDataset:
    def __init__(self, data_path, max_length=512, batch_size=32):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.max_length = max_length
        self.batch_size = batch_size
        self.data_path = data_path

    def _read_lines(self):
        """逐行生成数据，减少内存占用"""
        with open(self.data_path, 'r') as f:
            for line in f:
                yield json.loads(line)

    def batches(self):
        """动态生成batch"""
        batch = []
        batch_ids = []
        for data in self._read_lines():
            # 拼接title和abstract，处理abstract缺失情况
            text = data.get('abstract1', '')
            batch.append(text)
            batch_ids.append(data['PaperID'])
            
            if len(batch) == self.batch_size:
                # Tokenization
                inputs = self.tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length, 
                    return_tensors="pt"
                )
                yield inputs, batch_ids
                batch = []
                batch_ids = []
        
        # 处理最后一个不完整的batch
        if len(batch) > 0:
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            yield inputs, batch_ids

class SpecterModel:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = AutoModel.from_pretrained('allenai/specter').to(self.device)
        self.model.eval()

    def embed(self, inputs):
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :]  # CLS向量

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='JSON Lines格式的输入文件路径')
    parser.add_argument('--output', required=True, help='输出文件路径（JSON Lines格式）')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch大小')
    args = parser.parse_args()

    # 初始化模型和数据流
    model = SpecterModel()
    dataset = StreamingDataset(args.data_path, batch_size=args.batch_size)
    
    # 确保输出目录存在
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # 逐batch处理并写入结果
    with open(args.output, 'w') as fout:
        for inputs, batch_ids in tqdm(dataset.batches(), desc="Processing"):
            embeddings = model.embed(inputs)
            for paper_id, emb in zip(batch_ids, embeddings):
                # 直接写入文件，不保存到内存
                fout.write(json.dumps({
                    "PaperID": paper_id,
                    "embedding": emb.cpu().numpy().tolist()
                }) + '\n')

if __name__ == '__main__':
    main()