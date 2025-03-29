from transformers import AutoModel, AutoTokenizer
import json
import argparse
from tqdm.auto import tqdm
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
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
            text = data.get('abstract')
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
    
    # 使用PyArrow写入Parquet
    schema = pa.schema([
        ('PaperID', pa.string()),
        ('embedding', pa.list_(pa.float32()))  # 使用float32
    ])
    
    writer = pq.ParquetWriter(
        args.output, 
        schema, 
        compression='ZSTD',  # 使用Zstandard压缩
        flavor='spark'
    )
    
    for inputs, batch_ids in tqdm(dataset.batches(), total=23322430 // args.batch_size):
        embeddings = model.embed(inputs).cpu().numpy().astype(np.float32)
        
        # 转换为PyArrow数组
        ids_array = pa.array(batch_ids)
        emb_array = pa.array(embeddings.tolist())  # 转换为Python list
        
        # 创建RecordBatch
        batch = pa.RecordBatch.from_arrays(
            [ids_array, emb_array],
            schema=schema
        )
        
        # 直接写入批次数据
        writer.write_batch(batch)
    
    # 关闭写入器
    writer.close()

if __name__ == '__main__':
    main()