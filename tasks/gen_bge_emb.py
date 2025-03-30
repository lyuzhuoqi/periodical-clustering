import json
import argparse
from tqdm.auto import tqdm
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from fastembed import TextEmbedding
from typing import Iterator, Tuple, List
import torch

class GPUEmbeddingPipeline:
    """简化的GPU嵌入生成管道"""
    def __init__(self):
        """
        初始化GPU嵌入模型
        """
        # 使用小型模型并启用CUDA
        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._warmup()
        
    def _warmup(self):
        """预热GPU"""
        dummy_text = ["warmup"] * 4
        list(self.model.embed(dummy_text))
        torch.cuda.empty_cache()
        
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """处理文本批次"""
        embeddings = np.array(list(self.model.embed(texts)))
        return embeddings.astype(np.float32)

def stream_documents(data_path: str) -> Iterator[Tuple[str, str]]:
    """流式生成(PaperID, abstract)对"""
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            yield data['PaperID'], data.get('abstract', '')

def main():
    parser = argparse.ArgumentParser(description='简化的GPU嵌入生成')
    parser.add_argument('--data-path', required=True, help='JSONL输入文件路径')
    parser.add_argument('--output', required=True, help='输出Parquet路径')
    parser.add_argument('--batch-size', type=int, default=512, help='处理批量大小')
    args = parser.parse_args()

    # 初始化
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pipeline = GPUEmbeddingPipeline()
    document_stream = stream_documents(args.data_path)

    # 获取样本确定维度
    sample_id, sample_text = next(stream_documents(args.data_path))
    embedding_dim = len(pipeline.embed_batch([sample_text])[0])
    schema = pa.schema([
        ('PaperID', pa.string()),
        ('embedding', pa.list_(pa.float32(), embedding_dim))
    ])

    # 创建Parquet写入器
    writer = pq.ParquetWriter(
        args.output,
        schema=schema,
        compression='ZSTD',
        flavor='spark'
    )

    # 处理循环
    batch_ids = []
    batch_texts = []
    with tqdm(desc="生成嵌入", unit="doc", total=23322430) as pbar:
        for paper_id, text in document_stream:
            batch_ids.append(paper_id)
            batch_texts.append(text)
            
            if len(batch_texts) >= args.batch_size:
                # 处理批次
                embeddings = pipeline.embed_batch(batch_texts)
                batch = pa.RecordBatch.from_arrays([
                    pa.array(batch_ids),
                    pa.array(embeddings.tolist())
                ], schema=schema)
                writer.write_batch(batch)
                
                # 更新进度
                pbar.update(len(batch_ids))
                batch_ids = []
                batch_texts = []
                torch.cuda.empty_cache()

        # 处理最后一批
        if batch_texts:
            embeddings = pipeline.embed_batch(batch_texts)
            batch = pa.RecordBatch.from_arrays([
                pa.array(batch_ids),
                pa.array(embeddings.tolist())
            ], schema=schema)
            writer.write_batch(batch)
            pbar.update(len(batch_ids))

    writer.close()
    print(f"嵌入已保存至 {args.output}")

if __name__ == '__main__':
    assert torch.cuda.is_available(), "需要CUDA环境"
    main()