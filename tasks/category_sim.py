import numpy as np
import json
import pyarrow.parquet as pq
import pandas as pd
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import os
import argparse

class CategorySimilarityAnalyzer:
    def __init__(self, embedding_path: str, label_path: str, n_trials=1000, sample_ratio=0.001):
        self.embedding_path = embedding_path
        self.label_path = label_path
        self.n_trials = n_trials
        self.sample_ratio = sample_ratio
        
        # 加载全量数据
        print("Loading base data...")
        self._load_full_data()
        
        # 初始化结果容器
        self.metrics = {
            'intra': np.zeros((n_trials, len(self.unique_labels))),
            'inter': np.zeros((n_trials, len(self.unique_labels))),
            'disc': np.zeros((n_trials, len(self.unique_labels))),
            'heatmap': np.zeros((len(self.unique_labels), len(self.unique_labels), n_trials))
        }

    def _load_full_data(self):
        """加载全量数据并进行预处理"""
        # 加载嵌入向量
        embeddings_df = pq.read_table(self.embedding_path).to_pandas()
        embeddings_df.set_index('PaperID', inplace=True)
        print("Loaded embeddings data:", len(embeddings_df), "papers")
        
        # 加载标签数据
        with open(self.label_path, 'r') as f:
            label_dict = json.load(f)
        labels_df = pd.DataFrame(list(label_dict.items()), 
                               columns=['PaperID', 'label']).set_index('PaperID')
        print("Loaded labels data:", len(labels_df), "papers")
        
        # 合并数据并进行归一化
        self.full_df = embeddings_df.join(labels_df, how='inner').dropna()
        # 预处理嵌入向量
        self.full_df['embedding'] = self.full_df['embedding'].apply(
            lambda x: x / np.linalg.norm(x))  # 预归一化
        
        self.unique_labels = sorted(self.full_df['label'].unique())
        self.label_to_idx = {label: i for i, label in enumerate(self.unique_labels)}
        print(f"Loaded data with {len(self.full_df)} papers and {len(self.unique_labels)} categories")

    def run_experiment(self):
        """执行全量相似度分析实验"""
        for trial in tqdm(range(self.n_trials), desc="Processing trials"):
            # 随机采样
            sampled = self.full_df.sample(frac=self.sample_ratio, random_state=trial)
            
            # 转换为numpy数组
            sample_emb = np.stack(sampled['embedding'].values)
            sample_labels = sampled['label'].map(self.label_to_idx).values
            
            # 全量相似度计算
            sim_matrix = sample_emb @ sample_emb.T  # 余弦相似度矩阵
            
            # 计算指标
            self._calculate_metrics(trial, sim_matrix, sample_labels)
            
            # 计算热图数据
            self._calculate_heatmap(trial, sample_emb, sample_labels)

    def _calculate_metrics(self, trial: int, sim_matrix: np.ndarray, labels: np.ndarray):
        """计算单次试验的完整指标"""
        for label_idx, label in enumerate(self.unique_labels):
            mask = (labels == label_idx)
            n_samples = np.sum(mask)
            
            if n_samples < 2:
                # 样本不足时填充NaN
                self.metrics['intra'][trial, label_idx] = np.nan
                self.metrics['inter'][trial, label_idx] = np.nan
                self.metrics['disc'][trial, label_idx] = np.nan
                continue
            
            # 类内相似度（排除对角线）
            intra_mask = np.outer(mask, mask)
            np.fill_diagonal(intra_mask, False)
            intra_values = sim_matrix[intra_mask]
            
            # 类间相似度
            inter_mask = np.outer(mask, ~mask)
            inter_values = sim_matrix[inter_mask]
            
            # 记录指标
            self.metrics['intra'][trial, label_idx] = np.nanmean(intra_values)
            self.metrics['inter'][trial, label_idx] = np.nanmean(inter_values)
            self.metrics['disc'][trial, label_idx] = (self.metrics['intra'][trial, label_idx] - 
                                                     self.metrics['inter'][trial, label_idx])

    def _calculate_heatmap(self, trial: int, embeddings: np.ndarray, labels: np.ndarray):
        """计算类别间相似度热图"""
        category_embeddings = []
        for label_idx in range(len(self.unique_labels)):
            mask = (labels == label_idx)
            if np.sum(mask) > 0:
                category_embeddings.append(embeddings[mask].mean(axis=0))
            else:
                category_embeddings.append(np.zeros(embeddings.shape[1]))
        
        heatmap = np.dot(category_embeddings, np.array(category_embeddings).T)
        self.metrics['heatmap'][:, :, trial] = heatmap

    def report_results(self):
        """生成统计报告和可视化结果"""
        # 计算统计量
        valid_trials = ~np.isnan(self.metrics['intra'])
        intra_means = np.nanmean(self.metrics['intra'], axis=0)
        intra_stds = np.nanstd(self.metrics['intra'], axis=0)
        inter_means = np.nanmean(self.metrics['inter'], axis=0)
        inter_stds = np.nanstd(self.metrics['inter'], axis=0)
        disc_means = np.nanmean(self.metrics['disc'], axis=0)
        disc_stds = np.nanstd(self.metrics['disc'], axis=0)

        # 生成统计表格
        stats_df = pd.DataFrame({
            'Category': self.unique_labels,
            'Intra Similarity': [f"{m:.3f}±{s:.3f}" for m, s in zip(intra_means, intra_stds)],
            'Inter Similarity': [f"{m:.3f}±{s:.3f}" for m, s in zip(inter_means, inter_stds)],
            'Discrimination Score': [f"{m:.3f}±{s:.3f}" for m, s in zip(disc_means, disc_stds)]
        }).set_index('Category')

        print("\n=== 最终统计结果 ===")
        print(stats_df)
        stats_df.to_csv("category_similarity_stats.csv")

        # 绘制热图
        plt.figure(figsize=(15, 12))
        sns.heatmap(
            np.nanmean(self.metrics['heatmap'], axis=2),
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=self.unique_labels,
            yticklabels=self.unique_labels
        )
        plt.title(f"Category Similarity Heatmap (Averaged over {self.n_trials} trials)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("category_similarity_heatmap.png", dpi=300)
        plt.close()

        # 绘制分布图
        plt.figure(figsize=(10, 6))
        sns.kdeplot(intra_means, label='Intra-class')
        sns.kdeplot(inter_means, label='Inter-class')
        plt.title("Similarity Distribution Across Categories")
        plt.xlabel("Cosine Similarity")
        plt.legend()
        plt.savefig("similarity_distribution.png", dpi=150)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="执行大规模分类相似度分析")
    parser.add_argument("--embedding", required=True, help="嵌入向量文件路径（Parquet格式）")
    parser.add_argument("--label", required=True, help="标签文件路径（JSON格式）")
    args = parser.parse_args()

    analyzer = CategorySimilarityAnalyzer(
        embedding_path=args.embedding,
        label_path=args.label,
        n_trials=1000,
        sample_ratio=0.001
    )
    
    analyzer.run_experiment()
    analyzer.report_results()