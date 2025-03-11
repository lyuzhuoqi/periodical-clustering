import os
import logging
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics import ProgressBar

from gensim.models import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import (adjusted_mutual_info_score, 
                             v_measure_score,
                             adjusted_rand_score,
                             fowlkes_mallows_score)

import seaborn as sns
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')

class TopicLabelConsistencyAnalyzer:
    def __init__(self, 
                 data_dir: str,
                 num_topics: int = 40,
                 passes: int = 1,
                 iterations: int = 50,
                 random_state: int = 42):
        """
        初始化主题标签一致性分析器
        """
        self.data_dir = data_dir
        self.num_topics = num_topics
        self.passes = passes
        self.iterations = iterations
        self.random_state = random_state
        
        # 数据存储
        self.abstracts = {}
        self.label_sets = {}
        self.ddf = None
        
        # 模型和词典
        self.dictionary = None
        self.lda_model = None
        
        # 结果存储
        self.topic_distributions = None
        self.topic_words = None
        self.consistency_metrics = {}
        
        # 标签方法映射
        self.label_to_method = {
            'kmeans_label': 'Periodical2Vec+$k$-means',
            'skm_label': 'Periodical2Vec+Sperical $k$-means',
            'movmf_label': 'Periodical2Vec+movMF',
            'bert_kmeans_label': 'BERT+$k$-means',
            'n2v_kmeans_label': 'Node2Vec+$k$-means',
            'cm_kmeans_label': 'Citation Matrix+$k$-means',
            'gnn_kmeans_label': 'GNN+$k$-means',
            'scopus_label': 'Scopus'
        }

        # 添加停用词相关
        nltk.download('stopwords')  # 下载停用词
        self.stop_words = set(stopwords.words('english'))
        
        # 设置日志
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO
        )

        pbar = ProgressBar()
        pbar.register()


    def prepare_dataset(self, only_label=True, save_dir: Optional[str] = None) -> pd.DataFrame:
        """
        准备数据集，通过减少重复计算和使用更高效的数据结构来提高性能
        使用isalnum()和nltk的停用词，保留计算总行数
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_suffix = 'with_tokens' if not only_label else 'only_labels'
            dataset_path = os.path.join(save_dir, f'processed_dataset_{file_suffix}.parquet')
            
            # 检查文件是否存在
            if os.path.exists(dataset_path):
                logging.info(f"找到已处理的数据集文件，正在加载: {dataset_path}")
                try:
                    self.ddf = dd.read_parquet(dataset_path)
                    logging.info("数据集加载成功")
                    return self.ddf
                except Exception as e:
                    logging.warning(f"加载数据集失败: {str(e)}")
                    logging.warning("将重新处理数据集")
        
        logging.info('Preparing dataset...')
        logging.info("Processing abstracts...")
        abstracts_path = os.path.join(self.data_dir, 
                                    '2010s/classification_tasks/abstracts/paper_abstracts.json')
        self.ddf = dd.read_json(abstracts_path, lines=True, 
                                blocksize=2**28, meta={'PaperID': 'int', 'abstract': 'str'})
        # 定义需要读取的列
        if only_label:
            # 只读取PaperID列，节省内存
            columns_to_read = ['PaperID']
        else:
            # 读取PaperID和abstract列
            columns_to_read = ['PaperID', 'abstract']
        # 只保留需要的列以节省内存
        self.ddf = self.ddf[columns_to_read]
        
        if not only_label:
            logging.info("Tokenization...")

            # 定义Dask可并行执行的token处理函数
            def tokenize_abstract(row, stop_words: List):
                """处理单个文档的抽象并返回tokens"""                
                abstract_lower = row['abstract'].lower()
                tokens = []
                for token in word_tokenize(abstract_lower):
                    if (len(token) > 2 and 
                        token not in stop_words and
                        token.isalnum()):
                        tokens.append(token)
                return tokens
            
            # Use meta=('tokens', object) so that the output is correctly detected as an object type
            self.ddf['tokens'] = self.ddf.apply(tokenize_abstract, axis=1, 
                                                stop_words=self.stop_words, meta=('tokens', object))
            
            # 触发tokens计算（仅在此处触发一次计算）
            self.ddf = self.ddf.persist()
            # 删除原始abstract列，以节省内存
            self.ddf = self.ddf.drop('abstract', axis=1)
        
        # 加载标签
        labels_dir = os.path.join(self.data_dir, '2010s/classification_tasks/labels')
        logging.info("Loading label files...")
        
        # 预加载所有标签文件名，只处理JSON文件
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.json')]
        
        for label_file in label_files:
            label_name = label_file.replace('.json', '')
            logging.info(f"Loading {label_name}...")
            
            label_ddf = dd.read_json(os.path.join(labels_dir, label_file), lines=True,
                                     meta={'PaperID': 'int', label_name: 'str'})
            self.ddf = self.ddf.merge(label_ddf, on='PaperID', how='left')
            # Avoid triggering compute here to keep the processing lazy
        
        # 保存处理后的数据集
        if save_dir:
            logging.info("Saving processed dataset as parquet file...")
            self.ddf.to_parquet(dataset_path)
            logging.info(f"数据集已保存到: {dataset_path}")
        
        return self.ddf


    def select_optimal_topics(self,
                            topic_numbers: range = range(20, 201, 20),
                            save_dir: Optional[str] = None) -> Dict:
        """
        评估不同主题数量的模型表现，使用优化的参数配置并实时保存结果
        """        
        if self.ddf is None:
            self.prepare_dataset(only_label=False)
        
        # 确保save_dir存在
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # 提前准备语料库和texts
        logging.info('准备词典...')
        # For operations expected to be in-memory, compute the tokens column once.
        tokens_series = self.ddf['tokens'].compute()
        if self.dictionary is None:
            self.dictionary = Dictionary(tokens_series)
            self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        
        # 将corpus和texts预先准备好
        logging.info('准备corpus...')
        corpus = [self.dictionary.doc2bow(text) for text in tokens_series]
        texts = tokens_series.tolist()
        
        results = []
        logging.info("开始评估不同主题数量...")
        
        # 创建实时结果文件
        if save_dir:
            results_file = os.path.join(save_dir, f'topic_number_results.csv')
            fig_file = os.path.join(save_dir, f'topic_number_evaluation.png')
            
            # 写入CSV头
            pd.DataFrame(columns=['num_topics', 'coherence']).to_csv(
                results_file, index=False
            )
        
        for num_topics in tqdm(topic_numbers):
            model = LdaModel(
                corpus=corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                passes=self.passes,
                iterations=self.iterations,
                eval_every=None,        # 禁用评估
                per_word_topics=False,  # 禁用词主题分布
                random_state=self.random_state
            )
            
            # 计算一致性
            coherence_model = CoherenceModel(
                model=model,
                texts=texts,
                dictionary=self.dictionary,
                coherence='c_v',
                processes=8  # 减少进程数
            )
            coherence = coherence_model.get_coherence()
            
            result = {
                'num_topics': num_topics,
                'coherence': coherence
            }
            results.append(result)
            
            # 实时保存结果到CSV
            if save_dir:
                pd.DataFrame([result]).to_csv(
                    results_file, 
                    mode='a', 
                    header=False, 
                    index=False
                )
                
                # 更新并保存图表
                results_df = pd.DataFrame(results)
                plt.figure(figsize=(10, 5))
                plt.plot(results_df['num_topics'], 
                         results_df['coherence'], 
                         marker='o')
                plt.xlabel('Number of Topics')
                plt.ylabel('Coherence Score')
                plt.title('Topic Coherence vs Number of Topics')
                plt.grid(True)
                plt.savefig(fig_file)
                plt.close()
        
        # 完成后的处理
        results_df = pd.DataFrame(results)
        best_coherence_idx = results_df['coherence'].idxmax()
        
        optimal_results = {
            'best_num_topics': results_df.loc[best_coherence_idx, 'num_topics'],
            'best_coherence_score': results_df.loc[best_coherence_idx, 'coherence'],
            'all_results': results_df
        }
        
        logging.info(f"\n最优主题数量评估结果:")
        logging.info(f"最佳主题数量: {optimal_results['best_num_topics']} "
                     f"(coherence = {optimal_results['best_coherence_score']:.4f})")
        
        return optimal_results


    def train_lda_model(self, save_dir: Optional[str] = None) -> None:
        """训练LDA模型或加载已有模型"""
        if self.ddf is None:
            self.prepare_dataset()
            
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'lda_model_{self.num_topics}topics.model')
            dict_path = os.path.join(save_dir, f'lda_model_{self.num_topics}topics.model.id2word')
            
            # 检查模型文件是否存在
            if os.path.exists(model_path) and os.path.exists(dict_path):
                try:
                    logging.info(f"找到已有模型文件，正在加载...")
                    self.lda_model = LdaMulticore.load(model_path)
                    self.dictionary = Dictionary.load(dict_path)
                    logging.info(f"模型加载成功：{model_path}")
                    return
                except Exception as e:
                    logging.warning(f"加载模型失败: {str(e)}")
                    logging.warning("将重新训练模型")
        
        logging.info("开始训练新的LDA模型...")
        
        # 创建词典
        logging.info('准备词典...')
        tokens_series = self.ddf['tokens'].compute()
        self.dictionary = Dictionary(tokens_series)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        
        # 转换为BOW格式
        logging.info('准备corpus...')
        corpus = [self.dictionary.doc2bow(text) for text in tokens_series]
        
        # 训练模型
        self.lda_model = LdaMulticore(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            iterations=self.iterations,
            random_state=self.random_state,
            workers=8,
            eval_every=None,        # 禁用评估
            per_word_topics=False,  # 禁用词主题分布
            batch=False,            # 禁用批处理
        )
        
        # 保存模型和词典
        if save_dir:
            self.lda_model.save(model_path)
            self.dictionary.save(dict_path)
            logging.info(f"模型已保存到: {model_path}")
            logging.info(f"词典已保存到: {dict_path}")


    def analyze_vocabulary(self) -> Dict:
        """分析词汇统计信息"""
        if self.dictionary is None:
            raise ValueError("Dictionary not created yet. Run train_lda_model first.")
            
        vocab_stats = {
            'total_words': len(self.dictionary),
            'word_frequencies': sorted(
                [
                    (self.dictionary[id], freq)
                    for id, freq in self.dictionary.cfs.items()
                ],
                key=lambda x: x[1],
                reverse=True
            )[:50]  # 显示top 50词频
        }
        
        logging.info(f"词典大小: {vocab_stats['total_words']}")
        logging.info("Top 50 最常见的词:")
        for word, freq in vocab_stats['word_frequencies']:
            logging.info(f"{word}: {freq}")
            
        return vocab_stats


    def get_topic_distributions(self) -> np.ndarray:
        """获取文档-主题分布"""
        logging.info("Extracting topic distributions...")
        
        tokens_series = self.ddf['tokens'].compute()
        n_docs = len(tokens_series)
        doc_topics = np.zeros((n_docs, self.num_topics))
        
        logging.info("准备corpus...")
        corpus = [self.dictionary.doc2bow(text) for text in tokens_series]
        
        for i, bow in enumerate(tqdm(corpus, desc='Extracting topic distributions', unit='documents')):
            topic_dist = self.lda_model.get_document_topics(bow, minimum_probability=0)
            for topic_id, prob in topic_dist:
                doc_topics[i, topic_id] = prob
                
        self.topic_distributions = doc_topics
        return doc_topics
    

    def visualize_topic_label_matrix(self,
                                     label_type: str,
                                     metrics: Dict,
                                     save_dir: Optional[str] = None,
                                     figsize: Tuple[int, int] = (8, 8),
                                     cmap: str = 'YlOrRd',
                                     fmt: str = '.2f',
                                     cluster: bool = True) -> None:
        """
        可视化主题-标签矩阵，优化大规模主题和标签的显示
        """
        matrix = metrics['topic_label_matrix']
        
        # 使用clustermap并优化布局
        g = sns.clustermap(
            matrix,
            cmap=cmap,
            annot=False,
            fmt=fmt,
            figsize=figsize,
            xticklabels=False,  # 不显示x轴标签
            yticklabels=False,  # 不显示y轴标签
            row_cluster=True,
            col_cluster=True
        )
        
        # 调整标签大小和间距
        g.ax_heatmap.set_xlabel('Labels', fontsize=12)
        g.ax_heatmap.set_ylabel('Topics', fontsize=12)
        
        # 保存完整的聚类图
        if cluster and save_dir:
            filename = os.path.join(save_dir, f'topic_label_matrix_{label_type}.pdf')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logging.info(f"聚类图已保存至: {filename}")


    def analyze_all_label_types(self, label_to_method: dict, save_dir: Optional[str] = None) -> Dict:
        """分析所有标签类型"""
        if self.ddf is None:
            self.prepare_dataset(only_label=True, save_dir=save_dir)
            
        # 检查模型
        if self.lda_model is None:
            self.train_lda_model(save_dir)
        
        # 检查主题分布文件
        topic_dist_path = None
        if save_dir:
            topic_dist_path = os.path.join(save_dir, f'topic_distributions_{self.num_topics}topics.npy')
            if os.path.exists(topic_dist_path):
                try:
                    logging.info(f"找到已有主题分布文件，正在加载...")
                    self.topic_distributions = np.load(topic_dist_path)
                    logging.info("主题分布加载成功")
                except Exception as e:
                    logging.warning(f"加载主题分布失败: {str(e)}")
                    logging.warning("将重新计算主题分布")
                    self.topic_distributions = None
        
        # 计算主题分布
        if self.topic_distributions is None:
            logging.info("计算文档主题分布...")
            self.topic_distributions = self.get_topic_distributions()
            # 保存主题分布
            if topic_dist_path:
                np.save(topic_dist_path, self.topic_distributions)
                logging.info(f"主题分布已保存到: {topic_dist_path}")
        
        # 获取主题分配
        topic_assignments = np.argmax(self.topic_distributions, axis=1)
        
        results = {}
        summary_data = []
        
        # 遍历每种标签类型进行分析
        for label_type in label_to_method.keys():
            logging.info(f"\nAnalyzing {label_type}...")
            
            # 获取标签，转换为 in-memory 对象
            labels = self.ddf[label_type].compute().tolist()
            
            # 计算各种一致性指标
            nmi = normalized_mutual_info_score(labels, topic_assignments)
            ami = adjusted_mutual_info_score(labels, topic_assignments)
            v_measure = v_measure_score(labels, topic_assignments)
            ari = adjusted_rand_score(labels, topic_assignments)
            fmi = fowlkes_mallows_score(labels, topic_assignments)
            
            # 计算主题-标签对应关系
            unique_labels = sorted(self.ddf[label_type].unique().compute().tolist())
            topic_label_matrix = np.zeros((self.num_topics, len(unique_labels)))
            
            # 使用numpy的高效操作计算主题-标签矩阵
            for doc_idx, (topic_idx, label) in enumerate(zip(topic_assignments, labels)):
                label_idx = unique_labels.index(label)
                topic_label_matrix[topic_idx, label_idx] += 1
            
            # 标准化主题-标签矩阵
            row_sums = topic_label_matrix.sum(axis=1, keepdims=True)
            topic_label_matrix = np.divide(topic_label_matrix, row_sums, where=row_sums != 0)
            
            # 存储结果
            metrics = {
                'normalized_mutual_information': nmi,
                'adjusted_mutual_information': ami,
                'v_measure': v_measure,
                'adjusted_rand_index': ari,
                'fowlkes_mallows_index': fmi,
                'topic_label_matrix': topic_label_matrix,
                'unique_labels': unique_labels
            }
            
            results[label_type] = metrics
            
            # 可视化并保存结果
            self.visualize_topic_label_matrix(label_type, metrics, save_dir)
            
            # 收集总结数据
            summary_data.append({
                'Label Type': label_to_method[label_type],
                'NMI': nmi,
                'AMI': ami,
                'V-measure': v_measure,
                'ARI': ari,
                'FMI': fmi
            })
        
        # 创建和保存总结表格
        summary_df = pd.DataFrame(summary_data)
        if save_dir:
            summary_df.to_csv(os.path.join(save_dir, 'consistency_metrics_summary.csv'), index=False)
        
        return results

# 初始化分析器
analyzer = TopicLabelConsistencyAnalyzer(data_dir='/home/lyuzhuoqi/projects/clustering/data',
                                         passes=1,
                                         iterations=50)

# generate or load dataset
analyzer.prepare_dataset(only_label=False, 
                         save_dir='/home/lyuzhuoqi/projects/clustering/data/2010s/topic_consistency')