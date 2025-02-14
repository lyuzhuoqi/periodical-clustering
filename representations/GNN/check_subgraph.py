import torch
import os
import random

# 设置文件路径
data_path = "/home/lyuzhuoqi/projects/clustering/data/2010s/GNN/samples"

# 获取所有.pt文件
pt_files = [f for f in os.listdir(data_path) if f.endswith('.pt')]

# 随机选择一个文件
if pt_files:
    random_file = random.choice(pt_files)
    file_path = os.path.join(data_path, random_file)
    
    # 加载图数据
    graph_data = torch.load(file_path, weights_only=False)
    
    # 打印图的基本信息
    print(f"Loading file: {random_file}")
    print("\nGraph data keys:", graph_data.keys())
    
    # 如果有边的信息，检查是否有权重
    if hasattr(graph_data, 'edge_attr'):
        print("\nEdge attributes shape:", graph_data.edge_attr.shape)
        print("Edge attributes sample:", graph_data.edge_attr[:5])
    else:
        print("\nNo edge attributes found")
    
    # 打印边的信息
    if hasattr(graph_data, 'edge_index'):
        print("\nEdge index shape:", graph_data.edge_index.shape)
        print("First few edges:", graph_data.edge_index[:, :5])
else:
    print("No .pt files found in the directory")