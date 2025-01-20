import pickle

# 加载Pickle文件
print("正在加载Pickle文件...")
with open('filtered_walks.pkl', 'rb') as f:
    walks = pickle.load(f)
print("Pickle文件加载完成。")

# 设定每处理5000条walk输出一次信息
batch_size = 5000
processed_count = 0

# 遍历walks，转换每个PID为int
for walk in walks:
    for i in range(len(walk)):
        walk[i] = int(walk[i])

    # 更新处理计数器
    processed_count += 1

    # 每处理5000条输出一次进度信息
    if processed_count % batch_size == 0:
        print(f"已处理 {processed_count} 条walks")

with open('citation_trails.pkl', 'wb') as f:
    pickle.dump(walks, f)

print("PID转换完成，已保存修改后的Pickle文件。")
