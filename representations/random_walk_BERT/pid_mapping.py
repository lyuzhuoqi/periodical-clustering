from itertools import islice
import pickle
from collections import Counter
import argparse

def main(min_count=50):
    # 加载 citation_trail 数据
    print("Loading citation trails...")
    with open("citation_trails.pkl", "rb") as f:
        citation_trail = pickle.load(f)
    print("Start generating the PID set...")
    print(f"Total number of trails: {len(citation_trail)}")
    # 统计每个 PID 出现的次数
    print("Counting the number of occurrences of each PID...")
    all_pids = [pid for trail in citation_trail for pid in trail]
    pid_counts = Counter(all_pids)

    # 过滤掉出现次数少于 min_count 的 PID
    print("Filtering out PIDs that appear less than min_count times...")
    filtered_pids = [pid for pid, count in pid_counts.items() if count >= min_count]
    print(f"Filtered out PIDs that appear less than {min_count} times.")

    # 打印过滤后的 PID 数量
    print(f"Total number of PIDs after applying min_count filter: {len(filtered_pids)}")

    # 开始生成 PID 到索引的映射
    print("Start generating PID to index mapping...")
    # 预留特殊 token 的位置
    special_tokens = {'<pad>': 0, '<mask>': 1}
    # 生成 PID 到索引的映射，仅包含高频的 PID
    pid_to_idx = {**special_tokens, **{pid: idx + len(special_tokens) for idx, pid in enumerate(sorted(filtered_pids))}}
    # 输出 PID 到索引映射的前10项
    print("Finished. PID to model input IDs:", dict(islice(pid_to_idx.items(), 10)))
    # 输出总共有多少个 PID（包括特殊 token）
    num_pids = len(pid_to_idx)
    print(f"Total number of unique PIDs (plus 2 special token): {num_pids}")

    # 保存 PID 到索引映射
    with open(f"pid_to_idx_min_count_{min_count}.pkl", "wb") as f:
        pickle.dump(pid_to_idx, f)
    print("PID to index mapping saved.")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Generate PID to index mapping")
    parser.add_argument('--min_count', type=int, default=50, help='Minimum count for a PID to be included in the mapping')
    args = parser.parse_args()

    # 调用 main 函数并传递参数
    main(args.min_count)