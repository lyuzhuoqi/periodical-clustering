from itertools import islice
import pickle

with open("citation_trails.pkl", "rb") as f:
    citation_trail = pickle.load(f)

special_tokens = {'<pad>': 0, '<mask>': 1}
print("Start generating the PID set...")
all_pids = set(pid for trail in citation_trail for pid in trail)
print("Finished.")

print("Start generating PID to index mapping...")
pid_to_idx = {**special_tokens, **{pid: idx + len(special_tokens) for idx, pid in enumerate(sorted(all_pids))}}
num_pids = len(pid_to_idx)
print("Finished. PID to model input IDs:", dict(islice(pid_to_idx.items(), 10)))  # 仅打印前10项
print(f"Total number of unique PIDs (plus padding value): {num_pids}")

with open("pid_to_idx.pkl", "wb") as f:
    pickle.dump(pid_to_idx, f)