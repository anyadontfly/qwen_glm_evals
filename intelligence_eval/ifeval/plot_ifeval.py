import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys

# Define base directory relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")

if not os.path.exists(results_dir):
    print(f"Error: Results directory not found at {results_dir}")
    sys.exit(1)

# Map folder names to display names if desired, or just use folder names
# Structure seems to be results/<model_alias>/<model_name_sanitized>/results_*.json
# We can walk the directory to find all result files.

data = {}

print(f"Scanning results in {results_dir}...")

# Walk through the directory to find all json files
json_files = []
for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file.startswith("results_") and file.endswith(".json"):
            json_files.append(os.path.join(root, file))

# Group files by model (parent folder name or model name inside json)
# Let's try to group by the model alias which is the folder under results/
# e.g. results/qwen35_9b/...

model_results = {}

for json_file in json_files:
    # Get the model alias from the path
    # Path: .../results/<model_alias>/<sanitized_name>/results.json
    rel_path = os.path.relpath(json_file, results_dir)
    parts = rel_path.split(os.sep)
    if len(parts) < 2:
        continue
    
    model_alias = parts[0]
    
    # Read the JSON file
    try:
        with open(json_file, 'r') as f:
            result_data = json.load(f)
            
        # Check if it has the expected structure
        if 'results' not in result_data or 'ifeval' not in result_data['results']:
            print(f"Skipping {json_file}: 'results.ifeval' not found")
            continue
            
        # Get timestamp or file mtime to pick the latest if multiple exists
        mtime = os.path.getmtime(json_file)
        
        if model_alias not in model_results or mtime > model_results[model_alias]['mtime']:
             model_results[model_alias] = {
                 'mtime': mtime,
                 'file': json_file,
                 'data': result_data['results']['ifeval']
             }
             
    except Exception as e:
        print(f"Error reading {json_file}: {e}")

if not model_results:
    print("No valid result files found.")
    sys.exit(0)

# Extract metrics
models = []
prompt_accs = []
inst_accs = []
avg_accs = []

# Map aliases to nice names
display_names = {
    'qwen35_9b': 'Qwen3.5-9B',
    'qwen35_27b': 'Qwen3.5-27B', 
    'glm47_flash': 'GLM-4.7-Flash'
}

for alias, info in model_results.items():
    metrics = info['data']
    
    # Keys might have ,none or not depending on lm-evaluation-harness version
    # The user provided file has ,none
    prompt_acc = metrics.get('prompt_level_strict_acc,none')
    inst_acc = metrics.get('inst_level_strict_acc,none')
    
    if prompt_acc is None:
        prompt_acc = metrics.get('prompt_level_strict_acc')
    if inst_acc is None:
        inst_acc = metrics.get('inst_level_strict_acc')
        
    if prompt_acc is None or inst_acc is None:
        print(f"Warning: Metrics not found for {alias}")
        continue
        
    name = display_names.get(alias, alias)
    models.append(name)
    avg_acc = (prompt_acc + inst_acc) / 2
    prompt_accs.append(prompt_acc)
    inst_accs.append(inst_acc)
    avg_accs.append(avg_acc)
    print(f"Loaded {name}: Prompt Acc={prompt_acc:.4f}, Inst Acc={inst_acc:.4f}, Avg Acc={avg_acc:.4f}")

# Sort by model name for consistency
sorted_indices = np.argsort(models)
models = [models[i] for i in sorted_indices]
avg_accs = [avg_accs[i] for i in sorted_indices]

# Plotting
x = np.arange(len(models))
width = 0.5  # Wider bars since we only have one per model now

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x, avg_accs, width, label='Average Strict Accuracy', color='#4daf4a')

ax.set_ylabel('Average Accuracy')
ax.set_title('IFEval Average Strict Accuracy by Model')
ax.set_xticks(x)
ax.set_xticklabels(models)
# ax.legend()  # No legend needed for single series, but can keep if desired
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)

fig.tight_layout()

output_file = os.path.join(script_dir, 'ifeval_accuracy_comparison.png')
plt.savefig(output_file)
print(f"\nPlot saved to {output_file}")
